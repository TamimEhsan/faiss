# How FAISS handles add & delete: IVF-PQ, IVF-RaBitQ, and HNSW

Scope: vector insertion (`add` / `add_with_ids`) and deletion (`remove_ids`,
`update_vectors`) for the IVF family (IVF-PQ, IVF-RaBitQ) and the HNSW family
(incl. quantized `HNSWPQ` / `HNSWSQ`). All file:line refs are against this
checkout.

---

## TL;DR

| Index | Add | Delete (`remove_ids`) |
|---|---|---|
| **IVF-PQ** | ✅ coarse-assign → residual PQ-encode → append to inverted list | ✅ supported (swap-with-last in lists), with `DirectMap` caveats |
| **IVF-RaBitQ** | ✅ same IVF flow; RaBitQ encodes residual vs. centroid | ✅ inherited from `IndexIVF`, identical mechanics to IVF-PQ |
| **HNSW (Flat/PQ/SQ/…)** | ✅ append to `storage` index + incrementally wire graph | ❌ **not supported** — throws `"remove_ids not implemented…"` |

The deep difference: **IVF deletion is cheap because the structure is a flat
set of per-cluster lists**; **HNSW deletion is unimplemented because graph edges
are positional references and removal would require global graph repair.**

**Centroids** (IVF coarse list centers, and PQ codebooks) are learned **only at
`train()`** and never change on add/remove — see §2.

---

## 1. IVF family — shared add/delete machinery

Both IVF-PQ and IVF-RaBitQ derive from `IndexIVF` and reuse the same three
components:

- **Coarse quantizer** → which inverted list a vector belongs to.
- **`InvertedLists`** → per-list storage of `(id, code)` pairs.
- **`DirectMap`** → optional `external id → (list_no, offset)` map enabling
  reconstruct/remove/update.

### 1.1 Add path

`IndexIVF::add` → `add_with_ids` → `add_core` (`faiss/IndexIVF.cpp:187-288`):

1. **Coarse assignment** — `quantizer->assign(n, x, coarse_idx)` maps each
   vector to a list in `[0, nlist)` (`IndexIVF.cpp:192-195`).
2. **Batching** — adds >65536 are chunked to bound memory (`IndexIVF.cpp:~221`).
3. **Encode** — `encode_vectors(n, x, coarse_idx, flat_codes)`
   (`IndexIVF.cpp:250-251`), an abstract method each subclass implements.
4. **Append** — per entry, `invlists->add_entry(list_no, id, code)` returns the
   append `offset`; `DirectMapAdd` records it. ID is the caller's `xids[i]` or
   auto `ntotal + i` (`IndexIVF.cpp:253-278`). Lists are sharded across threads
   by `list_no % nthreads` to avoid contention.
5. `ntotal += n`.

Per-entry storage in `ArrayInvertedLists` (`faiss/invlists/InvertedLists.h:264-266`,
`InvertedLists.cpp:279-296`): two parallel arrays per list — `ids[list_no]`
(8-byte `idx_t`) and `codes[list_no]` (`code_size` bytes). `add_entries` simply
`resize`s and `memcpy`s onto the end and returns the starting offset.

### 1.2 Delete path — `remove_ids`

`IndexIVF::remove_ids` (`faiss/IndexIVF.cpp:1252-1256`) delegates to
`DirectMap::remove_ids(sel, invlists)` (`faiss/invlists/DirectMap.cpp:157-231`),
then decrements `ntotal`. Behavior depends on the **DirectMap mode**:

- **`NoMap` (default)** — exhaustive parallel scan of every list. For each list,
  a two-pointer **swap-with-last** compaction: if `sel.is_member(id)`, overwrite
  it with the list's last entry (`update_entry`) and shrink; else advance. Then
  `resize` each list down (`DirectMap.cpp:163-190`). O(ntotal).
- **`Hashtable`** — direct lookup per id, swap-with-last, fix the moved entry's
  hashtable record, `resize` (`DirectMap.cpp:191-227`). **Only works with
  `IDSelectorArray`** (throws otherwise) and **not** with `BlockInvertedLists`.
- **`Array`** — `FAISS_THROW_MSG("remove not supported with this direct_map
  format")` (`DirectMap.cpp:~228`).

**Key consequence — internal positions are not stable.** Swap-with-last means a
surviving vector can move to a different offset after a delete. External ids stay
valid (that's what `DirectMap` tracks), but anything relying on raw list offsets
must be refreshed.

### 1.3 DirectMap modes (`faiss/invlists/DirectMap.h:38-49`)

- `NoMap` — no id map (default). Reconstruct/update unavailable; remove works via
  full scan.
- `Array` — `vector<idx_t>` indexed by id; requires **sequential ids** (i.e.
  `add` without explicit ids). O(1) lookup, but **remove is disallowed**.
- `Hashtable` — `unordered_map<id, lo>` for arbitrary ids; supports remove
  (with `IDSelectorArray`) and update.

`lo` packs `(list_no, offset)` into a uint64: `lo_build = list<<32 | offset`
(`DirectMap.h:23-33`). Enable via `make_direct_map()` (→ Array) or
`set_direct_map_type(Hashtable)` (`IndexIVF.cpp:290-300`).

### 1.4 Update — `update_vectors` (`faiss/IndexIVF.cpp:1258-1283`)

Requires a DirectMap. `Hashtable`: remove-then-add. `Array`: in-place via
`DirectMap::update_codes` (re-assign list, swap-out old, append new) so the
sequential id space stays hole-free.

### 1.5 InvertedLists backends and remove support

| Backend | Add | Remove |
|---|---|---|
| `ArrayInvertedLists` (RAM) | ✅ | ✅ swap-with-last |
| `OnDiskInvertedLists` (mmap) | ✅ | ✅ but slow (shrinks); poor under parallel |
| `BlockInvertedLists` (fast-scan) | ✅ | ✅ via own `remove_ids`; **incompatible with Hashtable removal** |
| `ReadOnlyInvertedLists` | ❌ throws | ❌ throws |

---

## 2. How centroids are computed & when (IVF `train`)

"Centroids" in IVF means two **separately learned** things, both computed during
`index.train(xt)` — **never** during `add`/`remove`.

### 2.1 Coarse centroids (the IVF list centers)

The `nlist` cluster centers that define the inverted lists, learned by **k-means**
in `Level1Quantizer::train_q1` (`faiss/IndexIVF.cpp:56-122`):

```cpp
Clustering clus(d, nlist, cp);
clus.train(n, x, *quantizer);     // k-means over the training set
// clus.centroids are then stored as vectors inside the coarse quantizer
```

- IVF lowers k-means iterations to **`niter = 10`** (vs. the generic default 25),
  because `nlist` is usually large (`IndexIVF.cpp:40-46`).
- Centroids are stored *as vectors* inside the coarse quantizer (an `IndexFlat`),
  so centroid `i` is the anchor of list `i`.
- Training subsamples to `max_points_per_centroid = 256` per centroid and warns
  below `min_points_per_centroid = 39` (`faiss/Clustering.h:44-46`).
- **Reuse / skip:** if the quantizer is already trained with `ntotal == nlist`,
  `train_q1` skips k-means and uses your centroids as-is (`IndexIVF.cpp:63-67`).

### 2.2 Encoder codebooks — differ by index type

`IndexIVF::train` (`faiss/IndexIVF.cpp:1285-1318`):

1. `train_q1(...)` → coarse centroids (above).
2. With `by_residual = true` (default for IVF-PQ and IVF-RaBitQ): assign each
   training vector to its nearest coarse centroid, compute the **residual**
   `x − centroid`, then `train_encoder(residuals)`.

- **IVF-PQ** (`IndexIVFPQ::train_encoder`, `IndexIVFPQ.cpp:73-92`):
  `pq.train(residuals)` learns the **PQ sub-centroids** — for each of `M`
  subspaces, `ksub = 2^nbits` k-means centroids — then `precompute_table()`. So
  IVF-PQ has coarse centroids **and** PQ codebooks, both trained on residuals.
- **IVF-RaBitQ** (`IndexIVFRaBitQ::train_encoder`, `IndexIVFRaBitQ.cpp:49-54`):
  `rabitq.train(...)` is a **no-op** (`RaBitQuantizer::train` does nothing).
  RaBitQ has no learned codebook — it sign-bit-quantizes the residual against the
  coarse centroid. **The coarse k-means centroids are the only learned centroids.**

### 2.3 When (lifecycle)

- Computed **at `train()` only**; fixed thereafter.
- `add` *uses* them (residual vs. nearest centroid) but never updates them;
  `remove_ids` doesn't touch them. Distribution drift ⇒ retrain/rebuild to
  refresh centroids.

### 2.4 Python: where centroids come from and how to read them

```python
import faiss
import numpy as np

rng = np.random.default_rng(0)
d, nlist, nb = 32, 16, 5000
xt = rng.random((20000, d), dtype="float32")   # training set
xb = rng.random((nb, d), dtype="float32")       # database

# ---------- IVF-PQ ----------  (M=8 subquantizers, nbits=8 -> ksub=256)
quantizer = faiss.IndexFlatL2(d)
ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, 8, 8, faiss.METRIC_L2)

print("trained?", ivfpq.is_trained)            # False
ivfpq.train(xt)                                 # <-- centroids computed HERE
print("trained?", ivfpq.is_trained)            # True
ivfpq.add(xb)                                   # uses centroids, doesn't change them

# 1) Coarse centroids (nlist IVF list centers), shape (nlist, d)
coarse = ivfpq.quantizer.reconstruct_n(0, ivfpq.nlist)
print("coarse centroids:", coarse.shape)        # (16, 32)

# 2) PQ codebooks (sub-centroids), shape (M, ksub, dsub)
pq = ivfpq.pq
codebooks = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)
print("PQ codebooks:", codebooks.shape)         # (8, 256, 4)

# ---------- IVF-RaBitQ ----------
quantizer2 = faiss.IndexFlatL2(d)
ivfrq = faiss.IndexIVFRaBitQ(quantizer2, d, nlist, faiss.METRIC_L2)
ivfrq.train(xt)                                 # only coarse k-means; RaBitQ.train is a no-op
ivfrq.add(xb)
coarse_rq = ivfrq.quantizer.reconstruct_n(0, ivfrq.nlist)
print("IVF-RaBitQ coarse centroids:", coarse_rq.shape)   # (16, 32) — only learned centroids

# ---------- Reuse your own coarse centroids (skip k-means) ----------
my_centroids = xt[:nlist].copy()                # any nlist x d
q = faiss.IndexFlatL2(d); q.add(my_centroids)   # pre-filled -> is_trained, ntotal == nlist
ivf2 = faiss.IndexIVFPQ(q, d, nlist, 8, 8)
ivf2.train(xt)                                  # train_q1 skipped; PQ still trained on residuals
```

### 2.5 Cluster balance over time

Because centroids are frozen at `train()`, lists **can drift out of balance** as
you mutate the index — one cluster grows huge while others stay tiny or go empty.
This is expected, not a bug, and faiss does **not** auto-correct it.

**Causes.** `add` routes each vector to its nearest *existing* centroid; `remove_ids`
deletes from whatever list held the vector. Neither moves centroids. Imbalance
appears when live data ≠ training distribution: a non-representative training
sample, concept drift in added data, naturally non-uniform density, or deletes
that empty specific lists. `nlist` is constant — an emptied list keeps its
centroid at size 0.

**What faiss does: measure, not fix.** There is no rebalancing, list splitting, or
list merging anywhere in the IVF/InvertedLists code. The only tooling is
measurement:

- `InvertedLists::imbalance_factor()` (`faiss/invlists/InvertedLists.cpp:184-192`):
  `nlist·Σ(sizeᵢ²)/(Σsizeᵢ)²`. `1.0` = perfectly balanced; `>1` ≈ how much bigger
  the average *scanned* list is than ideal.
- `InvertedLists::print_stats()` (`:194-211`): power-of-two histogram of list sizes.

```python
print(index.invlists.imbalance_factor())   # ~1.0 good; 3.0 ≈ 3x worse than ideal
index.invlists.print_stats()
```

**Why it matters.** Search scans every vector in the `nprobe` probed lists. A query
hitting a giant list scans far more than budgeted (slower, and the result heap is
dominated by that cluster); tiny/empty lists hurt recall. Expected scan work
scales with `imbalance_factor`.

**Mitigations (manual — faiss is build-then-serve, no online re-clustering):**

1. **Rebuild/retrain periodically** — re-run `train()` on representative current
   data (recomputes centroids) and re-`add`. The only real re-fit. Standard
   pattern: serve the current index, rebuild in the background, swap.
2. **Train on representative data**; size `nlist ≈ √N` for a healthy average list.
3. **Raise `nprobe`** to recover recall under mild imbalance (doesn't fix scan skew).
4. **Monitor** `imbalance_factor()` and trigger a rebuild past a threshold.

---

## 3. IVF-PQ specifics (`IndexIVFPQ`)

- **Encoding** (`faiss/IndexIVFPQ.cpp:175-197`): when `by_residual` (default
  `true`, `IndexIVF.h:218-220`), compute the residual `x − centroid[list_no]`
  (`compute_residuals`, `IndexIVFPQ.cpp:155-173`) then `pq.compute_codes` on the
  residual. Decode = decoded residual + centroid. `by_residual=false` encodes
  `x` directly.
- **Outliers** (`list_no == -1`): residual zeroed, tracked in DirectMap as `-1`,
  **not** stored in any list (`IndexIVFPQ.cpp:~300`).
- **Precomputed tables** are a *search-time* acceleration derived from the
  centroids/codebook; add/delete just mutate list contents and don't
  special-case them.
- **Delete**: not overridden — uses the generic `IndexIVF` path in §1.2.

---

## 4. IVF-RaBitQ specifics (`IndexIVFRaBitQ`)

- Derives from `IndexIVF`, holds a `RaBitQuantizer rabitq`; constructor sets
  `by_residual = true` and `code_size = rabitq.code_size`
  (`faiss/IndexIVFRaBitQ.cpp:27-43`).
- **Encoding** (`IndexIVFRaBitQ.cpp:56-87`): reconstructs the list centroid from
  the coarse quantizer and calls `rabitq.compute_codes_core(xi, code, 1,
  centroid)` — RaBitQ subtracts the centroid internally, so residual encoding is
  implicit (comment notes `by_residual` and `!by_residual` yield the same code).
- **Add** (`IndexIVFRaBitQ.cpp:110-153`): overrides `add_core` only to encode one
  vector at a time against its centroid; otherwise the same DirectMapAdd +
  `invlists->add_entry` flow as base IVF.
- RaBitQ code layout: sign bits `(d+7)/8` + factor block(s); larger for
  multi-bit (`faiss/impl/RaBitQuantizer.h:51-69`).
- **Delete**: **not overridden** — fully inherits `IndexIVF::remove_ids` (§1.2).
  Mechanically identical to IVF-PQ. (`IndexIVFRaBitQFastScan` stores codes in a
  `BlockInvertedLists`, so the fast-scan remove caveats above apply.)

---

## 5. HNSW family — add yes, delete no

### 5.1 Structure

`IndexHNSW` is a wrapper (`faiss/IndexHNSW.h:31-40`):
- `Index* storage` — holds the actual vectors and provides the
  `DistanceComputer` (`IndexHNSW.cpp:696-698`; `reconstruct` delegates to
  storage, `IndexHNSW.cpp:403-405`).
- `HNSW hnsw` — only the **neighbor-link graph**: `levels`, `offsets`,
  `neighbors` (flat `storage_idx_t` arrays), and an `entry_point`
  (`faiss/impl/HNSW.h:62-127`). A node's HNSW id **is** its positional index in
  `storage`.

### 5.2 Add (`faiss/IndexHNSW.cpp:378-394`, `hnsw_add_vertices` 63-213)

1. Require `is_trained` and a non-null `storage`.
2. `storage->add(n, x)` appends raw/quantized vectors; `ntotal = storage->ntotal`.
3. `hnsw_add_vertices`: assign random levels (`prepare_level_tab`), bucket new
   points by level (highest first for entry-point logic), then in parallel call
   `hnsw.add_with_locks(dist, level, pt_id, locks, vt)` (`impl/HNSW.h:209-215`)
   to greedily find neighbors per level and wire bidirectional edges (with
   per-node locks and neighbor pruning to the M budget).

So "add" = append to storage + incrementally splice the new node into the graph.
No global rebuild.

### 5.3 Delete — unsupported

`IndexHNSW` does **not** override `remove_ids`, so it falls through to
`Index::remove_ids`, which throws
`"remove_ids not implemented for this type of index"`
(`faiss/Index.cpp:56-59`). This holds for **every** HNSW variant
(`IndexHNSWFlat`, `IndexHNSWPQ`, `IndexHNSWSQ`, `IndexHNSW2Level`,
`IndexHNSWCagra`) regardless of storage type.

**Why deletion is hard for HNSW:** edges are stored as `storage_idx_t` positional
ids. Removing a node would require (a) finding and rewriting every *incoming*
edge (the graph keeps no reverse index), (b) possibly re-selecting the
`entry_point` and repairing upper levels, and (c) compacting/remapping storage
positions, which shifts all higher ids and invalidates the entire neighbor array.
HNSW is designed append-only; practical "deletion" is done with an external
tombstone/`IDSelector` filter at search time, or by rebuilding.

### 5.4 Quantized HNSW variants

| Variant | `storage` | Training | Notes |
|---|---|---|---|
| `IndexHNSWFlat` | `IndexFlat(L2)` | none (`is_trained=true`) | full-precision; `IndexHNSW.cpp:708-715` |
| `IndexHNSWPQ` | `IndexPQ` | **required**; `train()` also builds SDC table | `IndexHNSW.cpp:779-793` |
| `IndexHNSWSQ` | `IndexScalarQuantizer` | depends on qtype (`is_trained` copied from storage) | `IndexHNSW.cpp:799-807` |
| `IndexHNSW2Level` | `Index2Layer` | special | `IndexHNSW.cpp:815-823` |

For PQ/SQ variants, **`add` before `train` throws** (the `is_trained` check at
`IndexHNSW.cpp:382`): the storage quantizer can't encode without a learned
codebook. The HNSW graph is then built over the *quantized* codes; graph search
uses the storage's asymmetric distance computer.

Factory strings (`faiss/index_factory.cpp`, `parse_IndexHNSW`): `HNSW32` /
`HNSW32,Flat` → Flat; `HNSW32,PQ64x8` → PQ; `HNSW32,SQ8` → SQ; `HNSW32,100+PQ64`
→ 2-level.

---

## 6. Practical guidance

- **Need deletes/updates?** Use the **IVF** family. For arbitrary ids and
  `remove_ids`, set `set_direct_map_type(DirectMap::Hashtable)` and delete via
  `IDSelectorArray`; or accept the default `NoMap` and pay an O(ntotal) scan with
  a general `IDSelector`. Avoid `Array` mode if you need removal.
- **Expect position churn:** swap-with-last reorders surviving entries; rely on
  external ids, not list offsets.
- **HNSW:** treat as append-only. Emulate deletion with an `IDSelector` filter at
  query time (search now supports selectors on `IndexPQ` too, per the #3559 fix),
  and rebuild periodically to reclaim space.
- **Quantized indexes:** always `train` before `add`.

---

## Appendix A. Exposed interfaces (`IndexIVFPQ`, `IndexIVFRaBitQ`)

Both derive `IndexIVF → Index`, so most of the API is inherited and identical;
only a thin layer is type-specific. C++ names; SWIG exposes the same in Python.
Lists below are generated from the installed `faiss` (`dir(cls)`), de-noised of
SWIG `_c`/`_ex` shims and `thisown`.

### A.1 Shared — from `Index` (generic vector-DB API)

`train`, `add`, `add_with_ids`, `add_sa_codes`, `assign`, `search`, `search1`,
`search_and_reconstruct`, `search_subset`, `range_search`, `remove_ids`,
`reconstruct`, `reconstruct_n`, `reconstruct_batch`, `reset`,
`compute_residual`, `compute_residual_n`, `merge_from`,
`check_compatible_for_merge`, `get_distance_computer`,
`sa_code_size`, `sa_encode`, `sa_decode`.
Fields: `d`, `ntotal`, `is_trained`, `metric_type`, `metric_arg`, `verbose`.
(Persistence is via the free functions `faiss.write_index` / `faiss.read_index`.)

### A.2 Shared — from `IndexIVF` (IVF core)

`add_core`, `encode_vectors`, `decode_vectors`, `encode_listno`,
`decode_listno`, `coarse_code_size`, `train_encoder`,
`train_encoder_num_vectors`, `train_q1`, `search_preassigned`,
`range_search_preassigned`, `search_and_return_codes`,
`get_InvertedListScanner`, `get_CodePacker`, `get_list_size`,
`reconstruct_from_offset`, `update_vectors`, `copy_subset_to`,
`check_ids_sorted`, `make_direct_map`, `set_direct_map_type`,
`replace_invlists`.
Fields: `quantizer`, `nlist`, `nprobe`, `max_codes`, `invlists`, `code_size`,
`by_residual`, `direct_map`, `own_fields`, `own_invlists`, `parallel_mode`,
`quantizer_trains_alone`, `clustering_index`, `cp`.
Diagnostics on `invlists`: `imbalance_factor()`, `print_stats()`.

### A.3 `IndexIVFPQ`-specific

`pq` (the `ProductQuantizer`: `pq.M`, `pq.nbits`, `pq.ksub`, `pq.dsub`,
`pq.centroids`, …), `add_core_o`, `encode`, `encode_multiple`,
`decode_multiple`, `find_duplicates`, `precompute_table`.
Fields: `precomputed_table`, `use_precomputed_table`, `scan_table_threshold`,
`do_polysemous_training`, `polysemous_training`, `polysemous_ht`.
Search params: `IVFPQSearchParameters`.

### A.4 `IndexIVFRaBitQ`-specific

`rabitq` (the `RaBitQuantizer`), and the field `qb` (query-quantization bits;
`0` = raw fp32 query). It also overrides `get_distance_computer`,
`encode_vectors`/`decode_vectors`, `add_core`, `reconstruct_from_offset`,
`sa_decode`.
Search params: `IVFRaBitQSearchParameters` (`qb`, `centered`, + IVF fields).

### A.5 Regenerate this list

```python
import faiss
base = set(dir(faiss.IndexIVF))
for name in ["IndexIVFPQ", "IndexIVFRaBitQ"]:
    cls = getattr(faiss, name)
    own = sorted(m for m in dir(cls)
                 if m not in base and not m.startswith("_"))
    print(name, "specific:", own)
```

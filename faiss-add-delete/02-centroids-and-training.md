# Centroids & training

[← index](README.md)

"Centroids" in IVF means two **separately learned** things, both computed during
`index.train(xt)` — **never** during `add`/`remove`.

## Coarse centroids (the IVF list centers)

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

## Encoder codebooks — differ by index type

`IndexIVF::train` (`faiss/IndexIVF.cpp:1285-1318`):

1. `train_q1(...)` → coarse centroids (above).
2. With `by_residual = true` (default for IVF-PQ and IVF-RaBitQ): assign each
   training vector to its nearest coarse centroid, compute the **residual**
   `x − centroid`, then `train_encoder(residuals)` (`IndexIVF.cpp:1305-1314`).

- **IVF-PQ** (`IndexIVFPQ::train_encoder`, `IndexIVFPQ.cpp:73-92`):
  `pq.train(residuals)` learns the **PQ sub-centroids** — for each of `M`
  subspaces, `ksub = 2^nbits` k-means centroids — then `precompute_table()`. So
  IVF-PQ has coarse centroids **and** PQ codebooks, both trained on residuals.
- **IVF-RaBitQ** (`IndexIVFRaBitQ::train_encoder`, `IndexIVFRaBitQ.cpp:49-54`):
  `rabitq.train(...)` is a **no-op** (`RaBitQuantizer::train` does nothing).
  RaBitQ has no learned codebook — it sign-bit-quantizes the residual against the
  coarse centroid. **The coarse k-means centroids are the only learned centroids.**

## `ivfrq.train()` ≠ `rabitq.train()`

A common confusion: if `rabitq.train` is a no-op, when are centroids computed?
`ivfrq.train(xt)` is `IndexIVF::train`, which is **not** a no-op — its first step
(`train_q1`) runs k-means and produces the coarse centroids. Only the separate
*encoder* substep (`RaBitQuantizer::train`) does nothing.

```
ivfrq.train(xt)  ->  IndexIVF::train          # IndexIVF.cpp:1285
   ├─ train_q1(...)        # k-means -> COARSE CENTROIDS   (runs)   :1290
   ├─ residual = x - nearest centroid                               :1305-1310
   └─ train_encoder(residuals) -> rabitq.train(...)  # NO-OP        :1312
```

## When (lifecycle)

- Computed **at `train()` only**; fixed thereafter.
- `add` *uses* them (residual vs. nearest centroid) but never updates them;
  `remove_ids` doesn't touch them. Distribution drift ⇒ retrain/rebuild to refresh
  centroids.

## Encoding reference point — `by_residual`

`by_residual` (`faiss/IndexIVF.h:220`) chooses what the stored code encodes:

- `by_residual = true` (default): code encodes `x − centroid[list]` — each
  cluster's own centroid (`IndexIVFPQ.cpp:181-191`).
- `by_residual = false`: code encodes `x` directly — i.e. residual against the
  **origin**. The k coarse centroids still route vectors to lists; only the
  encoding reference changes. (With L2 the precompute-table optimization is
  skipped — `IndexIVFPQ.cpp:410-413`.)

> The coarse quantizer's centroids **are** the cluster centroids — the same
> `nlist` vectors. You cannot have "1 quantizer centroid but k cluster centroids";
> `nlist = 1` would mean a single cluster (degenerate IVF).

**Caveat for IVF-RaBitQ:** `encode_vectors` always reconstructs the per-list
centroid and encodes against it — *"both by_residual and !by_residual lead to the
same code"* (`IndexIVFRaBitQ.cpp:76`). Origin-relative encoding is not available
for RaBitQ via the flag.

## Cluster balance over time

Because centroids are frozen at `train()`, lists **can drift out of balance** as
you mutate the index — one cluster grows huge while others stay tiny or go empty.
Expected, not a bug; FAISS does **not** auto-correct it.

**Causes.** `add` routes each vector to its nearest *existing* centroid;
`remove_ids` deletes from whatever list held the vector. Neither moves centroids.
Imbalance appears when live data ≠ training distribution: non-representative
training sample, concept drift, naturally non-uniform density, or deletes that
empty lists. `nlist` is constant — an emptied list keeps its centroid at size 0.

**What FAISS does: measure, not fix.** No rebalancing, splitting, or merging
exists in the IVF/InvertedLists code. The only tooling is measurement:

- `InvertedLists::imbalance_factor()` (`faiss/invlists/InvertedLists.cpp:184-192`):
  `nlist·Σ(sizeᵢ²)/(Σsizeᵢ)²`. `1.0` = perfectly balanced; `>1` ≈ how much bigger
  the average *scanned* list is than ideal.
- `InvertedLists::print_stats()` (`InvertedLists.cpp:194-211`): power-of-two
  histogram of list sizes.

```python
print(index.invlists.imbalance_factor())   # ~1.0 good; 3.0 ≈ 3x worse than ideal
index.invlists.print_stats()
```

**Why it matters.** Search scans every vector in the `nprobe` probed lists; a
query hitting a giant list scans far more than budgeted (slower, heap dominated by
that cluster); tiny/empty lists hurt recall. Expected work scales with
`imbalance_factor`.

**Mitigations (manual — FAISS is build-then-serve, no online re-clustering):**
rebuild/retrain periodically on representative data; train on representative data;
size `nlist ≈ √N`; raise `nprobe` to recover recall; monitor `imbalance_factor`.

## What if you skip `train()`?

`add` throws — a fresh IVF index has `is_trained == False`
(`IndexIVF.cpp:177-178`: `quantizer->is_trained && quantizer->ntotal == nlist`),
and `add_core` asserts `is_trained` (`IndexIVF.cpp:239`). Exception: hand the IVF a
quantizer already holding `nlist` centroids and `is_trained` becomes True at
construction (works for `IndexIVFFlat`; PQ still needs `train()` for its codebook,
but `train_q1` will skip k-means).

## Python: where centroids come from and how to read them

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

# ---------- Encode against origin instead of per-cluster centroid ----------
ivf3 = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, 8, 8)
ivf3.by_residual = False                        # set BEFORE train()
ivf3.train(xt); ivf3.add(xb)                    # k clusters for routing; encode vs origin
```

## Source references

| What | Location |
|---|---|
| `Level1Quantizer::train_q1` (k-means) | `faiss/IndexIVF.cpp:56-122` |
| IVF `cp.niter = 10` | `faiss/IndexIVF.cpp:40-46` |
| pre-trained quantizer skip | `faiss/IndexIVF.cpp:63-67` |
| `IndexIVF::train` (residual flow) | `faiss/IndexIVF.cpp:1285-1318` |
| `IndexIVFPQ::train_encoder` (PQ codebooks) | `faiss/IndexIVFPQ.cpp:73-92` |
| `IndexIVFRaBitQ::train_encoder` (no-op) | `faiss/IndexIVFRaBitQ.cpp:49-54` |
| clustering defaults | `faiss/Clustering.h:44-46` |
| `by_residual` field / use | `faiss/IndexIVF.h:220`; `faiss/IndexIVFPQ.cpp:181-191`, `410-413` |
| RaBitQ ignores `by_residual` | `faiss/IndexIVFRaBitQ.cpp:76` |
| `imbalance_factor` / `print_stats` | `faiss/invlists/InvertedLists.cpp:184` / `194` |
| construction `is_trained` | `faiss/IndexIVF.cpp:177-178` |
| `is_trained` guard in add | `faiss/IndexIVF.cpp:239` |

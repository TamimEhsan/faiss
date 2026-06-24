# HNSW family — add yes, delete no

[← index](README.md)

## Structure

`IndexHNSW` is a wrapper (`faiss/IndexHNSW.h:31-40`):

- `Index* storage` — holds the actual vectors and provides the `DistanceComputer`
  (`IndexHNSW.cpp:696-698`; `reconstruct` delegates to storage,
  `IndexHNSW.cpp:403-405`).
- `HNSW hnsw` — only the **neighbor-link graph**: `levels`, `offsets`, `neighbors`
  (flat `storage_idx_t` arrays), and an `entry_point`
  (`faiss/impl/HNSW.h:114-131`). A node's HNSW id **is** its positional index in
  `storage`.

## Add (`faiss/IndexHNSW.cpp:378-394`, `hnsw_add_vertices` `63-213`)

1. Require `is_trained` and a non-null `storage` (`IndexHNSW.cpp:382-383`).
2. `storage->add(n, x)` appends raw/quantized vectors; `ntotal = storage->ntotal`.
3. `hnsw_add_vertices`: assign random levels (`prepare_level_tab`,
   `HNSW.cpp:211-235`), bucket new points by level (highest first for entry-point
   logic), then in parallel call `hnsw.add_with_locks(dist, level, pt_id, locks,
   vt)` (`impl/HNSW.h:209-215`) to greedily find neighbors per level and wire
   bidirectional edges (per-node locks; neighbor pruning to the M budget).

So "add" = append to storage + incrementally splice the new node into the graph.
No global rebuild.

## Delete — unsupported

`IndexHNSW` does **not** override `remove_ids`, so it falls through to
`Index::remove_ids`, which throws
`"remove_ids not implemented for this type of index"` (`faiss/Index.cpp:56-59`).
This holds for **every** HNSW variant (`IndexHNSWFlat`, `IndexHNSWPQ`,
`IndexHNSWSQ`, `IndexHNSW2Level`, `IndexHNSWCagra`) regardless of storage type.

**Why deletion is hard:** edges are stored as `storage_idx_t` positional ids.
Removing a node would require (a) finding and rewriting every *incoming* edge (the
graph keeps no reverse index), (b) possibly re-selecting the `entry_point` and
repairing upper levels, and (c) compacting/remapping storage positions, which
shifts all higher ids and invalidates the entire neighbor array. HNSW is designed
append-only; practical "deletion" is an external tombstone / `IDSelector` filter
at search time, or a rebuild.

## Quantized variants

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

## Source references

| What | Location |
|---|---|
| `IndexHNSW` fields (`storage`, `hnsw`) | `faiss/IndexHNSW.h:31-40` |
| HNSW graph arrays | `faiss/impl/HNSW.h:114-131` |
| `IndexHNSW::add` | `faiss/IndexHNSW.cpp:378-394` |
| `hnsw_add_vertices` | `faiss/IndexHNSW.cpp:63-213` |
| `HNSW::prepare_level_tab` | `faiss/impl/HNSW.cpp:211-235` |
| `add_with_locks` decl | `faiss/impl/HNSW.h:209-215` |
| `reconstruct` / distance delegate to storage | `faiss/IndexHNSW.cpp:403-405` / `696-698` |
| `Index::remove_ids` (throws) | `faiss/Index.cpp:56-59` |
| `IndexHNSWFlat` ctor | `faiss/IndexHNSW.cpp:708-715` |
| `IndexHNSWPQ` ctor / `train` | `faiss/IndexHNSW.cpp:779-793` |
| `IndexHNSWSQ` ctor | `faiss/IndexHNSW.cpp:799-807` |

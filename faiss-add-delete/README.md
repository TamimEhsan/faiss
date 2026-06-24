# FAISS add & delete: IVF-PQ, IVF-RaBitQ, HNSW

How FAISS inserts (`add` / `add_with_ids`) and removes (`remove_ids` /
`update_vectors`) vectors, how centroids are trained, and how memory is managed —
for the IVF family (IVF-PQ, IVF-RaBitQ) and the HNSW family (incl. quantized
`HNSWPQ` / `HNSWSQ`). All `file:line` references are against this checkout.

## Contents

1. [IVF add & delete machinery](01-ivf-add-delete.md) — coarse assign → encode →
   inverted lists; `remove_ids`; `DirectMap`; `update_vectors`; invlists backends.
2. [Centroids & training](02-centroids-and-training.md) — what centroids are, when
   they're computed (k-means), `by_residual`, cluster balance over time, Python.
3. [IVF-PQ specifics](03-ivf-pq.md) — residual PQ encoding, precomputed tables.
4. [IVF-RaBitQ specifics](04-ivf-rabitq.md) — centroid-relative sign-bit encoding.
5. [HNSW family](05-hnsw.md) — add yes, delete no; quantized variants.
6. [Memory management](06-memory-management.md) — growth, shrink, reclamation;
   disk-based loading (mmap views, `OnDiskInvertedLists`).
7. [Exposed interfaces](07-exposed-interfaces.md) — `IndexIVFPQ` / `IndexIVFRaBitQ` API.

## TL;DR

| Index | Add | Delete (`remove_ids`) |
|---|---|---|
| **IVF-PQ** | ✅ coarse-assign → residual PQ-encode → append to inverted list | ✅ supported (swap-with-last in lists), with `DirectMap` caveats |
| **IVF-RaBitQ** | ✅ same IVF flow; RaBitQ encodes residual vs. centroid | ✅ inherited from `IndexIVF`, identical mechanics to IVF-PQ |
| **HNSW (Flat/PQ/SQ/…)** | ✅ append to `storage` index + incrementally wire graph | ❌ **not supported** — throws `"remove_ids not implemented…"` |

The deep difference: **IVF deletion is cheap because the structure is a flat set
of per-cluster lists**; **HNSW deletion is unimplemented because graph edges are
positional references and removal would require global graph repair.**

**Centroids** (IVF coarse list centers, and PQ codebooks) are learned **only at
`train()`** and never change on add/remove — see
[Centroids & training](02-centroids-and-training.md).

## Practical guidance

- **Need deletes/updates?** Use the **IVF** family. For arbitrary ids and
  `remove_ids`, `set_direct_map_type(DirectMap::Hashtable)` and delete via
  `IDSelectorArray`; or keep the default `NoMap` and pay an O(ntotal) scan with a
  general `IDSelector`. Avoid `Array` mode if you need removal.
- **Expect position churn:** swap-with-last reorders surviving entries; rely on
  external ids, not list offsets.
- **HNSW:** treat as append-only. Emulate deletion with an `IDSelector` filter at
  query time, and rebuild periodically to reclaim space.
- **Quantized indexes:** always `train` before `add`.
- **Reclaiming RAM:** deletes never shrink process RSS — rebuild or round-trip
  through `write_index`/`read_index`.

## Source map (entry points)

| Operation | Symbol | Location |
|---|---|---|
| IVF add | `IndexIVF::add` / `add_with_ids` / `add_core` | `faiss/IndexIVF.cpp:187` / `191` / `213` |
| IVF train | `IndexIVF::train`; `Level1Quantizer::train_q1` | `faiss/IndexIVF.cpp:1285` / `56` |
| IVF delete | `IndexIVF::remove_ids` → `DirectMap::remove_ids` | `faiss/IndexIVF.cpp:1252`; `faiss/invlists/DirectMap.cpp:157` |
| IVF-PQ encode | `IndexIVFPQ::encode_vectors`; `train_encoder` | `faiss/IndexIVFPQ.cpp:175` / `73` |
| IVF-RaBitQ encode | `IndexIVFRaBitQ::encode_vectors`; `add_core` | `faiss/IndexIVFRaBitQ.cpp:56` / `110` |
| HNSW add | `IndexHNSW::add`; `hnsw_add_vertices` | `faiss/IndexHNSW.cpp:378` / `63` |
| HNSW delete (throws) | `Index::remove_ids` | `faiss/Index.cpp:56` |

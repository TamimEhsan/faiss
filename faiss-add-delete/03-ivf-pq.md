# IVF-PQ specifics (`IndexIVFPQ`)

[← index](README.md) · builds on [IVF add & delete machinery](01-ivf-add-delete.md)

`IndexIVFPQ : IndexIVF`, with a `ProductQuantizer pq` that encodes each (residual)
vector into a PQ code.

## Encoding

`IndexIVFPQ::encode_vectors` (`faiss/IndexIVFPQ.cpp:175-197`): when `by_residual`
(default `true`, `IndexIVF.h:220`), compute the residual `x − centroid[list_no]`
via `compute_residuals` (`IndexIVFPQ.cpp:155-173`), then `pq.compute_codes` on the
residual. Decode = decoded residual + centroid. `by_residual = false` encodes `x`
directly (origin-relative).

## Outliers (`list_no == -1`)

Residual zeroed, tracked in `DirectMap` as `-1`, **not** stored in any list
(`IndexIVFPQ.cpp:300-301`).

## Precomputed tables

A *search-time* acceleration derived from the centroids/codebook
(`IndexIVFPQ::precompute_table`, built in `train_encoder` when `by_residual` &
L2). Add/delete just mutate list contents and don't special-case them.

## Add

- `IndexIVFPQ::add_core` (`IndexIVFPQ.cpp:146`) — standard IVF flow with PQ
  encoding.
- `IndexIVFPQ::add_core_o` (`IndexIVFPQ.cpp:233`) — like `add_core`, but can also
  output 2nd-level residuals and accept `precomputed_idx = nullptr`.

## Delete

Not overridden — uses the generic `IndexIVF` path
(see [IVF add & delete machinery → Delete](01-ivf-add-delete.md#delete-path--remove_ids)).

## Source references

| What | Location |
|---|---|
| `IndexIVFPQ::encode_vectors` | `faiss/IndexIVFPQ.cpp:175-197` |
| `compute_residuals` | `faiss/IndexIVFPQ.cpp:155-173` |
| `IndexIVFPQ::train_encoder` (PQ train + precompute) | `faiss/IndexIVFPQ.cpp:73-92` |
| `IndexIVFPQ::add_core` | `faiss/IndexIVFPQ.cpp:146` |
| `IndexIVFPQ::add_core_o` | `faiss/IndexIVFPQ.cpp:233` |
| outlier handling (`key < 0`) | `faiss/IndexIVFPQ.cpp:300-301` |
| `by_residual` field | `faiss/IndexIVF.h:220` |
| class / `pq` field | `faiss/IndexIVFPQ.h:33-50` |

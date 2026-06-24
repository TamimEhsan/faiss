# IVF-RaBitQ specifics (`IndexIVFRaBitQ`)

[← index](README.md) · builds on [IVF add & delete machinery](01-ivf-add-delete.md)

`IndexIVFRaBitQ : IndexIVF`, holding a `RaBitQuantizer rabitq`. The constructor
sets `by_residual = true` and `code_size = rabitq.code_size`
(`faiss/IndexIVFRaBitQ.cpp:27-43`).

## Encoding

`IndexIVFRaBitQ::encode_vectors` (`faiss/IndexIVFRaBitQ.cpp:56-87`): reconstructs
the list centroid from the coarse quantizer and calls
`rabitq.compute_codes_core(xi, code, 1, centroid)` — RaBitQ subtracts the centroid
internally, so residual encoding is implicit. The source notes *"both by_residual
and !by_residual lead to the same code"* (`IndexIVFRaBitQ.cpp:76`), i.e. encoding
is **always** centroid-relative; the `by_residual` flag has no effect.

RaBitQ code layout: sign bits `(d+7)/8` + factor block(s); larger for multi-bit
(`faiss/impl/RaBitQuantizer.h:51-69`).

## Add

`IndexIVFRaBitQ::add_core` (`faiss/IndexIVFRaBitQ.cpp:110-153`) overrides
`add_core` only to encode one vector at a time against its centroid; otherwise the
same `DirectMapAdd` + `invlists->add_entry` flow as base IVF.

## Train

`IndexIVFRaBitQ::train_encoder` (`faiss/IndexIVFRaBitQ.cpp:49-54`) calls
`rabitq.train(...)`, which is a **no-op**. The only learned centroids are the
coarse k-means centroids — see
[Centroids & training](02-centroids-and-training.md).

## Delete

**Not overridden** — fully inherits `IndexIVF::remove_ids`
(see [IVF add & delete machinery → Delete](01-ivf-add-delete.md#delete-path--remove_ids)).
Mechanically identical to IVF-PQ. (`IndexIVFRaBitQFastScan` stores codes in a
`BlockInvertedLists`, so the fast-scan remove caveats apply.)

## Source references

| What | Location |
|---|---|
| constructor (`by_residual`, `code_size`) | `faiss/IndexIVFRaBitQ.cpp:27-43` |
| `IndexIVFRaBitQ::encode_vectors` | `faiss/IndexIVFRaBitQ.cpp:56-87` |
| centroid-relative always (comment) | `faiss/IndexIVFRaBitQ.cpp:76` |
| `IndexIVFRaBitQ::train_encoder` (no-op) | `faiss/IndexIVFRaBitQ.cpp:49-54` |
| `IndexIVFRaBitQ::add_core` | `faiss/IndexIVFRaBitQ.cpp:110-153` |
| RaBitQ code layout | `faiss/impl/RaBitQuantizer.h:51-69` |
| class / fields (`rabitq`, `qb`) | `faiss/IndexIVFRaBitQ.h:25-33` |

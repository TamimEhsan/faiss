import faiss
import numpy as np

print("faiss version:", faiss.__version__)

d = 64
nb = 1000
rng = np.random.default_rng(0)
xb = rng.random((nb, d)).astype("float32")
xq = rng.random((5, d)).astype("float32")

# Case 1: plain IndexPQ from factory
index = faiss.index_factory(d, "PQ32x8")
index.train(xb)
index.add(xb)

sel = faiss.IDSelectorRange(100, 500)
params = faiss.SearchParametersPQ()
params.sel = sel

print("\n--- IndexPQ + SearchParametersPQ(sel=IDSelectorRange) ---")
try:
    D, I = index.search(xq, 5, params=params)
    print("OK, labels[0] =", I[0])
    print("all in [100,500)?", bool(((I >= 100) & (I < 500)).all()))
except Exception as e:
    print("RAISED:", type(e).__name__)
    print(str(e).strip().splitlines()[-1])

# Case 2: the factory string from the issue
print("\n--- IDMap2,PQ32x8 + SearchParametersPQ(sel) ---")
index2 = faiss.index_factory(d, "IDMap2,PQ32x8")
index2.train(xb)
index2.add_with_ids(xb, np.arange(nb).astype("int64"))
try:
    D, I = index2.search(xq, 5, params=params)
    print("OK, labels[0] =", I[0])
except Exception as e:
    print("RAISED:", type(e).__name__)
    print(str(e).strip().splitlines()[-1])

# Baseline: SearchParameters with sel works on IndexFlat
print("\n--- baseline: IndexFlatL2 + SearchParameters(sel) ---")
flat = faiss.IndexFlatL2(d)
flat.add(xb)
p2 = faiss.SearchParameters()
p2.sel = sel
try:
    D, I = flat.search(xq, 5, params=p2)
    print("OK, all in [100,500)?", bool(((I >= 100) & (I < 500)).all()))
except Exception as e:
    print("RAISED:", type(e).__name__, str(e).strip().splitlines()[-1])

from time import perf_counter
import numpy as np
import cupy as cp

# generate the same matrix as in the numpy case
N = 10000
np.random.seed(42)
X = np.random.randn(N, N).astype(np.float64)
X = cp.asarray(X)

trials = 3
times = []
for _ in range(trials):
    t0 = perf_counter()
    Y = cp.matmul(X, X)
    cp.cuda.Device(0).synchronize()
    times.append(perf_counter() - t0)
print("Execution time: ", min(times))
print("sum(Y) / N = ", Y.sum() / N)

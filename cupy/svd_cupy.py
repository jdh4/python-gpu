from time import perf_counter
import cupy as cp

N = 3000
X = cp.random.randn(N, N, dtype=cp.float64)

trials = 3
times = []
for _ in range(trials):
    t0 = perf_counter()
    u, s, v = cp.linalg.svd(X)
    cp.cuda.Device(0).synchronize()
    times.append(perf_counter() - t0)
print("Execution time: ", min(times))
print("sum(s) / N = ", s.sum() / N)

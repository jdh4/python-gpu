from time import perf_counter
import numpy as np

N = 3000
X = np.random.randn(N, N).astype(np.float64)

trials = 3
times = []
for _ in range(trials):
    t0 = perf_counter()
    u, s, v = np.linalg.svd(X)
    times.append(perf_counter() - t0)
print("Execution time: ", min(times))
print("sum(s) / N = ", s.sum() / N)

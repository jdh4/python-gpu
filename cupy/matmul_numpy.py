from time import perf_counter
import numpy as np

N = 10000
np.random.seed(42)
X = np.random.randn(N, N).astype(np.float64)

trials = 3
times = []
for _ in range(trials):
    t0 = perf_counter()
    Y = np.matmul(X, X)
    times.append(perf_counter() - t0)
print("Execution time: ", min(times))
print("sum(Y) / N = ", Y.sum() / N)

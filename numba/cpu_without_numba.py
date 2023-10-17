from time import perf_counter
import numpy as np

def myfunc(a):
    trace = 0.0
    # assuming square input matrix
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

x = np.arange(10000).reshape(100, 100)

trials = 3
times = []
for _ in range(trials):
    t0 = perf_counter()
    y = myfunc(x)
    times.append(perf_counter() - t0)
print(min(times))

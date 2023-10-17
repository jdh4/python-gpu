from time import perf_counter
import numpy as np
import numba

@numba.jit(nopython=True)
def myfunc(a): # function is compiled to machine code when called the first time
    trace = 0.0
    # assuming square input matrix
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace

x = np.arange(10000).reshape(100, 100)

trials = 3
times = []
for _ in range(trials):
    t0 = perf_counter()
    y = myfunc(x)
    times.append(perf_counter() - t0)
print(min(times))

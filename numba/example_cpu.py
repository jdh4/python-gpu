import numpy as np
import numba

@numba.jit(nopython=True)
def go_fast(a): # function is compiled to machine code when called the first time
    trace = 0.0
    # assuming square input matrix
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace

x = np.arange(100).reshape(10, 10)
go_fast(x)
y = go_fast(2 * x)

print(f"y.sum() = {y.sum()}")

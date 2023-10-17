# CuPy and Python GPU Libraries

### CuPy is a drop-in replacement for NumPy

NumPy:

```python
import np as np
X = np.random.randn(3000, 3000)
u, s, v = np.linalg.svd(X)
```

CuPy:

```python
import cupy as cp
X = cp.random.randn(3000, 3000)
u, s, v = cp.linalg.svd(X)
```


# CuPy and Python GPU Libraries

### CuPy is a drop-in replacement for NumPy

NumPy (for CPUs)

```python
import np as np
X = np.random.randn(3000, 3000)
u, s, v = np.linalg.svd(X)
```

CuPy (for GPUs)

```python
import cupy as cp
X = cp.random.randn(3000, 3000)
u, s, v = cp.linalg.svd(X)
```

Let's compare the performance of the two.

```
$ cd python-gpu/cupy
$ sbatch cupy.slurm

$ sbatch numpy.slurm
```

In the above case we are comparing the CuPy code running on 1 CPU-cores and 1 A100 GPU versus 16 CPU-cores and no GPU. The choice of 16 was found to optimal for the CPU case. What of the two libraries performs faster?


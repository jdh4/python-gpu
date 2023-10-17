# CuPy and Python GPU Libraries

## Obtain the files

Run the commands below to get started:

```bash
$ ssh <YourNetID>@adroit.princeton.edu
$ cd /scratch/network/$USER
$ git clone https://github.com/jdh4/python-gpu.git
$ cd python-gpu
```

## CuPy is a drop-in replacement for NumPy and SciPy

NumPy (for CPUs)

```python
import np as np
X = np.random.randn(10000, 10000)
Y = np.matmul(X, X)
```

[CuPy](https://docs.cupy.dev/en/stable/index.html) (for GPUs)

```python
import cupy as cp
X = cp.random.randn(10000, 10000)
Y = cp.matmul(X, X)
```

Let's compare the performance of the two.

```
$ cd python-gpu/cupy
$ cat matmul_numpy.py
$ cat matmul_cupy.py
```

The difference between the two scripts are minimal as expected. Run the jobs and compare the timings:

```
$ sbatch cupy.slurm
$ sbatch numpy.slurm
```

In the above case we are comparing the CuPy code running on 1 CPU-cores and 1 A100 GPU versus 8 CPU-cores and no GPU. Which of the two libraries performs faster for this problem?

### Exercise

The code below calculates the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) of a matrix using NumPy. Convert the code to CuPy and run it.

```
import numpy as np
X = np.random.randn(5000, 5000)
u, s, v = np.linalg.svd(X)
```

Hint: You only need to change 6 characters.

### CuPy Documentation

Take a look at the [CuPy reference manual](https://docs.cupy.dev/en/stable/reference/index.html). Could you use CuPy to speed-up your research?

## JAX


## PyTorch and TensorFlow


## Rapids


## Numba


## MATLAB, Julia and R

See the [Intro to GPU Computing](https://github.com/PrincetonUniversity/gpu_programming_intro) repo for GPU examples in other languages.

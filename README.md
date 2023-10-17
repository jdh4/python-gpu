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

<img src="https://raw.githubusercontent.com/cupy/cupy/main/docs/image/cupy_logo_1000px.png" alt="logo" width="300"></img>

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

Let's compare the performance of the two. Inspect the two Python scripts:

```
$ cd python-gpu/cupy
$ cat matmul_numpy.py
$ cat matmul_cupy.py
```

The difference between the two scripts is minimal as expected. Run the jobs and compare the timings:

```
$ sbatch cupy.slurm
$ sbatch numpy.slurm
```

In the above case we are comparing the CuPy code running on 1 CPU-core and 1 A100 GPU versus NumPy code running on 8 CPU-cores and no GPU. Which of the two libraries performs faster for this problem?

### Exercise

The code below calculates the [singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) of a matrix using NumPy. Convert the code to CuPy and run it.

```python
import numpy as np
X = np.random.randn(5000, 5000)
u, s, v = np.linalg.svd(X)
```

Hint: You only need to change 6 characters.

### CuPy Documentation

Take a look at the [CuPy reference manual](https://docs.cupy.dev/en/stable/reference/index.html). Could you use CuPy to speed-up your research?

## JAX

<img src="https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png" alt="logo"></img>

[JAX](https://github.com/google/jax) is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought
together for high-performance machine learning research. JAX can be used for:

- automatic differentiation of Python and NumPy functions (more general then TensorFlow)
- a good choice for non-conventional neural network architectures and loss functions
- accelerating code using a JIT
- carrying out computations using multiple GPUs/TPUs

```bash
$ cd python-gpu/jax
$ cat example.py
```
```python
import jax.numpy as jnp
from jax import grad, jit, vmap

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)  # inputs to the next layer
  return outputs                # no activation on last layer

def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2)

grad_loss = jit(grad(loss))  # compiled gradient evaluation function
perex_grads = jit(vmap(grad_loss, in_axes=(None, 0, 0)))  # fast per-example grads
```

Run the code with:

```
$ sbatch submit.sh
```

Take a look at all of the [JAX examples](https://github.com/google/jax). You can run any of the examples by modifying example.py with the example you want to run.

See our [JAX knowledgebase](https://researchcomputing.princeton.edu/support/knowledge-base/jax) page for installation directions.

## PyTorch and TensorFlow


## Rapids


## Numba


## MATLAB, Julia and R

See the [Intro to GPU Computing](https://github.com/PrincetonUniversity/gpu_programming_intro) repo for GPU examples in other languages.

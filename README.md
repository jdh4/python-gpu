# CuPy and Python GPU Libraries

## Obtain the files

Run the commands below to get started:

```bash
$ ssh <YourNetID>@adroit.princeton.edu
$ cd /scratch/network/$USER
$ git clone https://github.com/jdh4/python-gpu.git
$ cd python-gpu
```

# CuPy is a drop-in replacement for NumPy and SciPy

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

Take a look at the [CuPy reference manual](https://docs.cupy.dev/en/stable/reference/index.html). Can you use CuPy in your research?

# JAX

<img src="https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png" alt="logo"></img>

[JAX](https://github.com/google/jax) is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought
together for high-performance machine learning research. JAX can be used for:

- automatic differentiation of Python and NumPy functions (more general then TensorFlow)
- a good choice for non-conventional neural network architectures and loss functions
- accelerating code using a JIT
- carrying out computations using multiple GPUs/TPUs

Take a look at an example from the JAX GitHub page:

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

See our [JAX knowledge base](https://researchcomputing.princeton.edu/support/knowledge-base/jax) page for installation directions.

# Numba

[Numba](https://numba.pydata.org/) is an open source JIT compiler that translates a subset of Python and NumPy code into fast machine code.

On compiling Python to machine code:

> When Numba is translating Python to machine code, it uses the LLVM library to do most of the optimization and final code generation. This automatically enables a wide range of optimizations that you don't even have to think about.

Take a look at a sample CPU code:

```
$ cd python-gpu/numba
$ cat example_cpu.py
```

```python
import numpy as np
import numba

@numba.jit(nopython=True)
def go_fast(a): # function is compiled to machine code when called the first time
    trace = 0.0
    # assuming square input matrix
    for i in range(a.shape[0]):    # numba likes loops
        trace += np.tanh(a[i, i])  # numba likes numpy functions
    return a + trace

x = np.arange(100).reshape(10, 10)
go_fast(x)
go_fast(2 * x)
```

Run the example above:

```
$ sbatch numba_cpu.slurm
```

Let's look at a GPU example. According to the [Numba for GPUs](https://numba.readthedocs.io/en/stable/cuda/overview.html) webpage:

> Numba supports CUDA GPU programming by directly compiling a restricted subset of Python code into CUDA kernels and device functions following the CUDA execution model. Kernels written in Numba appear to have direct access to NumPy arrays. NumPy arrays are transferred between the CPU and the GPU automatically.

View the Python script:

```
$ cat example_gpu.py
```
```python
import numpy as np
from numba import cuda

@cuda.jit
def my_kernel(io_array):
    # thread id in a 1D block
    tx = cuda.threadIdx.x
    # block id in a 1D grid
    ty = cuda.blockIdx.x
    # block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # compute flattened index inside the array
    pos = tx + ty * bw
    if pos < io_array.size:  # check array boundaries
        io_array[pos] *= 2

if __name__ == "__main__":

    # create the data array - usually initialized some other way
    data = np.ones(100000)

    # set the number of threads in a block
    threadsperblock = 256

    # calculate the number of thread blocks in the grid
    blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock

    # call the kernel
    my_kernel[blockspergrid, threadsperblock](data)

    # print the result
    print(data[:3])
    print(data[-3:])
```

Run the job:

```
$ sbatch numba_gpu.slurm
```

Can you use Numba in your work to accelerate Python code on a CPU or GPU?

For more, see [High-Performance Python for GPUs](https://github.com/henryiii/pygpu-minicourse) by Henry Schreiner.

# Rapids

# PyTorch and TensorFlow

See the [Intro to GPU Computing](https://github.com/PrincetonUniversity/gpu_programming_intro) repo for PyTorch and TensorFlow examples. For installation directions and more, see our [PyTorch](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch) and [TensorFlow](https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow) knowledge base pages.

# MATLAB, Julia and R

See the [Intro to GPU Computing](https://github.com/PrincetonUniversity/gpu_programming_intro) repo for GPU examples in other languages.

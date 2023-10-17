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

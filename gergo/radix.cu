
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t maxWithCuda(int* data, unsigned int dataSize);
cudaError_t compactWithCuda(int* data, int* keep_data, int* not_keep_data, int* zero_offset_data, int* max_data, unsigned int nloops, unsigned int dataSize);

__global__
void mapNotKernel(int* x, unsigned int dataSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > dataSize) return;
    x[id] = x[id] == 0 ? 1 : 0;
}

__global__
void mapBitKernel(int* x, int n, unsigned int dataSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > dataSize) return;
    x[id] = (x[id] >> n) & 1U;
}

__global__
void predNonnegReduceKernel(int* data, int* keep, unsigned int dataSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (keep[id] == 0) data[id] = -1;
    for (int s = dataSize / 2; s > 0; s >>= 1) {
        if (id < s) {
            data[id] = (data[id] > data[id + s]) ? data[id] : data[id + s];
        }
        __syncthreads();
    }
}

__global__
void reduceKernel(int* data, unsigned int dataSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for (int s = dataSize / 2; s > 0; s >>= 1) {
        if (id < s) {
            data[id] = (data[id] > data[id + s]) ? data[id] : data[id + s];
        }
        __syncthreads();
    }
}

__global__
void exscanKernel(int* data, unsigned int dataSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > 0) data[id] = data[id - 1];
    __syncthreads();
    if (id == 0) data[id] = 0;
    __syncthreads();

    for (int s = 1; s < dataSize; s *= 2) {
        int tmp = data[id];
        if (id + s < dataSize) data[id + s] += tmp;
        __syncthreads();
    }
    if (id == 0) data[id] = 0;
}

__global__
void compactMapKernel(int* new_data, int* data, int* zero_offset_data, unsigned int zeroOffsetMax, int* one_offset_data, int* keep_data, unsigned int dataSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > dataSize) return; 
    if (keep_data[id] == 1) new_data[zeroOffsetMax + one_offset_data[id]] = data[id];
    else new_data[zero_offset_data[id]] = data[id];
}

int main()
{
    int data[8] = { 2, 36, 8, 11, 5, 20, 55, 1 };
    int max_data[8] = { 2, 36, 8, 11, 5, 20, 55, 1 };
    int keep_data[8] = { 0,1,1,0,0,1,1,1 };
    int not_keep_data[8] = { 0,1,1,0,0,1,1,1 };
    int zoffset_data[8] = { 0,1,1,0,0,1,1,1 };

    cudaError_t cudaStatus = maxWithCuda(max_data, 8);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "maxWithCuda failed!");
        return 1;
    }

    int max = max_data[0];
    unsigned int bitnum = 0;
    while (max > 0) {
        max = max >> 1;
        ++bitnum;
    }

    // Do the operation on vectors in parallel.
    cudaStatus = compactWithCuda(data, keep_data, not_keep_data, zoffset_data, max_data, bitnum, 8);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "compactWithCuda failed!");
        return 1;
    }

    printf("{ 2, 36, 8, 11, 5, 20, 55, 1 } radix sorted is {%d,%d,%d,%d,%d,%d,%d,%d}\n", data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t maxWithCuda(int* data, unsigned int dataSize)
{
    int* dev_data = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (three input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_data, dataSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_data, data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    reduceKernel << <1, dataSize >> > (dev_data, dataSize);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "reduceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduceKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(data, dev_data, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_data);

    return cudaStatus;
}

cudaError_t compactWithCuda(int* data, int* keep_data, int* not_keep_data, int* zero_offset_data, int* max_data, unsigned int nloops, unsigned int dataSize)
{
    int* keep_dev_data = 0;
    int* notkeep_dev_data = 0;
    int* zoffset_dev_data = 0;
    int* ooffset_dev_data = 0;
    int* max_dev_data = 0;
    int* dev_data = 0;
    int* dev_new_data = 0;
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    for(int i=0;i<nloops;++i){
        
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&keep_dev_data, dataSize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(keep_dev_data, data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        mapBitKernel << <1, dataSize >> > (keep_dev_data, i, dataSize);
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "mapBitKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mapBitKernel!\n", cudaStatus);
            goto Error;
        }
        cudaStatus = cudaMemcpy(keep_data, keep_dev_data, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }


        cudaStatus = cudaMalloc((void**)&ooffset_dev_data, dataSize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(ooffset_dev_data, keep_data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        // Launch a kernel on the GPU with one thread for each element.
        exscanKernel << <1, dataSize >> > (ooffset_dev_data, dataSize);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "exscanKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching exscanKernel!\n", cudaStatus);
            goto Error;
        }


        // Allocate GPU buffers for three vectors (three input, one output)
        cudaStatus = cudaMalloc((void**)&notkeep_dev_data, dataSize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(notkeep_dev_data, keep_data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        mapNotKernel << <1, dataSize >> > (notkeep_dev_data, dataSize);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "reduceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching mapNotKernel!\n", cudaStatus);
            goto Error;
        }
        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(not_keep_data, notkeep_dev_data, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }


        cudaStatus = cudaMalloc((void**)&zoffset_dev_data, dataSize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(zoffset_dev_data, not_keep_data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        // Launch a kernel on the GPU with one thread for each element.
        exscanKernel << <1, dataSize >> > (zoffset_dev_data, dataSize);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "exscanKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching exscanKernel!\n", cudaStatus);
            goto Error;
        }
        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(zero_offset_data, zoffset_dev_data, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }


        // Allocate GPU buffers for three vectors (three input, one output)
        cudaStatus = cudaMalloc((void**)&max_dev_data, dataSize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(max_dev_data, zero_offset_data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        predNonnegReduceKernel << <1, dataSize >> > (max_dev_data, notkeep_dev_data, dataSize);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "predNonnegReduceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching predNonnegReduceKernel!\n", cudaStatus);
            goto Error;
        }
        cudaStatus = cudaMemcpy(max_data, max_dev_data, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }


        cudaStatus = cudaMalloc((void**)&dev_data, dataSize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        cudaStatus = cudaMemcpy(dev_data, data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_new_data, dataSize * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        compactMapKernel << <1, dataSize >> > (dev_new_data, dev_data, zoffset_dev_data, max_data[0]+1, ooffset_dev_data, keep_dev_data, dataSize);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "compactMapKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching compactMapKernel!\n", cudaStatus);
            goto Error;
        }

        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(data, dev_new_data, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }
    }

Error:
    cudaFree(ooffset_dev_data);
    cudaFree(zoffset_dev_data);
    cudaFree(max_dev_data);
    cudaFree(dev_data);
    cudaFree(keep_dev_data);
    cudaFree(notkeep_dev_data);
    cudaFree(dev_new_data);

    return cudaStatus;
}


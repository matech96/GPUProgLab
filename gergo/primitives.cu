 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t mapWithCuda(int *y, int *a, int* x, int *b, unsigned int dataSize);
cudaError_t maxWithCuda(int *data, unsigned int dataSize);
cudaError_t exscanWithCuda(int *data, unsigned int dataSize);
cudaError_t compactWithCuda(int *data, int* keep_data, int* offset_data, int* max_data, unsigned int dataSize);

// ID := threadIdx.x + blockIdx.x * blockDim.x
// IF ID > dataSize THEN return
// y[ID] := a[ID] * x[ID] + b[x] 

__global__
void mapKernel(int* y, int* a, int* x, int* b, unsigned int dataSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > dataSize) return;
    y[id] = a[id] * x[id] + b[id];
}

// FOR s = dataSize / 2 ; s > 0 ; s >>= 1 DO:
//  IF (ID < s)
//    data[ID] = max(data[ID], data[ID + s])
//  SYNCHRONIZE THREADS
//

__global__
void reduceKernel(int* data, unsigned int dataSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    for (int s = dataSize / 2; s > 0; s>>=1) {
        if (id < s) {
            data[id] = (data[id]>data[id+s]) ? data[id] : data[id+s];
        }
        __syncthreads();
    }
}

// IF ID > 0 THEN data[ID] = data[ID - 1]
//           ELSE data[ID] = 0
// SYNCHRONIZE THREADS
//
// FOR s = 1; s < dataSize; s *= 2 DO:
//   tmp := data[ID]
//   IF ( ID + s < dataSize THEN
//     data[ID + s] += tmp;
//   SYNCHRONIZE THREADS
//
// IF(ID = 0) THEN data[ID] = 0; 
__global__
void exscanKernel(int* data, unsigned int dataSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > 0) data[id] = data[id - 1];
    __syncthreads();
    if(id == 0) data[id] = 0;
    __syncthreads();

    for (int s = 1; s < dataSize; s *= 2) {
        int tmp = data[id];
        if (id + s < dataSize) data[id + s] += tmp;
        __syncthreads();
    }
    if (id == 0) data[id] = 0;
}

__global__
void compactMapKernel(int* new_data, int* data, int* offset_data, int* keep_data, unsigned int dataSize)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > dataSize) return;
    if (keep_data[id] == 1) new_data[offset_data[id]] = data[id];
}

int main()
{
    const int arraySize = 5;
    int a[arraySize] = { 4, 4, 4, 4, 4 };
    int x[arraySize] = { 2, 4, 2, 1, 8 };
    int b[arraySize] = { 1, 2, 3, 4, 5 };
    int max_data[] = { 3,1,6,4,-8,7,1,0 };
    int y[arraySize] = { 0 };
    int data[8] = { 3,1,6,4,-8,7,1,0 };
    int keep_data[8] = {0,1,1,0,0,1,1,1};
    int offset_data[8] = { 0,1,1,0,0,1,1,1 };
    int compact_max_data[8] = { 0,1,1,0,0,1,1,1 };

    // Do the operation on vectors in parallel.
    //cudaError_t cudaStatus = mapWithCuda(y, a, x, b, arraySize);
    //cudaError_t cudaStatus = maxWithCuda(max_data, 8);
    //cudaError_t cudaStatus = exscanWithCuda(b, arraySize);
    cudaError_t cudaStatus = compactWithCuda(data,keep_data,offset_data, compact_max_data, 8);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "compactWithCuda failed!");
        return 1;
    }

    //printf("{ 4, 4, 4, 4, 4 } * { 2, 4, 2, 1, 8 } + { 1, 2, 3, 4, 5 } = {%d,%d,%d,%d,%d}\n", y[0], y[1], y[2], y[3], y[4]);
    //printf("max of { 3,1,6,4,-8,7,1,0 } is %d\n",max_data[0]);
    //printf("Exclusive scan sum of { 1, 2, 3, 4, 5 } is {%d,%d,%d,%d,%d}\n", b[0], b[1], b[2], b[3], b[4]);
    printf("Compact of { 3,1,6,4,-8,7,1,0 } with {0,1,1,0,0,1,1,1} is {%d,%d,%d,%d,%d}\n", data[0], data[1], data[2], data[3], data[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to map vectors in parallel.
cudaError_t mapWithCuda(int* y, int* a, int* x, int* b, unsigned int dataSize)
{
    int *dev_a = 0;
    int *dev_x = 0;
    int *dev_b = 0;
    int *dev_y = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (three input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_y, dataSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, dataSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_x, dataSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, dataSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_x, x, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
    mapKernel<<<1, dataSize>>>(dev_y, dev_a, dev_x, dev_b, dataSize);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(y, dev_y, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_y);
    cudaFree(dev_a);
    cudaFree(dev_x);
    cudaFree(dev_b);
    
    return cudaStatus;
}

cudaError_t maxWithCuda(int *data, unsigned int dataSize)
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
    reduceKernel<<<1, dataSize >>> (dev_data, dataSize);

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

cudaError_t exscanWithCuda(int* data, unsigned int dataSize)
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
    exscanKernel << <1, dataSize >> > (dev_data, dataSize);

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

cudaError_t compactWithCuda(int* data, int* keep_data, int* offset_data, int* max_data, unsigned int dataSize)
{
    int* offset_dev_data = 0;
    int* max_dev_data = 0;
    int* dev_keep = 0;
    int* dev_data = 0;
    int* dev_new_data = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (three input, one output)    .
    cudaStatus = cudaMalloc((void**)&offset_dev_data, dataSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(offset_dev_data, keep_data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    exscanKernel << <1, dataSize >> > (offset_dev_data, dataSize);


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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduceKernel!\n", cudaStatus);
        goto Error;
    }
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(offset_data, offset_dev_data, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
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
    cudaStatus = cudaMemcpy(max_dev_data, offset_data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    reduceKernel << <1, dataSize >> > (max_dev_data, dataSize);

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
    
    cudaStatus = cudaMalloc((void**)&dev_keep, dataSize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_keep, keep_data, dataSize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&dev_new_data, (max_data[0] + 1) * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    compactMapKernel << <1, dataSize >> > (dev_new_data,dev_data,offset_dev_data,dev_keep, dataSize);

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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduceKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(data, dev_new_data, (max_data[0]+1) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(offset_dev_data);
    cudaFree(max_dev_data);
    cudaFree(dev_data);
    cudaFree(dev_keep);
    cudaFree(dev_new_data);

    return cudaStatus;
}


#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include <cmath>
#include "ticktock.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "freshman.h"




__global__ void kernel(int * arr, int n){
    for(int i = blockDim.x * blockIdx.x + threadIdx.x ; i < n; i += blockDim.x * gridDim.x){
        arr[i] = i;
    }
}

int main(int argc,char **argv)
{
    int n = 65535;
    int * arr;
    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));

    int threadsPerBlock = 128;
    int blocksPerGrid = (n + threadsPerBlock -1)/ threadsPerBlock;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(arr, n);

    checkCudaErrors(cudaDeviceSynchronize());

    for(int i = 0; i < n ; i++){
        printf("arr[%d] : %d\n", i, arr[i]);
    }
    cudaFree(arr);
    return 0;
}
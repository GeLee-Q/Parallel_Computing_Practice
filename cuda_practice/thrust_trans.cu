#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "ticktock.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// template <class Func>
// __global__ void parallel_for(int n, Func func) {
//     for (int i = blockDim.x * blockIdx.x + threadIdx.x;
//          i < n; i += blockDim.x * gridDim.x) {
//         func(i);
//     }
// }

// int main() {
//     int n = 65536;
//     float a = 3.14f;
//     thrust::host_vector<float> x_host(n);
//     thrust::host_vector<float> y_host(n);

//     for (int i = 0; i < n; i++) {
//         x_host[i] = std::rand() * (1.f / RAND_MAX);
//         y_host[i] = std::rand() * (1.f / RAND_MAX);
//     }

//     thrust::device_vector<float> x_dev = x_host;
//     thrust::device_vector<float> y_dev = x_host;

//     parallel_for<<<n / 512, 128>>>(n, [a, x_dev = x_dev.data(), y_dev = y_dev.data()] __device__ (int i) {
//         x_dev[i] = a * x_dev[i] + y_dev[i];
//     });

//     x_host = x_dev;

//     for (int i = 0; i < n; i++) {
//         printf("x[%d] = %f\n", i, x_host[i]);
//     }

//     return 0;
// }

template<class T>
__global__ void parallel_transpose(T out , T in ,int nx , int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= nx || y >= ny) return ; // 防止越界
    out[y * nx + x] = in[ x * nx + y];
}


int main(){
    int nx = 1 << 12;
    int ny = 1 << 12;
    thrust::host_vector<int> host_in(nx * ny);
    thrust::host_vector<int> host_out(nx * ny);

    for(int i = 0 ; i < nx * ny ; i++){
        host_in[i] = i;
    }

    thrust::device_vector<int> dev_in = host_in;
    thrust::device_vector<int> dev_out = host_out;

    TICK(parallel_transpose);
    parallel_transpose<<<dim3(nx/32, ny/32 , 1), dim3(32, 32, 1)>>>
        (dev_out.data(), dev_in.data() , nx , ny);
    host_out = dev_out;
    TOCK(parallel_transpose);

    for(int x = 0;  x < nx ; x++){
        for(int y = 0; y < ny; y++){
            if(host_out [y * nx + x] != host_in[x * nx + y]){
                printf("Wrong At x = %d , y = %d : % d != %d \n", x, y,
                   host_out [y * nx + x], host_in[x * nx + y] );
                return -1;
            }
        }
    }

    printf("All Correct!\n");
    return 0;
}
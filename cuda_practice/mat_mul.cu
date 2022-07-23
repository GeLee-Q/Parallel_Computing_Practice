
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>
#include "CudaAllocator.h"
#include "ticktock.h"

template <int blockSize, class T>
__global__ void parallel_mul(T * c, T const * a , T const * b, int M, int N, int K) {
    int m = threadIdx.x + blockIdx.x * blockDim.x;
    int n = threadIdx.y + blockIdx.y * blockDim.y;

    if(m >= M || n >= N) return;

    int tmp = 0;
    for(int t = 0 ; t < K; t++){
        tmp += a[m * K + t] * b[t * N + n];
    } 

    c[m * N + n] = tmp;
}


int main() {
    int m = 1 << 11;
    int k = 1 << 11;
    int n = 1 << 11;
    /* 
    CPU 初始化
    GPU 初始化
    */

    std::vector<int> c_a( m * k );
    std::vector<int> c_b( k * n );
    std::vector<int> c_res(m * n , 0);


    std::vector<int, CudaAllocator<int>> g_a(m * k);
    std::vector<int, CudaAllocator<int>> g_b(k * n);
    std::vector<int, CudaAllocator<int>> g_res(m * n , 0);
    for (int i = 0; i < m * k; i++) {
        int cur = rand() % 100;
        c_a[i] = cur;
        g_a[i] = cur;
    }

    for (int i = 0; i < k * n; i++) {
        int cur = rand() % 100;
        c_b[i] = cur;
        g_b[i] = cur;
    }

    /* 
    CPU 计算
     */
    
    TICK(cpu_time)
    for(int i = 0 ; i < m; i++){
        for(int j = 0; j < n; j++){
            int c = 0;
            for(int t = 0; t < k; t++){
                c += c_a[i * k + t] * c_b[t * n + j];
            }
            c_res [i * n + j]= c;
        }   
    }
    TOCK(cpu_time)

    

    TICK(gpu_mul);
    parallel_mul<32><<<dim3((m + 32 -1 ) / 32, (n + 32 - 1) / 32, 1), dim3(32, 32, 1)>>>
        (g_res.data(), g_a.data(), g_b.data(),m, n, k);
    checkCudaErrors(cudaDeviceSynchronize());
    TOCK(gpu_mul);

    for (int x = 0; x < m; x++) {
        for (int y = 0; y < n; y++) {
            if (c_res[x * n + y] != g_res[x * n + y]) {
                printf("Wrong At x=%d,y=%d: %d != %d\n", x, y,
                      c_res[x * n + y],  g_res[x * n + y]);
                return -1;
            }
        }
    }

    printf("All Correct!\n");
    return 0;
}

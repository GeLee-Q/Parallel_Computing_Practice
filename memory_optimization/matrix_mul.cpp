#include <iostream>
//#include <x86intrin.h>  // _mm 系列指令都来自这个头文件
#include <xmmintrin.h>  // 如果上面那个不行，试试这个
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <omp.h>

#include <mylib/ndarray.h>
#include <mylib/ticktock.h>

constexpr int n = 1 << 10;

ndarray<2, float> a(n, n);
ndarray<2, float> b(n, n);
ndarray<2, float> c(n, n);

void BM_matmul(benchmark::State& bm) {
    for (auto _ : bm) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                for (int t = 0; t < n; t++) {
                    a(i, j) += b(i, t) * c(t, j);
                }
            }
        }
    }
}
BENCHMARK(BM_matmul);

void BM_matmul_blocked(benchmark::State& bm) {
    for (auto _ : bm) {
        for (int j = 0; j < n; j++) {
            for (int iBase = 0; iBase < n; iBase += 32) {
                for (int t = 0; t < n; t++) {
                    for (int i = iBase; i < iBase + 32; i++) {
                        a(i, j) += b(i, t) * c(t, j);
                    }
                }
            }
        }
    }
}
BENCHMARK(BM_matmul_blocked);

void BM_matmul_blocked_both(benchmark::State& bm) {
    for (auto _ : bm) {
        for (int jBase = 0; jBase < n; jBase += 16) {
            for (int iBase = 0; iBase < n; iBase += 16) {
                for (int j = jBase; j < jBase + 16; j++) {
                    for (int t = 0; t < n; t++) {
                        for (int i = iBase; i < iBase + 16; i++) {
                            a(i, j) += b(i, t) * c(t, j);
                        }
                    }
                }
            }
        }
    }
}
BENCHMARK(BM_matmul_blocked_both);

BENCHMARK_MAIN();



// -----------------------------------------------------------------
// Benchmark                       Time             CPU   Iterations
// -----------------------------------------------------------------
// BM_matmul              4993302648 ns   4991044993 ns            1
// BM_matmul_blocked       549093089 ns    548267631 ns            1
// BM_matmul_blocked_both  656365535 ns    655582080 ns            1
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <xmmintrin.h>
#include <omp.h>
#include <mylib/ticktock.h>
#include <tbb/cache_aligned_allocator.h>
#include <mylib/alignalloc.h>

using namespace std;

typedef long long LL;





constexpr LL nx = 1 << 12;
constexpr LL ny = 1 << 13;

vector<float> a(nx * ny);
//array<array<float, nx>, ny> a;



void BM_yx_loop_yx_array(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                a[y * nx + x] = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_yx_loop_yx_array);

void BM_xy_loop_yx_array(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for collapse(2)
        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
                a[y * nx + x] = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_xy_loop_yx_array);

void BM_yx_loop_xy_array(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for collapse(2)
        for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                a[y + x * ny] = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_yx_loop_xy_array);

void BM_xy_loop_xy_array(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for collapse(2)
        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
                a[y + x * ny] = 1;
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_xy_loop_xy_array);


BENCHMARK_MAIN();



//---------------------------------------------------- -
//Benchmark           Time             CPU   Iterations
//---------------------------------------------------- -
//BM_yx_yx     25546222 ns     17578125 ns           32
//BM_xy_xy    471703700 ns    304687500 ns            2
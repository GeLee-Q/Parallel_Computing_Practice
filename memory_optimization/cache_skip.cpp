#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <xmmintrin.h>
#include <omp.h>
#include <tbb/parallel_do.h>


constexpr long long  n = 1 << 28;
std::vector<float> a(n);  // 1GB

void BM_skip1(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for
        for (long long i = 0; i < n; i += 1) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip1);

void BM_skip2(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for
        for (long long i = 0; i < n; i += 2) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip2);

void BM_skip4(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for
        for (long long i = 0; i < n; i += 4) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip4);

void BM_skip8(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for
        for (long long i = 0; i < n; i += 8) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip8);

void BM_skip16(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for
        for (long long i = 0; i < n; i += 16) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip16);

void BM_skip32(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for
        for (long long i = 0; i < n; i += 32) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip32);

void BM_skip64(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for
        for (long long i = 0; i < n; i += 64) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip64);

void BM_skip128(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for
        for (long long i = 0; i < n; i += 128) {
            a[i] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_skip128);

BENCHMARK_MAIN();


//---------------------------------------------------- -
//Benchmark           Time             CPU   Iterations
//---------------------------------------------------- -
//BM_skip1    102646560 ns     68750000 ns           10
//BM_skip2     99033136 ns     68181818 ns           11
//BM_skip4    100191140 ns     71875000 ns           10
//BM_skip8     99176482 ns     76704545 ns           11
//BM_skip16    98516089 ns     85069444 ns            9
//
//BM_skip32    53514369 ns     41992188 ns           16
//BM_skip64    27356263 ns     21354167 ns           30
//BM_skip128   16106304 ns     15066964 ns           56

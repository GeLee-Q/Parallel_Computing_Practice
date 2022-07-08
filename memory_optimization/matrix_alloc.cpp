#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <array>
#include <benchmark/benchmark.h>
#include <xmmintrin.h>
#include <omp.h>
#include <tbb/cache_aligned_allocator.h>

#include <mylib/ticktock.h>
#include <mylib/alignalloc.h>


using namespace std;

typedef long long LL;

//constexpr LL n = 1 << 20;



constexpr size_t n = 1 << 9;

static uint32_t randomize(uint32_t i) {
    i = (i ^ 61) ^ (i >> 16);
    i *= 9;
    i ^= i << 4;
    i *= 0x27d4eb2d;
    i ^= i >> 15;
    return i;
}

void BM_java_alloc(benchmark::State& bm) {
    for (auto _ : bm) {
        std::vector<std::vector<std::vector<float>>> a(n, std::vector<std::vector<float>>(n, std::vector<float>(n)));
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_java_alloc);

void BM_flatten_alloc(benchmark::State& bm) {
    for (auto _ : bm) {
        std::vector<float> a(n * n * n);
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_flatten_alloc);

void BM_java(benchmark::State& bm) {
    std::vector<std::vector<std::vector<float>>> a(n, std::vector<std::vector<float>>(n, std::vector<float>(n)));
    for (auto _ : bm) {
        for (int i = 0; i < n * n * n; i++) {
            int x = randomize(i) % n;
            int y = randomize(i ^ x) % n;
            int z = randomize(i ^ y) % n;
            a[x][y][z] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_java);

void BM_flatten(benchmark::State& bm) {
    std::vector<float> a(n * n * n);
    for (auto _ : bm) {
        for (int i = 0; i < n * n * n; i++) {
            int x = randomize(i) % n;
            int y = randomize(i ^ x) % n;
            int z = randomize(i ^ y) % n;
            a[(x * n + y) * n + z] = 1;
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_flatten);

BENCHMARK_MAIN();


//---------------------------------------------------------- -
//Benchmark                 Time             CPU   Iterations
//---------------------------------------------------------- -
//BM_java_alloc     334151500 ns    328125000 ns            2
//BM_flatten_alloc  123628560 ns    121875000 ns            5
//BM_java          8364048600 ns   8359375000 ns            1
//BM_flatten       1768750800 ns   1765625000 ns            1
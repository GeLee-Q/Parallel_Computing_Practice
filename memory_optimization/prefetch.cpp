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

using namespace std;

typedef long long LL;

constexpr LL  n = 1 << 27; // 512M

vector<float> a(n);

void BM_ordered(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for
        for (LL i = 0; i < n; i++) {
            benchmark::DoNotOptimize(a[i]);
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_ordered);

static uint32_t randomize(uint32_t i) {
    i = (i ^ 61) ^ (i >> 16);
    i *= 9;
    i ^= i << 4;
    i *= 0x27d4ebed;
    i ^= i >> 15;
    return i;
}


void BM_random(benchmark::State& bm) {


    for (auto _ : bm) {
#pragma omp parallel for
        for (LL i = 0; i < n; i ++) {
            LL r = randomize(i) % n;
            benchmark::DoNotOptimize(a[r]);

        }
        benchmark::DoNotOptimize(a);
       
    }
}
BENCHMARK(BM_random);

void BM_random_64B(benchmark::State& bm) {


    for (auto _ : bm) {
#pragma omp parallel for
        for (LL i = 0; i < n / 16; i++) {
            LL r = randomize(i) % (n / 16);
            for (int j = 0; j < 16; j++) {
                benchmark::DoNotOptimize(a[r * 16 + j]);
            }
        }
        benchmark::DoNotOptimize(a);

    }
}
BENCHMARK(BM_random_64B);

void BM_random_64B_prefetch(benchmark::State& bm) {
    for (auto _ : bm) {
#pragma omp parallel for
        for (LL i = 0; i < n / 16; i++) {
            LL next_r = randomize(i + 64) % (n / 16);
            _mm_prefetch(&a[next_r * 16], _MM_HINT_T0);
            LL r = randomize(i) % (n / 16);
            for (LL j = 0; j < 16; j++) {
                benchmark::DoNotOptimize(a[r * 16 + j]);
            }
        }
        benchmark::DoNotOptimize(a);
    }
}
BENCHMARK(BM_random_64B_prefetch);


void BM_random_4KB(benchmark::State& bm) {


    for (auto _ : bm) {
#pragma omp parallel for
        for (LL i = 0; i < n / 1024; i++) {
            LL r = randomize(i) % (n / 1024);
            for (int j = 0; j < 1024; j++) {
                benchmark::DoNotOptimize(a[r * 1024 + j]);
            }
        }
        benchmark::DoNotOptimize(a);

    }
    
}
BENCHMARK(BM_random_4KB);


void BM_random_4KB_aligned(benchmark::State& bm) {

    float* a = (float*)_mm_malloc(n * sizeof(float), 4096);
    memset(a, 0, n * sizeof(float));
    for (auto _ : bm) {
#pragma omp parallel for
        for (LL i = 0; i < n / 1024; i++) {
            LL r = randomize(i) % (n / 1024);
            for (int j = 0; j < 1024; j++) {
                benchmark::DoNotOptimize(a[r * 1024 + j]);
            }
        }
        benchmark::DoNotOptimize(a);

    }

}
BENCHMARK(BM_random_4KB_aligned);




BENCHMARK_MAIN();


//--------------------------------------------------------
//Benchmark              Time             CPU   Iterations
//--------------------------------------------------------
//BM_ordered      31680775 ns     28125000 ns           20
//BM_random       44303012 ns     42279412 ns           17
//BM_random_64B   27419696 ns     25111607 ns           28
//BM_random_4KB   26230910 ns     24479167 ns           30


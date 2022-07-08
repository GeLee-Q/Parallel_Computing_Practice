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

constexpr LL  n = 1 << 26; // 512M



void BM_aos(benchmark::State& bm) {

    struct MyClass {
        float x;
        float y;
        float z;
    };

    vector<MyClass> mc(n);


    for (auto _ : bm) {
#pragma omp parallel for
        for (LL i = 0; i < n; i++) {
            mc[i].x += 1;
            mc[i].y += 1;
            mc[i].z += 1;
        }
        benchmark::DoNotOptimize(mc);
    }
}
BENCHMARK(BM_aos);

void BM_soa(benchmark::State& bm) {

    vector<float> mc_x(n);
    vector<float> mc_y(n);
    vector<float> mc_z(n);


    for (auto _ : bm) {
#pragma omp parallel for
        for (LL i = 0; i < n; i ++) {
            mc_x[i]++;
            mc_y[i]++;
            mc_z[i]++;

        }
        benchmark::DoNotOptimize(mc_x);
        benchmark::DoNotOptimize(mc_y);
        benchmark::DoNotOptimize(mc_z);
    }
}
BENCHMARK(BM_soa);

void BM_aosoa(benchmark::State& bm) {

    struct MyClass {
        float x[1024];
        float y[1024];
        float z[1024];
    };

    vector<MyClass> mc(n / 1024);


    for (auto _ : bm) {
#pragma omp parallel for
        for (LL i = 0; i < n / 1024; i++) {
//#pragma omp simd
            for (int j = 0; j < 1024; j++) {
                mc[i].x[j] += 1;
                mc[i].y[j] += 1;
                mc[i].z[j] += 1;
            }
            benchmark::DoNotOptimize(mc);

        }
    }
}
BENCHMARK(BM_aosoa);


//void BM_aosoa_16(benchmark::State& bm) {
//
//    struct MyClass {
//        float x[16];
//        float y[16];
//        float z[16];
//    };
//
//    vector<MyClass> mc(n / 16);
//
//
//    for (auto _ : bm) {
//#pragma omp parallel for
//        for (LL i = 0; i < n / 16; i++) {
//            for (int j = 0; j < 16; j++) {
//                mc[i].x[j] = mc[i].x[j] + mc[i].y[j];
//            }
//            benchmark::DoNotOptimize(mc);
//
//        }
//    }
//}
//BENCHMARK(BM_aosoa_16);

//void BM_aosoa_2048(benchmark::State& bm) {
//
//    struct MyClass {
//        float x[2048];
//        float y[2048];
//        float z[2048];
//    };
//
//    vector<MyClass> mc(n / 2048);
//
//
//    for (auto _ : bm) {
//#pragma omp parallel for
//        for (LL i = 0; i < n / 2048; i++) {
//            for (int j = 0; j < 2048; j++) {
//                mc[i].x[j] = mc[i].x[j] + mc[i].y[j];
//            }
//            benchmark::DoNotOptimize(mc);
//
//        }
//    }
//}
//BENCHMARK(BM_aosoa_2048);





BENCHMARK_MAIN();

//--------------------------------------------------------
//Benchmark              Time             CPU   Iterations
//--------------------------------------------------------
//BM_aos          73930750 ns     56250000 ns           10
//BM_soa          35302992 ns     30000000 ns           25
//BM_aosoa        36422382 ns     28409091 ns           22
//BM_aosoa_16     53888073 ns     51136364 ns           11
//BM_aosoa_2048   35885952 ns     28750000 ns           25



//---------------------------------------------------- -
//Benchmark           Time             CPU   Iterations
//---------------------------------------------------- -
//BM_aos       75143320 ns     70312500 ns           10
//BM_soa       81357256 ns     65972222 ns            9
//BM_aosoa     81570180 ns     62500000 ns           10
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>

#include "tensor.hpp"

typedef Tensor (*MatmulFn)(const Tensor&, const Tensor&);

static Tensor run_matmul(const Tensor& lhs, const Tensor& rhs) {
    return lhs.matmul(rhs);
}

static Tensor run_matmul_naive(const Tensor& lhs, const Tensor& rhs) {
    return lhs.matmul_naive(rhs);
}

void benchmark_matmul(const char* label, MatmulFn fn, int N, int iterations, int warmup_iters, bool check_against_naive) {
    Tensor lhs({N, N});
    Tensor rhs({N, N});

    // fill in deterministic values
    for (int i = 0; i < N * N; i++) {
        lhs.data[i] = (i % 13) * 0.01f;
        rhs.data[i] = (i % 17) * 0.01f;
    }

    // warmup
    float warmup_checksum = 0.0f;
    for (int i = 0; i < warmup_iters; i++) {
        Tensor out = fn(lhs, rhs);
        warmup_checksum += out.data[0];
    }

    // start benchmark
    auto start = std::chrono::steady_clock::now();

    float checksum = 0.0f;
    Tensor out;
    for (int i = 0; i < iterations; i++) {
        out = fn(lhs, rhs);
        checksum += out.data[0];
    }

    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    double seconds = elapsed.count();
    double ms = seconds * 1000.0;
    double avg_ms = ms / iterations;

    double flops = static_cast<double>(iterations) * 2.0 * N * N * N;
    double gflops = flops / 1e9;
    double gflops_per_sec = gflops / seconds;

    std::cout
        << "kernel = " << label
        << ", N = " << N
        << ", iters = " << iterations
        << ", warmup_iters = " << warmup_iters
        << ", avg_ms = " << avg_ms
        << ", gflop/s = " << gflops_per_sec
        << ", checksum = " << checksum
        << ", warmup_checksum = " << warmup_checksum;

    // test correct output; asserts compile out under -DNDEBUG so check at runtime
    if (check_against_naive) {
        Tensor gold = lhs.matmul_naive(rhs);
        float max_diff = 0.0f;
        for (size_t i = 0; i < gold.numel(); i++) {
            float diff = std::fabs(gold.data[i] - out.data[i]);
            if (diff > max_diff) max_diff = diff;
        }
        std::cout << ", max_diff_vs_naive = " << max_diff;
        if (max_diff > 1e-2f) {
            std::cout << " MISMATCH" << std::endl;
            std::exit(1);
        }
    }

    std::cout << std::endl;
}

int main(int argc, char** argv) {
    const char* label = "matmul";
    MatmulFn fn = run_matmul;

    if (argc > 1) {
        if (std::strcmp(argv[1], "naive") == 0) {
            label = "naive";
            fn = run_matmul_naive;
        } else if (std::strcmp(argv[1], "matmul") != 0) {
            std::cerr << "usage: " << argv[0] << " [matmul|naive]" << std::endl;
            return 1;
        }
    }

    struct { int N, iters, warmup; } sizes[] = {
        {64, 512, 128},
        {128, 256, 64},
        {256, 64, 16},
        {512, 16, 4},
        {1024, 8, 2},
        {2048, 4, 1},
        {4096, 3, 1},
    };

    for (const auto& s : sizes) {
        // elementwise gold check only at sizes where naive stays cheap
        benchmark_matmul(label, fn, s.N, s.iters, s.warmup, s.N <= 1024);
    }

    return 0;
}

#include <utility>
#include <cassert>
#include <cstring>
#include <iostream>
#include <chrono>
#include <thread>

#include "tensor.hpp"


#include <utility>
#include <cassert>
#include <cstring>
#include <iostream>
#include <chrono>

#include "tensor.hpp"

void benchmark_matmul(int N, int iterations, int warmup_iters) {
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
        Tensor out = lhs.matmul_2d_threaded(rhs, 11);
        warmup_checksum += out.data[0];
    }

    // start benchmark
    auto start = std::chrono::steady_clock::now();

    float checksum = 0.0f;
    Tensor out;
    for (int i = 0; i < iterations; i++) {
        Tensor out = lhs.matmul_2d_threaded(rhs, 11);
        checksum += out.data[0];
    }

    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    double seconds = elapsed.count();
    double ms = seconds * 1000.0;
    // double avg_seconds = seconds / iterations;
    double avg_ms = ms / iterations;

    double flops = static_cast<double>(iterations) * 2.0 * N * N * N;
    double gflops = flops / 1e9;
    double gflops_per_sec = gflops / seconds;

    std::cout
        << "N = " << N
        << ", iters = " << iterations
        << ", warmup_iters = " << warmup_iters
        << ", avg_ms = " << avg_ms
        << ", gflop/s = " << gflops_per_sec
        << ", checksum = " << checksum
        << ", warmup_checksum = " << warmup_checksum
        << std::endl;
}

int main() {
    benchmark_matmul(64, 512, 128);
    benchmark_matmul(128, 256, 64);
    benchmark_matmul(256, 64, 16);
    benchmark_matmul(512, 16, 4);
    benchmark_matmul(1024, 8, 2);
    benchmark_matmul(2048, 4, 1);
    benchmark_matmul(4096, 3, 1);

    return 0;
}
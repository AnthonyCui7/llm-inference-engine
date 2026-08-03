#include <chrono>
#include <cstdlib>
#include <iostream>

#include "generate.hpp"

int main(int argc, char** argv) {
    const char* weights_path = argc > 1 ? argv[1] : "gpt2.bin";
    int new_tokens = argc > 2 ? std::atoi(argv[2]) : 32;
    int iterations = argc > 3 ? std::atoi(argv[3]) : 3;
    int warmup_iters = 1;

    Fp32FileLoader loader(weights_path);
    ModelWeights model = load_model(loader);

    // "The capital of France is"
    std::vector<int> prompt = {464, 3139, 286, 4881, 318};

    // greedy decoding is deterministic, so the id sum makes a stable checksum
    long warmup_checksum = 0;
    for (int i = 0; i < warmup_iters; i++) {
        std::vector<int> out = generate(model, prompt, new_tokens);
        for (int t : out) warmup_checksum += t;
    }

    auto start = std::chrono::steady_clock::now();

    long checksum = 0;
    for (int i = 0; i < iterations; i++) {
        std::vector<int> out = generate(model, prompt, new_tokens);
        for (int t : out) checksum += t;
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double seconds = elapsed.count();
    double ms_per_token = seconds * 1000.0 / (static_cast<double>(iterations) * new_tokens);
    double tokens_per_sec = (static_cast<double>(iterations) * new_tokens) / seconds;

    std::cout
        << "prompt_tokens = " << prompt.size()
        << ", new_tokens = " << new_tokens
        << ", iters = " << iterations
        << ", warmup_iters = " << warmup_iters
        << ", ms_per_token = " << ms_per_token
        << ", tok/s = " << tokens_per_sec
        << ", checksum = " << checksum
        << ", warmup_checksum = " << warmup_checksum
        << std::endl;

    return 0;
}

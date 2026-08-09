#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>

#include "generate.hpp"
#include "speculative.hpp"

// runs generation iterations around a warmup and prints one stats line;
// the token id sum makes a stable checksum since every mode is seeded
static void bench_loop(const char* mode, int new_tokens, int iterations, int warmup_iters,
                       const std::function<std::vector<int>(int)>& run) {
    long warmup_checksum = 0;
    for (int i = 0; i < warmup_iters; i++) {
        for (int t : run(i)) warmup_checksum += t;
    }

    auto start = std::chrono::steady_clock::now();

    long checksum = 0;
    for (int i = 0; i < iterations; i++) {
        for (int t : run(i)) checksum += t;
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double seconds = elapsed.count();
    double ms_per_token = seconds * 1000.0 / (static_cast<double>(iterations) * new_tokens);
    double tokens_per_sec = (static_cast<double>(iterations) * new_tokens) / seconds;

    std::cout
        << "mode = " << mode
        << ", new_tokens = " << new_tokens
        << ", iters = " << iterations
        << ", warmup_iters = " << warmup_iters
        << ", ms_per_token = " << ms_per_token
        << ", tok/s = " << tokens_per_sec
        << ", checksum = " << checksum
        << ", warmup_checksum = " << warmup_checksum
        << std::endl;
}

int main(int argc, char** argv) {
    const char* weights_path = argc > 1 ? argv[1] : "gpt2.bin";
    int new_tokens = argc > 2 ? std::atoi(argv[2]) : 32;
    int iterations = argc > 3 ? std::atoi(argv[3]) : 3;
    const char* draft_path = argc > 4 ? argv[4] : nullptr;
    int draft_len = argc > 5 ? std::atoi(argv[5]) : 4;
    float temperature = argc > 6 ? static_cast<float>(std::atof(argv[6])) : 1.0f;
    int warmup_iters = 1;

    Fp32FileLoader loader(weights_path);
    ModelWeights model = load_model(loader);

    // "The capital of France is"
    std::vector<int> prompt = {464, 3139, 286, 4881, 318};

    if (draft_path == nullptr) {
        bench_loop("greedy", new_tokens, iterations, warmup_iters, [&](int) {
            return generate(model, prompt, new_tokens);
        });
        return 0;
    }

    // with a draft model, compare plain sampling against speculative
    // decoding at the same temperature and matching per-iteration seeds
    Fp32FileLoader draft_loader(draft_path);
    ModelWeights draft = load_model(draft_loader);

    bench_loop("sampled", new_tokens, iterations, warmup_iters, [&](int i) {
        std::mt19937 rng(42 + i);
        return generate_sampled(model, prompt, new_tokens, temperature, rng);
    });

    bench_loop("speculative", new_tokens, iterations, warmup_iters, [&](int i) {
        std::mt19937 rng(42 + i);
        return generate_speculative(model, draft, prompt, new_tokens, draft_len, temperature, rng);
    });

    return 0;
}

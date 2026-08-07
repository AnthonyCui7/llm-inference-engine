#include <cassert>
#include <cmath>

#include "sample.hpp"

std::vector<float> logits_to_probs(const float* logits, int vocab, float temperature) {
    assert(vocab > 0);
    assert(temperature > 0.0f);

    // softmax with the usual max subtraction for stability
    float max = logits[0];
    for (int t = 1; t < vocab; t++) {
        if (logits[t] > max) max = logits[t];
    }

    std::vector<float> probs(vocab);
    double sum = 0.0;
    for (int t = 0; t < vocab; t++) {
        probs[t] = std::exp((logits[t] - max) / temperature);
        sum += probs[t];
    }

    for (int t = 0; t < vocab; t++) {
        probs[t] = static_cast<float>(probs[t] / sum);
    }

    return probs;
}

int sample_from(const std::vector<float>& probs, std::mt19937& rng) {
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    float u = uniform(rng);

    float cumulative = 0.0f;
    for (size_t t = 0; t < probs.size(); t++) {
        cumulative += probs[t];
        if (u < cumulative) return static_cast<int>(t);
    }

    // float rounding can leave the total a hair under 1
    return static_cast<int>(probs.size()) - 1;
}

int sample_token(const float* logits, int vocab, float temperature, std::mt19937& rng) {
    return sample_from(logits_to_probs(logits, vocab, temperature), rng);
}

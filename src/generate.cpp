#include "generate.hpp"

// argmax over the last position of [1, seq, vocab] logits
static int next_token(const Tensor& logits) {
    assert(logits.ndim == 3 && logits.shape[0] == 1);

    int seq = logits.shape[1];
    int vocab = logits.shape[2];
    const float* row = logits.data + static_cast<size_t>(seq - 1) * vocab;

    int best = 0;
    for (int t = 1; t < vocab; t++) {
        if (row[t] > row[best]) best = t;
    }

    return best;
}

std::vector<int> generate(const ModelWeights& model, const std::vector<int>& prompt,
                          int max_new_tokens) {
    assert(!prompt.empty());
    assert(static_cast<int>(prompt.size()) + max_new_tokens <= model.config.max_seq);

    std::vector<int> tokens = prompt;
    if (max_new_tokens == 0) return tokens;

    // prefill the whole prompt in one pass, then each new token only runs
    // its single query against the cached history
    KVCache cache(model.config.num_layers, model.config.max_seq, model.config.hidden);

    Tensor prompt_ids({1, static_cast<int>(prompt.size())});
    for (size_t i = 0; i < prompt.size(); i++) {
        prompt_ids.data[i] = static_cast<float>(prompt[i]);
    }

    Tensor logits = model_forward_cached(model, prompt_ids, cache);

    for (int step = 0; step < max_new_tokens; step++) {
        int next = next_token(logits);
        tokens.push_back(next);
        if (step + 1 == max_new_tokens) break;

        Tensor next_ids({1, 1});
        next_ids.data[0] = static_cast<float>(next);
        logits = model_forward_cached(model, next_ids, cache);
    }

    return tokens;
}

std::vector<int> generate_sampled(const ModelWeights& model, const std::vector<int>& prompt,
                                  int max_new_tokens, float temperature, std::mt19937& rng) {
    assert(!prompt.empty());
    assert(static_cast<int>(prompt.size()) + max_new_tokens <= model.config.max_seq);

    std::vector<int> tokens = prompt;
    if (max_new_tokens == 0) return tokens;

    KVCache cache(model.config.num_layers, model.config.max_seq, model.config.hidden);

    Tensor prompt_ids({1, static_cast<int>(prompt.size())});
    for (size_t i = 0; i < prompt.size(); i++) {
        prompt_ids.data[i] = static_cast<float>(prompt[i]);
    }

    Tensor logits = model_forward_cached(model, prompt_ids, cache);
    int vocab = model.config.vocab_size;

    for (int step = 0; step < max_new_tokens; step++) {
        const float* row = logits.data + static_cast<size_t>(logits.shape[1] - 1) * vocab;
        int next = sample_token(row, vocab, temperature, rng);
        tokens.push_back(next);
        if (step + 1 == max_new_tokens) break;

        Tensor next_ids({1, 1});
        next_ids.data[0] = static_cast<float>(next);
        logits = model_forward_cached(model, next_ids, cache);
    }

    return tokens;
}

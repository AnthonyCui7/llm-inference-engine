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

    for (int step = 0; step < max_new_tokens; step++) {
        Tensor ids({1, static_cast<int>(tokens.size())});
        for (size_t i = 0; i < tokens.size(); i++) {
            ids.data[i] = static_cast<float>(tokens[i]);
        }

        tokens.push_back(next_token(model_forward(model, ids)));
    }

    return tokens;
}

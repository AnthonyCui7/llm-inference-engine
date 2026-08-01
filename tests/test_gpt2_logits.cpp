/*
End to end check of the forward pass against huggingface logits.

  ./bin/test_gpt2_logits [weights.bin] [reference.bin]

The reference file comes from scripts/reference_logits.py.
*/
#include <cmath>
#include <cstdio>

#include "model.hpp"
#include "weights.hpp"

int main(int argc, char** argv) {
    const char* weights_path = argc > 1 ? argv[1] : "gpt2.bin";
    const char* reference_path = argc > 2 ? argv[2] : "reference.bin";

    Fp32FileLoader loader(weights_path);
    ModelWeights model = load_model(loader);

    Fp32FileLoader reference(reference_path);
    Tensor token_ids = reference.load("token_ids");
    Tensor expected = reference.load("logits");

    int seq = token_ids.shape[0];
    int vocab = expected.shape[1];

    Tensor ids({1, seq});
    for (int i = 0; i < seq; i++) {
        ids.data[i] = token_ids.data[i];
    }

    Tensor logits = model_forward(model, ids);

    if (logits.shape[1] != seq || logits.shape[2] != vocab) {
        std::fprintf(stderr, "shape mismatch: got [%d, %d], reference [%d, %d]\n",
                     logits.shape[1], logits.shape[2], seq, vocab);
        return 1;
    }

    float max_diff = 0.0f;
    double sum_diff = 0.0;
    int argmax_matches = 0;

    for (int i = 0; i < seq; i++) {
        const float* got = logits.data + static_cast<size_t>(i) * vocab;
        const float* ref = expected.data + static_cast<size_t>(i) * vocab;

        int got_best = 0;
        int ref_best = 0;
        for (int t = 0; t < vocab; t++) {
            float diff = std::fabs(got[t] - ref[t]);
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;

            if (got[t] > got[got_best]) got_best = t;
            if (ref[t] > ref[ref_best]) ref_best = t;
        }
        if (got_best == ref_best) argmax_matches++;
    }

    std::printf("positions: %d  max_abs_diff: %.6f  mean_abs_diff: %.8f  argmax_match: %d/%d\n",
                seq, max_diff, sum_diff / (static_cast<double>(seq) * vocab),
                argmax_matches, seq);

    if (max_diff > 1e-2f || argmax_matches != seq) {
        std::printf("MISMATCH\n");
        return 1;
    }

    std::printf("OK\n");
    return 0;
}

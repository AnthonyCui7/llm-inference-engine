/*
Command line driver: generation over a dumped weight file.

  ./bin/generate <weights.bin> <n_tokens> <token id> [token id ...]
  ./bin/generate <weights.bin> --draft <draft.bin> <n_tokens> <token id> [token id ...]

Plain runs decode greedily; with a draft model, tokens are sampled
speculatively at temperature 1. Token ids come from scripts/tokenizer.py,
and generated ids go back through the same script to get text out.
*/
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "generate.hpp"
#include "speculative.hpp"

int main(int argc, char** argv) {
    bool speculative = argc > 2 && std::strcmp(argv[2], "--draft") == 0;
    int arg = speculative ? 4 : 2;

    if (argc < arg + 2) {
        std::fprintf(stderr,
                     "usage: %s <weights.bin> [--draft <draft.bin>] <n_tokens> <token id> [token id ...]\n",
                     argv[0]);
        return 1;
    }

    Fp32FileLoader loader(argv[1]);
    ModelWeights model = load_model(loader);

    ModelWeights draft;
    if (speculative) {
        Fp32FileLoader draft_loader(argv[3]);
        draft = load_model(draft_loader);
    }

    int max_new_tokens = std::atoi(argv[arg]);
    std::vector<int> prompt;
    for (int i = arg + 1; i < argc; i++) {
        prompt.push_back(std::atoi(argv[i]));
    }

    std::vector<int> tokens;
    if (speculative) {
        std::mt19937 rng(42);
        tokens = generate_speculative(model, draft, prompt, max_new_tokens, 3, 1.0f, rng);
    } else {
        tokens = generate(model, prompt, max_new_tokens);
    }

    for (size_t i = 0; i < tokens.size(); i++) {
        std::printf("%d%s", tokens[i], i + 1 < tokens.size() ? " " : "\n");
    }

    return 0;
}

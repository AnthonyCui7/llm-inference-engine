/*
Command line driver: greedy generation over a dumped weight file.

  ./bin/generate <weights.bin> <n_tokens> <token id> [token id ...]

Token ids come from scripts/tokenizer.py, and generated ids go back
through the same script to get text out.
*/
#include <cstdio>
#include <cstdlib>

#include "generate.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <weights.bin> <n_tokens> <token id> [token id ...]\n", argv[0]);
        return 1;
    }

    Fp32FileLoader loader(argv[1]);
    ModelWeights model = load_model(loader);

    int max_new_tokens = std::atoi(argv[2]);
    std::vector<int> prompt;
    for (int i = 3; i < argc; i++) {
        prompt.push_back(std::atoi(argv[i]));
    }

    std::vector<int> tokens = generate(model, prompt, max_new_tokens);

    for (size_t i = 0; i < tokens.size(); i++) {
        std::printf("%d%s", tokens[i], i + 1 < tokens.size() ? " " : "\n");
    }

    return 0;
}

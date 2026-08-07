/*
Autoregressive generation on top of the model forward pass.
*/
#pragma once

#include <vector>

#include "model.hpp"
#include "sample.hpp"

// greedy decoding over the kv cache: prefills the prompt once, then each
// step feeds only the newest token and takes the argmax over its logits;
// returns prompt plus new tokens
std::vector<int> generate(const ModelWeights& model, const std::vector<int>& prompt,
                          int max_new_tokens);

// same loop, but each token is drawn from the temperature scaled
// distribution instead of the argmax
std::vector<int> generate_sampled(const ModelWeights& model, const std::vector<int>& prompt,
                                  int max_new_tokens, float temperature, std::mt19937& rng);

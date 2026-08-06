/*
Autoregressive generation on top of the model forward pass.
*/
#pragma once

#include <vector>

#include "model.hpp"

// greedy decoding over the kv cache: prefills the prompt once, then each
// step feeds only the newest token and takes the argmax over its logits;
// returns prompt plus new tokens
std::vector<int> generate(const ModelWeights& model, const std::vector<int>& prompt,
                          int max_new_tokens);

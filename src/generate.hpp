/*
Autoregressive generation on top of the model forward pass.
*/
#pragma once

#include <vector>

#include "model.hpp"

// greedy decoding: re-runs the full forward pass every step and takes the
// argmax over the last position's logits; returns prompt plus new tokens
std::vector<int> generate(const ModelWeights& model, const std::vector<int>& prompt,
                          int max_new_tokens);

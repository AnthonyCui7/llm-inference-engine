/*
Speculative decoding: a small draft model proposes a few tokens per round,
the target model scores them all in one batched forward pass, and
rejection sampling keeps the output distributed exactly as if the target
had decoded alone.
*/
#pragma once

#include "generate.hpp"

// draft_len proposals per round; both models must share a vocabulary.
// output matches generate_sampled on the target in distribution
std::vector<int> generate_speculative(const ModelWeights& target, const ModelWeights& draft,
                                      const std::vector<int>& prompt, int max_new_tokens,
                                      int draft_len, float temperature, std::mt19937& rng);

/*
Token sampling from logits.
*/
#pragma once

#include <random>
#include <vector>

// softmax over one row of logits at the given temperature
std::vector<float> logits_to_probs(const float* logits, int vocab, float temperature);

// draws an index from the distribution; probs must sum to about 1
int sample_from(const std::vector<float>& probs, std::mt19937& rng);

// samples from one row of logits at the given temperature
int sample_token(const float* logits, int vocab, float temperature, std::mt19937& rng);

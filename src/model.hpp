/*
GPT-2 style model built from the transformer blocks: learned token and
positional embeddings, block stack, final layernorm, tied unembedding.
*/
#pragma once

#include <string>

#include "tensor.hpp"
#include "transformer.hpp"
#include "weights.hpp"

// model dimensions, matching the GPT-2 family
struct ModelConfig {
    int vocab_size;
    int max_seq;
    int hidden;
    int num_heads;
    int num_layers;
};

// full set of model weights; the unembedding is tied to the token
// embedding, so lm_head is just wte transposed once up front
struct ModelWeights {
    ModelConfig config;
    Tensor wte;      // [vocab, hidden]
    Tensor wpe;      // [max_seq, hidden]
    std::vector<TransformerBlockWeights> blocks;
    Tensor final_gamma;
    Tensor final_beta;
    Tensor lm_head;  // [hidden, vocab]
};

// learned GPT-2 style embeddings: wte[token] + wpe[position] for every
// position in token_ids, [batch, seq] -> [batch, seq, hidden]
Tensor embed(const Tensor& wte, const Tensor& wpe, const Tensor& token_ids);

// full forward pass over pre tokenized input:
// [batch, seq] ids -> [batch, seq, vocab] logits
Tensor model_forward(const ModelWeights& model, const Tensor& token_ids);

// pulls a full model out of a weight file; expects the naming scheme
// written by scripts/dump_gpt2.py ("config", "wte", "wpe", "h0.w_q", ...)
ModelWeights load_model(const WeightLoader& loader);

/*
GPT-2 style model built from the transformer blocks: learned token and
positional embeddings, block stack, final layernorm, tied unembedding.
*/
#pragma once

#include "tensor.hpp"
#include "transformer.hpp"

// learned GPT-2 style embeddings: wte[token] + wpe[position] for every
// position in token_ids, [batch, seq] -> [batch, seq, hidden]
Tensor embed(const Tensor& wte, const Tensor& wpe, const Tensor& token_ids);

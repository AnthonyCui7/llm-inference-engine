/*
Attention ops for the transformer forward pass, built on the Tensor primitives.
*/
#pragma once

#include "tensor.hpp"

// causal scaled dot product attention over [..., seq, head_dim] tensors:
// softmax(Q K^T / sqrt(head_dim), masked so a query only sees keys at or
// before its own position) V
Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value);

// splits the hidden dim of [batch, seq, hidden] inputs into num_heads,
// runs causal attention per head, and merges the heads back
Tensor multi_head_attention(const Tensor& query, const Tensor& key, const Tensor& value, int num_heads);

// full self attention on [batch, seq, hidden]: q/k/v projections,
// multi head causal attention, output projection
Tensor self_attention(const Tensor& x, const Tensor& w_q, const Tensor& w_k,
                      const Tensor& w_v, const Tensor& w_o, int num_heads);

// pre layernorm attention block, GPT-2 style: x + self_attention(layernorm(x))
Tensor attention_block(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                       const Tensor& w_q, const Tensor& w_k, const Tensor& w_v,
                       const Tensor& w_o, int num_heads);

/*
Attention ops for the transformer forward pass, built on the Tensor primitives.
*/
#pragma once

#include "tensor.hpp"
#include "kv_cache.hpp"

// causal scaled dot product attention over [..., seq, head_dim] tensors:
// softmax(Q K^T / sqrt(head_dim), masked so a query only sees keys at or
// before its own position) V. q_offset is the absolute position of the
// first query row, for queries that continue a longer key/value history
// (kv cache decoding); the plain overload is full self attention where
// queries and keys line up
Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    int q_offset);
Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value);

// splits the hidden dim of [batch, seq, hidden] inputs into num_heads,
// runs causal attention per head, and merges the heads back; same
// q_offset story as scaled_dot_product_attention
Tensor multi_head_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                            int num_heads, int q_offset);
Tensor multi_head_attention(const Tensor& query, const Tensor& key, const Tensor& value, int num_heads);

// full self attention on [batch, seq, hidden]: biased q/k/v projections,
// multi head causal attention, biased output projection
Tensor self_attention(const Tensor& x, const Tensor& w_q, const Tensor& b_q,
                      const Tensor& w_k, const Tensor& b_k,
                      const Tensor& w_v, const Tensor& b_v,
                      const Tensor& w_o, const Tensor& b_o, int num_heads);

// self attention over the kv cache: x holds only the new positions of a
// single sequence, its keys/values are appended to the layer's cache, and
// the queries attend over the whole cached history
Tensor self_attention_cached(const Tensor& x, const Tensor& w_q, const Tensor& b_q,
                             const Tensor& w_k, const Tensor& b_k,
                             const Tensor& w_v, const Tensor& b_v,
                             const Tensor& w_o, const Tensor& b_o, int num_heads,
                             KVCache& cache, int layer);

// pre layernorm attention block, GPT-2 style: x + self_attention(layernorm(x))
Tensor attention_block(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                       const Tensor& w_q, const Tensor& b_q,
                       const Tensor& w_k, const Tensor& b_k,
                       const Tensor& w_v, const Tensor& b_v,
                       const Tensor& w_o, const Tensor& b_o, int num_heads);

// same block over the cached attention path
Tensor attention_block_cached(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                              const Tensor& w_q, const Tensor& b_q,
                              const Tensor& w_k, const Tensor& b_k,
                              const Tensor& w_v, const Tensor& b_v,
                              const Tensor& w_o, const Tensor& b_o, int num_heads,
                              KVCache& cache, int layer);

/*
Transformer blocks: pre layernorm attention and MLP sublayers with residuals,
and stacking into the full model.
*/
#pragma once

#include <vector>

#include "tensor.hpp"

// weights for one transformer block
struct TransformerBlockWeights {
    Tensor attn_gamma;   // layernorm before attention
    Tensor attn_beta;
    Tensor w_q;          // attention projections, [hidden, hidden]
    Tensor w_k;
    Tensor w_v;
    Tensor w_o;
    Tensor mlp_gamma;    // layernorm before the mlp
    Tensor mlp_beta;
    Tensor w_fc;         // mlp linears, [hidden, ff] and [ff, hidden]
    Tensor w_proj;
};

// pre layernorm mlp block: x + w_proj(gelu(w_fc(layernorm(x))))
Tensor mlp_block(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 const Tensor& w_fc, const Tensor& w_proj);

// one full block: attention sublayer then mlp sublayer, each with its own
// layernorm and residual
Tensor transformer_block(const Tensor& x, const TransformerBlockWeights& weights, int num_heads);

// runs x through a stack of blocks in order
Tensor transformer_stack(const Tensor& x, const std::vector<TransformerBlockWeights>& layers, int num_heads);

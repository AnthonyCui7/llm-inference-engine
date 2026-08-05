#include "transformer.hpp"
#include "attention.hpp"

Tensor mlp_block(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 const Tensor& w_fc, const Tensor& b_fc,
                 const Tensor& w_proj, const Tensor& b_proj) {
    assert(x.ndim == 3);

    int batch = x.shape[0];
    int seq = x.shape[1];
    int hidden = x.shape[2];

    assert(w_fc.ndim == 2 && w_fc.shape[0] == hidden);
    int ff = w_fc.shape[1];
    assert(w_proj.ndim == 2 && w_proj.shape[0] == ff && w_proj.shape[1] == hidden);
    assert(b_fc.ndim == 1 && b_fc.shape[0] == ff);
    assert(b_proj.ndim == 1 && b_proj.shape[0] == hidden);

    // pre layernorm: normalize the input, run the mlp, add back the residual
    Tensor normed = x.layernorm(gamma, beta);
    Tensor flat = normed.reshape({batch * seq, hidden});
    Tensor mlp = flat.matmul(w_fc).add(b_fc.reshape({1, ff})).gelu()
                     .matmul(w_proj).add(b_proj.reshape({1, hidden}))
                     .reshape({batch, seq, hidden});

    return x.add(mlp);
}

Tensor transformer_block(const Tensor& x, const TransformerBlockWeights& weights, int num_heads) {
    Tensor attn_out = attention_block(x, weights.attn_gamma, weights.attn_beta,
                                      weights.w_q, weights.b_q, weights.w_k, weights.b_k,
                                      weights.w_v, weights.b_v, weights.w_o, weights.b_o,
                                      num_heads);

    return mlp_block(attn_out, weights.mlp_gamma, weights.mlp_beta,
                     weights.w_fc, weights.b_fc, weights.w_proj, weights.b_proj);
}

Tensor transformer_stack(const Tensor& x, const std::vector<TransformerBlockWeights>& layers, int num_heads) {
    Tensor out = x;

    for (const TransformerBlockWeights& layer : layers) {
        out = transformer_block(out, layer, num_heads);
    }

    return out;
}

Tensor transformer_block_cached(const Tensor& x, const TransformerBlockWeights& weights,
                                int num_heads, KVCache& cache, int layer) {
    Tensor attn_out = attention_block_cached(x, weights.attn_gamma, weights.attn_beta,
                                             weights.w_q, weights.b_q, weights.w_k, weights.b_k,
                                             weights.w_v, weights.b_v, weights.w_o, weights.b_o,
                                             num_heads, cache, layer);

    return mlp_block(attn_out, weights.mlp_gamma, weights.mlp_beta,
                     weights.w_fc, weights.b_fc, weights.w_proj, weights.b_proj);
}

Tensor transformer_stack_cached(const Tensor& x, const std::vector<TransformerBlockWeights>& layers,
                                int num_heads, KVCache& cache) {
    assert(static_cast<int>(layers.size()) == static_cast<int>(cache.layers.size()));

    Tensor out = x;

    for (size_t i = 0; i < layers.size(); i++) {
        out = transformer_block_cached(out, layers[i], num_heads, cache, static_cast<int>(i));
    }

    return out;
}

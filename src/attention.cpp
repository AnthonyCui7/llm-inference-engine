#include <cmath>

#include "attention.hpp"

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value) {
    assert(query.ndim >= 2);
    assert(key.ndim == query.ndim && value.ndim == query.ndim);

    int d_k = query.shape[query.ndim - 1];
    int seq_q = query.shape[query.ndim - 2];
    int seq_k = key.shape[key.ndim - 2];

    assert(key.shape[key.ndim - 1] == d_k);
    assert(value.shape[value.ndim - 2] == seq_k);

    // the causal mask assumes query and key positions line up
    assert(seq_q == seq_k);

    Tensor key_t = key.transpose(key.ndim - 2, key.ndim - 1);
    Tensor scores = query.matmul(key_t).mul(1.0f / std::sqrt(static_cast<float>(d_k)));

    // causal mask: a query attends only to keys at or before its own position
    assert(scores.is_contiguous());
    int batches = static_cast<int>(scores.numel()) / (seq_q * seq_k);

    for (int b = 0; b < batches; b++) {
        float* mat = scores.data + static_cast<size_t>(b) * seq_q * seq_k;
        for (int i = 0; i < seq_q; i++) {
            for (int j = i + 1; j < seq_k; j++) {
                mat[i * seq_k + j] = -INFINITY;
            }
        }
    }

    Tensor weights = scores.softmax(scores.ndim - 1);

    return weights.matmul(value);
}

Tensor multi_head_attention(const Tensor& query, const Tensor& key, const Tensor& value, int num_heads) {
    assert(query.ndim == 3);
    assert(key.ndim == 3 && value.ndim == 3);

    int batch = query.shape[0];
    int seq = query.shape[1];
    int hidden = query.shape[2];

    assert(key.shape[0] == batch && value.shape[0] == batch);
    assert(key.shape[1] == seq && value.shape[1] == seq);
    assert(key.shape[2] == hidden && value.shape[2] == hidden);
    assert(num_heads > 0);
    assert(hidden % num_heads == 0);

    int head_dim = hidden / num_heads;

    // [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]
    Tensor q = query.reshape({batch, seq, num_heads, head_dim}).transpose(1, 2);
    Tensor k = key.reshape({batch, seq, num_heads, head_dim}).transpose(1, 2);
    Tensor v = value.reshape({batch, seq, num_heads, head_dim}).transpose(1, 2);

    Tensor heads_out = scaled_dot_product_attention(q, k, v);

    // [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
    return heads_out.transpose(1, 2).reshape({batch, seq, hidden});
}

Tensor self_attention(const Tensor& x, const Tensor& w_q, const Tensor& w_k,
                      const Tensor& w_v, const Tensor& w_o, int num_heads) {
    assert(x.ndim == 3);

    int batch = x.shape[0];
    int seq = x.shape[1];
    int hidden = x.shape[2];

    assert(w_q.ndim == 2 && w_q.shape[0] == hidden && w_q.shape[1] == hidden);
    assert(w_k.ndim == 2 && w_k.shape[0] == hidden && w_k.shape[1] == hidden);
    assert(w_v.ndim == 2 && w_v.shape[0] == hidden && w_v.shape[1] == hidden);
    assert(w_o.ndim == 2 && w_o.shape[0] == hidden && w_o.shape[1] == hidden);

    // projections as 2d matmuls over the flattened [batch * seq, hidden] rows
    Tensor flat = x.reshape({batch * seq, hidden});
    Tensor q = flat.matmul(w_q).reshape({batch, seq, hidden});
    Tensor k = flat.matmul(w_k).reshape({batch, seq, hidden});
    Tensor v = flat.matmul(w_v).reshape({batch, seq, hidden});

    Tensor heads_out = multi_head_attention(q, k, v, num_heads);

    return heads_out.reshape({batch * seq, hidden}).matmul(w_o).reshape({batch, seq, hidden});
}

Tensor attention_block(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                       const Tensor& w_q, const Tensor& w_k, const Tensor& w_v,
                       const Tensor& w_o, int num_heads) {
    assert(x.ndim == 3);

    // pre layernorm: normalize the input, attend, add back the unnormalized residual
    Tensor normed = x.layernorm(gamma, beta);
    Tensor attn = self_attention(normed, w_q, w_k, w_v, w_o, num_heads);

    return x.add(attn);
}

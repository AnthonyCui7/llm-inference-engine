#include <cmath>

#include "attention.hpp"

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                                    int q_offset) {
    assert(query.ndim >= 2);
    assert(key.ndim == query.ndim && value.ndim == query.ndim);

    int d_k = query.shape[query.ndim - 1];
    int seq_q = query.shape[query.ndim - 2];
    int seq_k = key.shape[key.ndim - 2];

    assert(key.shape[key.ndim - 1] == d_k);
    assert(value.shape[value.ndim - 2] == seq_k);

    // query row i sits at absolute position q_offset + i, and the keys
    // must cover exactly the history up to the last query
    assert(q_offset >= 0);
    assert(seq_k == q_offset + seq_q);

    Tensor key_t = key.transpose(key.ndim - 2, key.ndim - 1);
    Tensor scores = query.matmul(key_t).mul(1.0f / std::sqrt(static_cast<float>(d_k)));

    // causal mask: a query attends only to keys at or before its own position
    assert(scores.is_contiguous());
    int batches = static_cast<int>(scores.numel()) / (seq_q * seq_k);

    for (int b = 0; b < batches; b++) {
        float* mat = scores.data + static_cast<size_t>(b) * seq_q * seq_k;
        for (int i = 0; i < seq_q; i++) {
            for (int j = q_offset + i + 1; j < seq_k; j++) {
                mat[i * seq_k + j] = -INFINITY;
            }
        }
    }

    Tensor weights = scores.softmax(scores.ndim - 1);

    return weights.matmul(value);
}

Tensor scaled_dot_product_attention(const Tensor& query, const Tensor& key, const Tensor& value) {
    return scaled_dot_product_attention(query, key, value, 0);
}

Tensor multi_head_attention(const Tensor& query, const Tensor& key, const Tensor& value,
                            int num_heads, int q_offset) {
    assert(query.ndim == 3);
    assert(key.ndim == 3 && value.ndim == 3);

    int batch = query.shape[0];
    int seq_q = query.shape[1];
    int seq_k = key.shape[1];
    int hidden = query.shape[2];

    assert(key.shape[0] == batch && value.shape[0] == batch);
    assert(value.shape[1] == seq_k);
    assert(key.shape[2] == hidden && value.shape[2] == hidden);
    assert(num_heads > 0);
    assert(hidden % num_heads == 0);

    int head_dim = hidden / num_heads;

    // [batch, seq, hidden] -> [batch, num_heads, seq, head_dim]
    Tensor q = query.reshape({batch, seq_q, num_heads, head_dim}).transpose(1, 2);
    Tensor k = key.reshape({batch, seq_k, num_heads, head_dim}).transpose(1, 2);
    Tensor v = value.reshape({batch, seq_k, num_heads, head_dim}).transpose(1, 2);

    Tensor heads_out = scaled_dot_product_attention(q, k, v, q_offset);

    // [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
    return heads_out.transpose(1, 2).reshape({batch, seq_q, hidden});
}

Tensor multi_head_attention(const Tensor& query, const Tensor& key, const Tensor& value, int num_heads) {
    return multi_head_attention(query, key, value, num_heads, 0);
}

Tensor self_attention(const Tensor& x, const Tensor& w_q, const Tensor& b_q,
                      const Tensor& w_k, const Tensor& b_k,
                      const Tensor& w_v, const Tensor& b_v,
                      const Tensor& w_o, const Tensor& b_o, int num_heads) {
    assert(x.ndim == 3);

    int batch = x.shape[0];
    int seq = x.shape[1];
    int hidden = x.shape[2];

    assert(w_q.ndim == 2 && w_q.shape[0] == hidden && w_q.shape[1] == hidden);
    assert(w_k.ndim == 2 && w_k.shape[0] == hidden && w_k.shape[1] == hidden);
    assert(w_v.ndim == 2 && w_v.shape[0] == hidden && w_v.shape[1] == hidden);
    assert(w_o.ndim == 2 && w_o.shape[0] == hidden && w_o.shape[1] == hidden);
    assert(b_q.ndim == 1 && b_q.shape[0] == hidden);
    assert(b_k.ndim == 1 && b_k.shape[0] == hidden);
    assert(b_v.ndim == 1 && b_v.shape[0] == hidden);
    assert(b_o.ndim == 1 && b_o.shape[0] == hidden);

    // projections as 2d matmuls over the flattened [batch * seq, hidden] rows,
    // with each bias broadcast across the rows
    Tensor flat = x.reshape({batch * seq, hidden});
    Tensor q = flat.matmul(w_q).add(b_q.reshape({1, hidden})).reshape({batch, seq, hidden});
    Tensor k = flat.matmul(w_k).add(b_k.reshape({1, hidden})).reshape({batch, seq, hidden});
    Tensor v = flat.matmul(w_v).add(b_v.reshape({1, hidden})).reshape({batch, seq, hidden});

    Tensor heads_out = multi_head_attention(q, k, v, num_heads);

    return heads_out.reshape({batch * seq, hidden}).matmul(w_o)
        .add(b_o.reshape({1, hidden})).reshape({batch, seq, hidden});
}

Tensor self_attention_cached(const Tensor& x, const Tensor& w_q, const Tensor& b_q,
                             const Tensor& w_k, const Tensor& b_k,
                             const Tensor& w_v, const Tensor& b_v,
                             const Tensor& w_o, const Tensor& b_o, int num_heads,
                             KVCache& cache, int layer) {
    assert(x.ndim == 3);
    assert(x.shape[0] == 1);  // the cache holds a single sequence

    int seq = x.shape[1];
    int hidden = x.shape[2];
    assert(hidden == cache.hidden);

    Tensor flat = x.reshape({seq, hidden});
    Tensor q = flat.matmul(w_q).add(b_q.reshape({1, hidden}));
    Tensor k_new = flat.matmul(w_k).add(b_k.reshape({1, hidden}));
    Tensor v_new = flat.matmul(w_v).add(b_v.reshape({1, hidden}));

    cache.append(layer, k_new, v_new);

    // attend over everything cached so far, new rows included
    int total = cache.layers[layer].cursor;
    Tensor k_all = cache.filled_k(layer).reshape({1, total, hidden});
    Tensor v_all = cache.filled_v(layer).reshape({1, total, hidden});

    Tensor heads_out = multi_head_attention(q.reshape({1, seq, hidden}), k_all, v_all,
                                            num_heads, total - seq);

    return heads_out.reshape({seq, hidden}).matmul(w_o)
        .add(b_o.reshape({1, hidden})).reshape({1, seq, hidden});
}

Tensor attention_block(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                       const Tensor& w_q, const Tensor& b_q,
                       const Tensor& w_k, const Tensor& b_k,
                       const Tensor& w_v, const Tensor& b_v,
                       const Tensor& w_o, const Tensor& b_o, int num_heads) {
    assert(x.ndim == 3);

    // pre layernorm: normalize the input, attend, add back the unnormalized residual
    Tensor normed = x.layernorm(gamma, beta);
    Tensor attn = self_attention(normed, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o, num_heads);

    return x.add(attn);
}

Tensor attention_block_cached(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                              const Tensor& w_q, const Tensor& b_q,
                              const Tensor& w_k, const Tensor& b_k,
                              const Tensor& w_v, const Tensor& b_v,
                              const Tensor& w_o, const Tensor& b_o, int num_heads,
                              KVCache& cache, int layer) {
    assert(x.ndim == 3);

    Tensor normed = x.layernorm(gamma, beta);
    Tensor attn = self_attention_cached(normed, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
                                        num_heads, cache, layer);

    return x.add(attn);
}

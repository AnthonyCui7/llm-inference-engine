#include <cstring>

#include "kv_cache.hpp"

KVCache::KVCache(int num_layers, int max_seq, int hidden)
    : max_seq(max_seq), hidden(hidden) {
    assert(num_layers > 0 && max_seq > 0 && hidden > 0);

    layers.reserve(num_layers);
    for (int i = 0; i < num_layers; i++) {
        KVLayer layer;
        layer.k = Tensor({max_seq, hidden});
        layer.v = Tensor({max_seq, hidden});
        layer.cursor = 0;
        layers.push_back(std::move(layer));
    }
}

void KVCache::append(int layer, const Tensor& k_new, const Tensor& v_new) {
    assert(layer >= 0 && layer < static_cast<int>(layers.size()));
    assert(k_new.ndim == 2 && k_new.shape[1] == hidden);
    assert(v_new.ndim == 2 && v_new.shape[0] == k_new.shape[0] && v_new.shape[1] == hidden);
    assert(k_new.is_contiguous() && v_new.is_contiguous());

    KVLayer& l = layers[layer];
    int rows = k_new.shape[0];
    assert(l.cursor + rows <= max_seq);

    std::memcpy(l.k.data + static_cast<size_t>(l.cursor) * hidden,
                k_new.data, static_cast<size_t>(rows) * hidden * sizeof(float));
    std::memcpy(l.v.data + static_cast<size_t>(l.cursor) * hidden,
                v_new.data, static_cast<size_t>(rows) * hidden * sizeof(float));

    l.cursor += rows;
}

Tensor KVCache::filled_k(int layer) const {
    assert(layer >= 0 && layer < static_cast<int>(layers.size()));
    const KVLayer& l = layers[layer];
    assert(l.cursor > 0);

    Tensor out({l.cursor, hidden});
    std::memcpy(out.data, l.k.data, static_cast<size_t>(l.cursor) * hidden * sizeof(float));
    return out;
}

Tensor KVCache::filled_v(int layer) const {
    assert(layer >= 0 && layer < static_cast<int>(layers.size()));
    const KVLayer& l = layers[layer];
    assert(l.cursor > 0);

    Tensor out({l.cursor, hidden});
    std::memcpy(out.data, l.v.data, static_cast<size_t>(l.cursor) * hidden * sizeof(float));
    return out;
}

void KVCache::truncate(int new_len) {
    assert(new_len >= 0);

    for (KVLayer& l : layers) {
        assert(new_len <= l.cursor);
        l.cursor = new_len;
    }
}

int KVCache::seq_len() const {
    for (const KVLayer& l : layers) {
        assert(l.cursor == layers.front().cursor);
        (void)l;
    }
    return layers.front().cursor;
}

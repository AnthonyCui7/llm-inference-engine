/*
Per layer key/value cache for autoregressive decoding.

The buffers break with the usual exact-size-per-construction tensor story:
each layer's k/v is preallocated at max_seq capacity up front and filled
left to right, with a cursor tracking how many rows are valid. append
writes new rows in place at the cursor instead of reallocating, and
attention gets a copy of the filled prefix. The layer cursors stay in sync
because every forward step appends to every layer.
*/
#pragma once

#include <vector>

#include "tensor.hpp"

struct KVLayer {
    Tensor k;  // [max_seq, hidden]
    Tensor v;
    int cursor;
};

struct KVCache {
    KVCache(int num_layers, int max_seq, int hidden);

    // copies [rows, hidden] of new keys/values into the layer at its
    // cursor and moves the cursor past them
    void append(int layer, const Tensor& k_new, const Tensor& v_new);

    // copies of the filled prefix, [cursor, hidden]
    Tensor filled_k(int layer) const;
    Tensor filled_v(int layer) const;

    // rolls every layer back to the first new_len positions, for dropping
    // rejected draft tokens; only the cursors move, later rows just get
    // overwritten by the next append
    void truncate(int new_len);

    // positions filled so far
    int seq_len() const;

    int max_seq;
    int hidden;
    std::vector<KVLayer> layers;
};

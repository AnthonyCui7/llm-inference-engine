#include "model.hpp"

Tensor embed(const Tensor& wte, const Tensor& wpe, const Tensor& token_ids) {
    assert(wte.ndim == 2 && wpe.ndim == 2);
    assert(wte.shape[1] == wpe.shape[1]);
    assert(token_ids.ndim == 2);

    int seq = token_ids.shape[1];
    assert(seq <= wpe.shape[0]);

    // positions are just 0..seq-1, looked up like token ids
    Tensor positions({1, seq});
    for (int i = 0; i < seq; i++) {
        positions.data[i] = static_cast<float>(i);
    }

    // [batch, seq, hidden] + [1, seq, hidden] broadcasts over the batch
    return wte.embedding(token_ids).add(wpe.embedding(positions));
}

Tensor model_forward(const ModelWeights& model, const Tensor& token_ids) {
    assert(token_ids.ndim == 2);

    int batch = token_ids.shape[0];
    int seq = token_ids.shape[1];
    assert(seq <= model.config.max_seq);

    Tensor x = embed(model.wte, model.wpe, token_ids);
    x = transformer_stack(x, model.blocks, model.config.num_heads);
    x = x.layernorm(model.final_gamma, model.final_beta);

    return x.reshape({batch * seq, model.config.hidden}).matmul(model.lm_head)
        .reshape({batch, seq, model.config.vocab_size});
}

#include "model.hpp"

Tensor embed(const Tensor& wte, const Tensor& wpe, const Tensor& token_ids, int pos_offset) {
    assert(wte.ndim == 2 && wpe.ndim == 2);
    assert(wte.shape[1] == wpe.shape[1]);
    assert(token_ids.ndim == 2);
    assert(pos_offset >= 0);

    int seq = token_ids.shape[1];
    assert(pos_offset + seq <= wpe.shape[0]);

    // positions continue from pos_offset, looked up like token ids
    Tensor positions({1, seq});
    for (int i = 0; i < seq; i++) {
        positions.data[i] = static_cast<float>(pos_offset + i);
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

Tensor model_forward_cached(const ModelWeights& model, const Tensor& token_ids, KVCache& cache) {
    assert(token_ids.ndim == 2);
    assert(token_ids.shape[0] == 1);

    int seq = token_ids.shape[1];
    int offset = cache.seq_len();
    assert(offset + seq <= model.config.max_seq);

    Tensor x = embed(model.wte, model.wpe, token_ids, offset);
    x = transformer_stack_cached(x, model.blocks, model.config.num_heads, cache);
    x = x.layernorm(model.final_gamma, model.final_beta);

    return x.reshape({seq, model.config.hidden}).matmul(model.lm_head)
        .reshape({1, seq, model.config.vocab_size});
}

ModelWeights load_model(const WeightLoader& loader) {
    ModelWeights model;

    Tensor config = loader.load("config");
    assert(config.ndim == 1 && config.shape[0] == 5);
    model.config.vocab_size = static_cast<int>(config.data[0]);
    model.config.max_seq = static_cast<int>(config.data[1]);
    model.config.hidden = static_cast<int>(config.data[2]);
    model.config.num_heads = static_cast<int>(config.data[3]);
    model.config.num_layers = static_cast<int>(config.data[4]);

    model.wte = loader.load("wte");
    model.wpe = loader.load("wpe");
    assert(model.wte.ndim == 2 && model.wte.shape[0] == model.config.vocab_size
           && model.wte.shape[1] == model.config.hidden);
    assert(model.wpe.ndim == 2 && model.wpe.shape[0] == model.config.max_seq
           && model.wpe.shape[1] == model.config.hidden);

    model.blocks.reserve(model.config.num_layers);
    for (int i = 0; i < model.config.num_layers; i++) {
        std::string prefix = "h" + std::to_string(i) + ".";

        TransformerBlockWeights block;
        block.attn_gamma = loader.load(prefix + "attn_gamma");
        block.attn_beta = loader.load(prefix + "attn_beta");
        block.w_q = loader.load(prefix + "w_q");
        block.b_q = loader.load(prefix + "b_q");
        block.w_k = loader.load(prefix + "w_k");
        block.b_k = loader.load(prefix + "b_k");
        block.w_v = loader.load(prefix + "w_v");
        block.b_v = loader.load(prefix + "b_v");
        block.w_o = loader.load(prefix + "w_o");
        block.b_o = loader.load(prefix + "b_o");
        block.mlp_gamma = loader.load(prefix + "mlp_gamma");
        block.mlp_beta = loader.load(prefix + "mlp_beta");
        block.w_fc = loader.load(prefix + "w_fc");
        block.b_fc = loader.load(prefix + "b_fc");
        block.w_proj = loader.load(prefix + "w_proj");
        block.b_proj = loader.load(prefix + "b_proj");

        model.blocks.push_back(std::move(block));
    }

    model.final_gamma = loader.load("final_gamma");
    model.final_beta = loader.load("final_beta");

    // the unembedding is tied to the token embedding
    model.lm_head = model.wte.transpose(0, 1);

    return model;
}

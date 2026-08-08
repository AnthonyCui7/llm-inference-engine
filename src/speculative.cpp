#include "speculative.hpp"

static Tensor one_token(int token) {
    Tensor ids({1, 1});
    ids.data[0] = static_cast<float>(token);
    return ids;
}

// Each round both caches hold everything before the pending token (the
// newest one not yet fed to either model). The draft feeds pending and
// then its own samples one at a time; the target scores pending plus all
// draft_len proposals in a single pass. Proposal x_i is accepted with
// probability min(1, p_target(x_i) / p_draft(x_i)); on the first reject a
// replacement is drawn from the normalized residual max(0, p_t - p_d) and
// both caches are truncated back past the dead proposals. If everything
// is accepted, the extra target row gives one bonus token for free.
std::vector<int> generate_speculative(const ModelWeights& target, const ModelWeights& draft,
                                      const std::vector<int>& prompt, int max_new_tokens,
                                      int draft_len, float temperature, std::mt19937& rng) {
    assert(!prompt.empty());
    assert(draft_len > 0);
    assert(target.config.vocab_size == draft.config.vocab_size);

    // a round can run up to draft_len tokens past the budget before trimming
    int worst_case = static_cast<int>(prompt.size()) + max_new_tokens + draft_len;
    assert(worst_case <= target.config.max_seq && worst_case <= draft.config.max_seq);
    (void)worst_case;

    std::vector<int> tokens = prompt;
    if (max_new_tokens == 0) return tokens;

    int vocab = target.config.vocab_size;

    KVCache target_cache(target.config.num_layers, target.config.max_seq, target.config.hidden);
    KVCache draft_cache(draft.config.num_layers, draft.config.max_seq, draft.config.hidden);

    int pending = prompt.back();
    if (prompt.size() > 1) {
        Tensor context_ids({1, static_cast<int>(prompt.size()) - 1});
        for (size_t i = 0; i + 1 < prompt.size(); i++) {
            context_ids.data[i] = static_cast<float>(prompt[i]);
        }
        model_forward_cached(target, context_ids, target_cache);
        model_forward_cached(draft, context_ids, draft_cache);
    }

    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    int generated = 0;

    while (generated < max_new_tokens) {
        int base = target_cache.seq_len();
        assert(draft_cache.seq_len() == base);

        // draft proposes draft_len tokens one at a time
        std::vector<int> proposed;
        std::vector<std::vector<float>> draft_probs;

        int feed = pending;
        for (int i = 0; i < draft_len; i++) {
            Tensor draft_logits = model_forward_cached(draft, one_token(feed), draft_cache);
            draft_probs.push_back(logits_to_probs(draft_logits.data, vocab, temperature));
            proposed.push_back(sample_from(draft_probs.back(), rng));
            feed = proposed.back();
        }

        // keep the draft cache in step with its last proposal
        model_forward_cached(draft, one_token(proposed.back()), draft_cache);

        // target scores pending plus every proposal in one batched pass
        Tensor batch_ids({1, draft_len + 1});
        batch_ids.data[0] = static_cast<float>(pending);
        for (int i = 0; i < draft_len; i++) {
            batch_ids.data[i + 1] = static_cast<float>(proposed[i]);
        }
        Tensor target_logits = model_forward_cached(target, batch_ids, target_cache);

        int accepted = 0;
        bool rejected = false;

        for (int i = 0; i < draft_len && generated < max_new_tokens; i++) {
            std::vector<float> target_probs =
                logits_to_probs(target_logits.data + static_cast<size_t>(i) * vocab,
                                vocab, temperature);
            const std::vector<float>& dp = draft_probs[i];
            int x = proposed[i];

            if (uniform(rng) < target_probs[x] / dp[x]) {
                tokens.push_back(x);
                generated++;
                accepted++;
                continue;
            }

            // rejected: replacement comes from where the target puts more
            // mass than the draft, which keeps the total output exactly
            // target distributed
            std::vector<float> residual(vocab);
            double norm = 0.0;
            for (int t = 0; t < vocab; t++) {
                float diff = target_probs[t] - dp[t];
                residual[t] = diff > 0.0f ? diff : 0.0f;
                norm += residual[t];
            }

            int replacement;
            if (norm > 0.0) {
                for (int t = 0; t < vocab; t++) {
                    residual[t] = static_cast<float>(residual[t] / norm);
                }
                replacement = sample_from(residual, rng);
            } else {
                // distributions were identical up to rounding
                replacement = sample_from(target_probs, rng);
            }

            // drop the dead proposals from both caches
            target_cache.truncate(base + 1 + accepted);
            draft_cache.truncate(base + 1 + accepted);

            pending = replacement;
            tokens.push_back(replacement);
            generated++;
            rejected = true;
            break;
        }

        if (!rejected && generated < max_new_tokens) {
            // every proposal survived, so the last target row is a free token
            std::vector<float> bonus =
                logits_to_probs(target_logits.data + static_cast<size_t>(draft_len) * vocab,
                                vocab, temperature);
            pending = sample_from(bonus, rng);
            tokens.push_back(pending);
            generated++;
        }
    }

    return tokens;
}

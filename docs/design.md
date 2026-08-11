# Design notes

- Positional embeddings are GPT-2's learned kind, so they are just one
  more embedding lookup added to the token embeddings.
- Tokenization stays in python; a C++ BPE tokenizer was not worth it for
  getting a real model running.
- Weights are fp32 and load through the `WeightLoader` interface in
  `src/weights.hpp`. Quantized formats would be added as another loader
  behind the same interface.
- CPU only for now; a GPU backend would be a separate compute path rather
  than a change to the tensor layout.
- The kv cache (`src/kv_cache.hpp`) is the one place that breaks the
  exact-size-per-construction tensor story: per layer k/v buffers are
  preallocated at max_seq capacity and a cursor tracks how much is
  filled, so decoding appends rows in place instead of reallocating.
- Speculative decoding is distribution exact (rejection sampling against
  the target's probabilities). Decode is memory bound, so it only pays
  off when the draft streams far fewer bytes per token than the target:
  gpt2 small with a distilgpt2 draft loses outright (the draft costs
  0.6x the target, capping the speedup near 1.1x even at perfect
  acceptance), while gpt2-large with the same draft and 3 proposals per
  round runs about 1.4x faster (129 -> 92 ms/token at temperature 1).
  The dump script takes any gpt2 family name, so the pairing is just a
  choice of weight files.
- The reference logits script runs the huggingface model in float64:
  torch's fp32 cpu blas on apple silicon returns NaNs for the larger
  gpt2 head projections even with finite inputs.

# llm-inference-engine

From scratch C++17 tensor library and transformer inference engine, no
dependencies beyond the STL. A custom `Tensor` struct with manual strides,
a persistent thread pool and SIMD matmul kernels underneath, and a GPT-2
forward pass on top that runs the real pretrained weights.

## Usage

Dump GPT-2 small into the engine's weight format, then generate:

```
python3 scripts/dump_gpt2.py gpt2 gpt2.bin
make generate
./bin/generate gpt2.bin 10 464 3139 286 4881 318
```

The driver works on raw token ids; `scripts/tokenizer.py encode`/`decode`
converts to and from text.

With a second weight file the driver decodes speculatively, sampling at
temperature 1 with a fixed seed:

```
python3 scripts/dump_gpt2.py distilgpt2 distilgpt2.bin
./bin/generate gpt2.bin --draft distilgpt2.bin 10 464 3139 286 4881 318
```

Check the forward pass against huggingface logits on a fixed prompt:

```
python3 scripts/reference_logits.py
make validate
```

`make run-test` runs the unit tests, `make run-bench` the matmul benchmark
(numbers in `benchmarks/matmul.md`).

## Notes

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
  the target's probabilities) but does not pay off with gpt2 small as
  target and distilgpt2 as draft: decode is memory bound and the draft
  streams about 0.6x the target's weights per token, since the 154MB
  unembedding is the same for both. Even perfect acceptance would cap
  the speedup near 1.1x, so this pairing stays a correctness
  demonstration; winning needs a much smaller draft.

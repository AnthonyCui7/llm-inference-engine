# llm-inference-engine

From scratch C++17 tensor library and transformer inference engine, no
dependencies beyond the STL. A custom `Tensor` struct with manual strides,
a persistent thread pool and SIMD matmul kernels underneath, and a GPT-2
forward pass on top that runs the real pretrained weights.

## Features

- Up to 8 dimensional tensors with broadcasting, batched matmul, gelu,
  layernorm, softmax, and embedding lookups
- Threaded, SIMD (NEON/AVX2), cache blocked matmul, bitwise identical to
  the naive reference kernel
- Full GPT-2 forward pass: multi head causal attention, pre layernorm
  blocks, learned positional embeddings, tied unembedding
- Binary weight format plus a loader that runs any GPT-2 family
  checkpoint (gpt2, distilgpt2, gpt2-medium, gpt2-large)
- KV cached decoding with rollback, greedy and temperature sampling, and
  distribution exact speculative decoding
- Forward pass validated against huggingface logits

## Architecture

```
src/tensor.hpp       tensor struct, strides, kernels
src/thread_pool.hpp  persistent worker pool behind matmul
src/attention.hpp    causal attention, with and without the kv cache
src/transformer.hpp  block weights, mlp block, block stack
src/model.hpp        embeddings, forward pass, weight loading
src/kv_cache.hpp     preallocated per layer k/v buffers with a cursor
src/sample.hpp       temperature sampling
src/speculative.hpp  draft/verify decoding with rejection sampling
src/weights.hpp      weight file format and loader interface
```

Design decisions are written up in `docs/design.md`.

## Performance

- Matmul: about 7x over the naive kernel (roughly 30 to 200-215 gflop/s
  on an M3 Pro for N >= 512), from threading, an SIMD row kernel, and
  k blocking; full history in `benchmarks/matmul.md`
- Generation: the kv cache takes gpt2 small from 75.8 to 16.5 ms/token,
  and speculative decoding runs gpt2-large about 1.4x faster with a
  distilgpt2 draft

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

`make run-test` runs the unit tests, `make run-bench` the matmul
benchmark, and `make run-bench-generate` the end to end generation
benchmark.

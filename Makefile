CC := clang++
CFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Isrc
THREAD_FLAGS := -pthread

DEBUG_FLAGS := $(CFLAGS) -O0 -g
RELEASE_FLAGS := $(CFLAGS) -O3 -march=native -DNDEBUG
FAST_MATH_FLAGS := $(RELEASE_FLAGS) -ffast-math

# everything except the driver's main
CORE_SRC := $(filter-out src/main.cpp, $(wildcard src/*.cpp))

.PHONY: test run-test bench run-bench bench-fast-math generate clean

test:
	$(CC) $(DEBUG_FLAGS) $(THREAD_FLAGS) $(CORE_SRC) tests/test_tensor.cpp -o bin/test_tensor

run-test: test
	./bin/test_tensor

bench:
	$(CC) $(RELEASE_FLAGS) $(THREAD_FLAGS) $(CORE_SRC) benchmarks/bench_matmul.cpp -o bin/bench_matmul

run-bench: bench
	./bin/bench_matmul

bench-fast-math:
	$(CC) $(FAST_MATH_FLAGS) $(THREAD_FLAGS) $(CORE_SRC) benchmarks/bench_matmul.cpp -o bin/bench_matmul_fast_math

generate:
	$(CC) $(RELEASE_FLAGS) $(THREAD_FLAGS) $(CORE_SRC) src/main.cpp -o bin/generate

clean:
	rm -rf bin/*

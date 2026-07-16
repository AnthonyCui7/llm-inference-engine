CC := clang++
CFLAGS := -std=c++17 -Wall -Wextra -Wpedantic -Isrc
THREAD_FLAGS := -pthread

DEBUG_FLAGS := $(CFLAGS) -O0 -g
RELEASE_FLAGS := $(CFLAGS) -O3 -march=native -DNDEBUG
FAST_MATH_FLAGS := $(RELEASE_FLAGS) -ffast-math

.PHONY: test run-test bench run-bench bench-fast-math bench-lto clean

test:
	$(CC) $(DEBUG_FLAGS) $(THREAD_FLAGS) src/tensor.cpp src/thread_pool.cpp src/attention.cpp tests/test_tensor.cpp -o bin/test_tensor

run-test: test
	./bin/test_tensor

bench:
	$(CC) $(RELEASE_FLAGS) $(THREAD_FLAGS) src/tensor.cpp src/thread_pool.cpp src/attention.cpp benchmarks/bench_matmul.cpp -o bin/bench_matmul

run-bench: bench
	./bin/bench_matmul

bench-fast-math:
	$(CC) $(FAST_MATH_FLAGS) $(THREAD_FLAGS) src/tensor.cpp src/thread_pool.cpp src/attention.cpp benchmarks/bench_matmul.cpp -o bin/bench_matmul_fast_math

clean:
	rm -rf bin/*
# Matmul Benchmarks

## Benchmark Method

Compiler: `clang++`
Benchmark flags: `-std=c++17 -Wall -Wextra -Wpedantic -Isrc -O3 -march=native -DNDEBUG`
Machine: Apple M3 Pro (5P + 6E cores, 11 hardware threads)

For benchmark code, see `bench_matmul.cpp`.

- Matrices are square: N x N
- Input values are deterministic
- Warmup iterations are excluded from timing
- FLOPs estimate: `2 * N^3`
- Output correctness sanity check: checksum
- Kernel is selectable: `bench_matmul [matmul|naive]`
- For N <= 1024, output is also checked elementwise against `matmul_naive` (`max_diff_vs_naive`)

Iterations:
```text
N = 64: 512 timed iterations, 128 warmup iterations
N = 128: 256 timed iterations, 64 warmup iterations
N = 256: 64 timed iterations, 16 warmup iterations
N = 512: 16 timed iterations, 4 warmup iterations
N = 1024: 8 timed iterations, 2 warmup iterations
N = 2048: 4 timed iterations, 1 warmup iteration
N = 4096: 3 timed iterations, 1 warmup iteration
```

## Naive contiguous 2D matmul, ijk loop order

```text
kernel = naive, N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.165408, gflop/s = 3.16966, checksum = 137.625, warmup_checksum = 34.4064, max_diff_vs_naive = 0
kernel = naive, N = 128, iters = 256, warmup_iters = 64, avg_ms = 1.18985, gflop/s = 3.52507, checksum = 152.013, warmup_checksum = 38.0032, max_diff_vs_naive = 0
kernel = naive, N = 256, iters = 64, warmup_iters = 16, avg_ms = 11.8869, gflop/s = 2.8228, checksum = 77.376, warmup_checksum = 19.344, max_diff_vs_naive = 0
kernel = naive, N = 512, iters = 16, warmup_iters = 4, avg_ms = 111.663, gflop/s = 2.40397, checksum = 39.0672, warmup_checksum = 9.7668, max_diff_vs_naive = 0
kernel = naive, N = 1024, iters = 8, warmup_iters = 2, avg_ms = 1001.95, gflop/s = 2.14329, checksum = 39.296, warmup_checksum = 9.82399, max_diff_vs_naive = 0
kernel = naive, N = 2048, iters = 4, warmup_iters = 1, avg_ms = 8722.37, gflop/s = 1.96963, checksum = 39.2923, warmup_checksum = 9.82308
kernel = naive, N = 4096, iters = 3, warmup_iters = 1, avg_ms = 214595, gflop/s = 0.640457, checksum = 58.9486, warmup_checksum = 19.6495
```

## Baseline contiguous 2D matmul, ikj loop order

```text
N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0337554, gflop/s = 15.532, checksum = 137.625, warmup_checksum = 34.4064
N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.129176, gflop/s = 32.4696, checksum = 152.013, warmup_checksum = 38.0032
N = 256, iters = 64, warmup_iters = 16, avg_ms = 1.1465, gflop/s = 29.2669, checksum = 77.376, warmup_checksum = 19.344
N = 512, iters = 16, warmup_iters = 4, avg_ms = 8.9965, gflop/s = 29.8378, checksum = 39.0672, warmup_checksum = 9.7668
N = 1024, iters = 8, warmup_iters = 2, avg_ms = 71.8334, gflop/s = 29.8953, checksum = 39.296, warmup_checksum = 9.82399
N = 2048, iters = 4, warmup_iters = 1, avg_ms = 577.636, gflop/s = 29.7417, checksum = 39.2923, warmup_checksum = 9.82308
N = 4096, iters = 3, warmup_iters = 1, avg_ms = 4718.69, gflop/s = 29.1265, checksum = 58.9486, warmup_checksum = 19.6495
```

## Blocked contiguous 2D matmul, block_size = 32

```text
N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0569882, gflop/s = 9.19994, checksum = 137.625, warmup_checksum = 34.4064
N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.234237, gflop/s = 17.9062, checksum = 152.013, warmup_checksum = 38.0032
N = 256, iters = 64, warmup_iters = 16, avg_ms = 1.78585, gflop/s = 18.7891, checksum = 77.376, warmup_checksum = 19.344
N = 512, iters = 16, warmup_iters = 4, avg_ms = 14.3432, gflop/s = 18.7152, checksum = 39.0672, warmup_checksum = 9.7668
N = 1024, iters = 8, warmup_iters = 2, avg_ms = 126.35, gflop/s = 16.9963, checksum = 39.296, warmup_checksum = 9.82399
N = 2048, iters = 4, warmup_iters = 1, avg_ms = 1257.09, gflop/s = 13.6664, checksum = 39.2923, warmup_checksum = 9.82308
N = 4096, iters = 3, warmup_iters = 1, avg_ms = 10599.5, gflop/s = 12.9666, checksum = 58.9486, warmup_checksum = 19.6495
```

## Blocked contiguous 2D matmul, block_size = 64

```text
N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0437582, gflop/s = 11.9815, checksum = 137.625, warmup_checksum = 34.4064
N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.173213, gflop/s = 24.2147, checksum = 152.013, warmup_checksum = 38.0032
N = 256, iters = 64, warmup_iters = 16, avg_ms = 1.19989, gflop/s = 27.9647, checksum = 77.376, warmup_checksum = 19.344
N = 512, iters = 16, warmup_iters = 4, avg_ms = 10.4394, gflop/s = 25.7137, checksum = 39.0672, warmup_checksum = 9.7668
N = 1024, iters = 8, warmup_iters = 2, avg_ms = 100.233, gflop/s = 21.425, checksum = 39.296, warmup_checksum = 9.82399
N = 2048, iters = 4, warmup_iters = 1, avg_ms = 978.899, gflop/s = 17.5502, checksum = 39.2923, warmup_checksum = 9.82308
N = 4096, iters = 3, warmup_iters = 1, avg_ms = 7930.56, gflop/s = 17.3303, checksum = 58.9486, warmup_checksum = 19.6495
```

## Threaded contiguous 2D matmul, threads = 2
```text
N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0531331, gflop/s = 9.86745, checksum = 137.625, warmup_checksum = 34.4064
N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.0910218, gflop/s = 46.0802, checksum = 152.013, warmup_checksum = 38.0032
N = 256, iters = 64, warmup_iters = 16, avg_ms = 0.621308, gflop/s = 54.0061, checksum = 77.376, warmup_checksum = 19.344
N = 512, iters = 16, warmup_iters = 4, avg_ms = 4.79713, gflop/s = 55.9575, checksum = 39.0672, warmup_checksum = 9.7668
N = 1024, iters = 8, warmup_iters = 2, avg_ms = 44.8117, gflop/s = 47.9224, checksum = 39.296, warmup_checksum = 9.82399
N = 2048, iters = 4, warmup_iters = 1, avg_ms = 303.081, gflop/s = 56.6841, checksum = 39.2923, warmup_checksum = 9.82308
N = 4096, iters = 3, warmup_iters = 1, avg_ms = 2541.94, gflop/s = 54.0685, checksum = 58.9486, warmup_checksum = 19.6495
```

## Threaded contiguous 2D matmul, threads = 4
```text
N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0492963, gflop/s = 10.6354, checksum = 137.625, warmup_checksum = 34.4064
N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.0688587, gflop/s = 60.9117, checksum = 152.013, warmup_checksum = 38.0032
N = 256, iters = 64, warmup_iters = 16, avg_ms = 0.349254, gflop/s = 96.0746, checksum = 77.376, warmup_checksum = 19.344
N = 512, iters = 16, warmup_iters = 4, avg_ms = 2.50071, gflop/s = 107.344, checksum = 39.0672, warmup_checksum = 9.7668
N = 1024, iters = 8, warmup_iters = 2, avg_ms = 20.2142, gflop/s = 106.236, checksum = 39.296, warmup_checksum = 9.82399
N = 2048, iters = 4, warmup_iters = 1, avg_ms = 185.746, gflop/s = 92.491, checksum = 39.2923, warmup_checksum = 9.82308
N = 4096, iters = 3, warmup_iters = 1, avg_ms = 1694.61, gflop/s = 81.1037, checksum = 58.9486, warmup_checksum = 19.6495
```

## Threaded contiguous 2D matmul, threads = 8
```text
N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0812304, gflop/s = 6.45433, checksum = 137.625, warmup_checksum = 34.4064
N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.0911401, gflop/s = 46.0204, checksum = 152.013, warmup_checksum = 38.0032
N = 256, iters = 64, warmup_iters = 16, avg_ms = 0.327262, gflop/s = 102.531, checksum = 77.376, warmup_checksum = 19.344
N = 512, iters = 16, warmup_iters = 4, avg_ms = 1.9864, gflop/s = 135.137, checksum = 39.0672, warmup_checksum = 9.7668
N = 1024, iters = 8, warmup_iters = 2, avg_ms = 15.3544, gflop/s = 139.861, checksum = 39.296, warmup_checksum = 9.82399
N = 2048, iters = 4, warmup_iters = 1, avg_ms = 106.777, gflop/s = 160.895, checksum = 39.2923, warmup_checksum = 9.82308
N = 4096, iters = 3, warmup_iters = 1, avg_ms = 1279.4, gflop/s = 107.425, checksum = 58.9486, warmup_checksum = 19.6495
```

## Threaded contiguous 2D matmul, threads = 10
```text
N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0897239, gflop/s = 5.84335, checksum = 137.625, warmup_checksum = 34.4064
N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.0852586, gflop/s = 49.1951, checksum = 152.013, warmup_checksum = 38.0032
N = 256, iters = 64, warmup_iters = 16, avg_ms = 0.321444, gflop/s = 104.387, checksum = 77.376, warmup_checksum = 19.344
N = 512, iters = 16, warmup_iters = 4, avg_ms = 1.71247, gflop/s = 156.754, checksum = 39.0672, warmup_checksum = 9.7668
N = 1024, iters = 8, warmup_iters = 2, avg_ms = 13.172, gflop/s = 163.034, checksum = 39.296, warmup_checksum = 9.82399
N = 2048, iters = 4, warmup_iters = 1, avg_ms = 101.091, gflop/s = 169.944, checksum = 39.2923, warmup_checksum = 9.82308
N = 4096, iters = 3, warmup_iters = 1, avg_ms = 1361.75, gflop/s = 100.928, checksum = 58.9486, warmup_checksum = 19.6495
```


## Threaded contiguous 2D matmul, threads = 11 (max hardware threads)
```text
N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0835247, gflop/s = 6.27704, checksum = 137.625, warmup_checksum = 34.4064
N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.0886198, gflop/s = 47.3292, checksum = 152.013, warmup_checksum = 38.0032
N = 256, iters = 64, warmup_iters = 16, avg_ms = 0.335259, gflop/s = 100.085, checksum = 77.376, warmup_checksum = 19.344
N = 512, iters = 16, warmup_iters = 4, avg_ms = 1.7594, gflop/s = 152.572, checksum = 39.0672, warmup_checksum = 9.7668
N = 1024, iters = 8, warmup_iters = 2, avg_ms = 13.626, gflop/s = 157.602, checksum = 39.296, warmup_checksum = 9.82399
N = 2048, iters = 4, warmup_iters = 1, avg_ms = 100.561, gflop/s = 170.84, checksum = 39.2923, warmup_checksum = 9.82308
N = 4096, iters = 3, warmup_iters = 1, avg_ms = 1300.96, gflop/s = 105.644, checksum = 58.9486, warmup_checksum = 19.6495
```

## Batched matmul, thread spawn per call, threads = 11 (default)

```text
kernel = matmul, N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0930399, gflop/s = 5.63509, checksum = 137.625, warmup_checksum = 34.4064, max_diff_vs_naive = 0
kernel = matmul, N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.0862887, gflop/s = 48.6078, checksum = 152.013, warmup_checksum = 38.0032, max_diff_vs_naive = 0
kernel = matmul, N = 256, iters = 64, warmup_iters = 16, avg_ms = 0.330342, gflop/s = 101.575, checksum = 77.376, warmup_checksum = 19.344, max_diff_vs_naive = 0
kernel = matmul, N = 512, iters = 16, warmup_iters = 4, avg_ms = 1.79047, gflop/s = 149.925, checksum = 39.0672, warmup_checksum = 9.7668, max_diff_vs_naive = 0
kernel = matmul, N = 1024, iters = 8, warmup_iters = 2, avg_ms = 13.6878, gflop/s = 156.891, checksum = 39.296, warmup_checksum = 9.82399, max_diff_vs_naive = 0
kernel = matmul, N = 2048, iters = 4, warmup_iters = 1, avg_ms = 105.254, gflop/s = 163.223, checksum = 39.2923, warmup_checksum = 9.82308
kernel = matmul, N = 4096, iters = 3, warmup_iters = 1, avg_ms = 1278.37, gflop/s = 107.511, checksum = 58.9486, warmup_checksum = 19.6495
```

## Batched matmul, persistent thread pool, threads = 11 (default)

```text
kernel = matmul, N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0307115, gflop/s = 17.0714, checksum = 137.625, warmup_checksum = 34.4064, max_diff_vs_naive = 0
kernel = matmul, N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.0443994, gflop/s = 94.4676, checksum = 152.013, warmup_checksum = 38.0032, max_diff_vs_naive = 0
kernel = matmul, N = 256, iters = 64, warmup_iters = 16, avg_ms = 0.283855, gflop/s = 118.21, checksum = 77.376, warmup_checksum = 19.344, max_diff_vs_naive = 0
kernel = matmul, N = 512, iters = 16, warmup_iters = 4, avg_ms = 1.74678, gflop/s = 153.674, checksum = 39.0672, warmup_checksum = 9.7668, max_diff_vs_naive = 0
kernel = matmul, N = 1024, iters = 8, warmup_iters = 2, avg_ms = 13.75, gflop/s = 156.181, checksum = 39.296, warmup_checksum = 9.82399, max_diff_vs_naive = 0
kernel = matmul, N = 2048, iters = 4, warmup_iters = 1, avg_ms = 106.676, gflop/s = 161.047, checksum = 39.2923, warmup_checksum = 9.82308
kernel = matmul, N = 4096, iters = 3, warmup_iters = 1, avg_ms = 1383.69, gflop/s = 99.3276, checksum = 58.9486, warmup_checksum = 19.6495
```

## Batched matmul, persistent thread pool + SIMD row kernel, threads = 11 (default)

```text
kernel = matmul, N = 64, iters = 512, warmup_iters = 128, avg_ms = 0.0337056, gflop/s = 15.5549, checksum = 137.625, warmup_checksum = 34.4064, max_diff_vs_naive = 0
kernel = matmul, N = 128, iters = 256, warmup_iters = 64, avg_ms = 0.0440474, gflop/s = 95.2226, checksum = 152.013, warmup_checksum = 38.0032, max_diff_vs_naive = 0
kernel = matmul, N = 256, iters = 64, warmup_iters = 16, avg_ms = 0.236398, gflop/s = 141.941, checksum = 77.376, warmup_checksum = 19.344, max_diff_vs_naive = 0
kernel = matmul, N = 512, iters = 16, warmup_iters = 4, avg_ms = 1.29867, gflop/s = 206.7, checksum = 39.0672, warmup_checksum = 9.7668, max_diff_vs_naive = 0
kernel = matmul, N = 1024, iters = 8, warmup_iters = 2, avg_ms = 10.5165, gflop/s = 204.201, checksum = 39.296, warmup_checksum = 9.82399, max_diff_vs_naive = 0
kernel = matmul, N = 2048, iters = 4, warmup_iters = 1, avg_ms = 103.412, gflop/s = 166.131, checksum = 39.2923, warmup_checksum = 9.82308
kernel = matmul, N = 4096, iters = 3, warmup_iters = 1, avg_ms = 1516.32, gflop/s = 90.6399, checksum = 58.9486, warmup_checksum = 19.6495
```

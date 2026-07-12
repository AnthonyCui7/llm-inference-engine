/*
Persistent thread pool for tensor ops. Workers are spawned once and reused across
calls, with parallel_for splitting a row range into chunks across the pool.
*/
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

struct ThreadPool {
    // global pool, created on first use
    static ThreadPool& instance();

    // number of parallel lanes: worker threads + the calling thread
    int lanes() const;

    // split [0, total) into up to max_chunks contiguous ranges, run fn(start, end) on
    // the workers with the calling thread taking one chunk, return when all are done.
    // max_chunks = 0 means use every lane. Must not be called from inside a pool task.
    void parallel_for(int total, int max_chunks, const std::function<void(int, int)>& fn);

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

private:
    ThreadPool();
    ~ThreadPool();

    void worker_loop();

    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool stopping = false;
};

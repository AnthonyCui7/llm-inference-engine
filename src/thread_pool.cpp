#include <cassert>

#include "thread_pool.hpp"

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool;
    return pool;
}

ThreadPool::ThreadPool() {
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    if (hw < 1) hw = 1;

    // the calling thread participates in parallel_for, so keep hw - 1 workers
    int worker_count = hw - 1;
    workers.reserve(worker_count);
    for (int i = 0; i < worker_count; i++) {
        workers.emplace_back(&ThreadPool::worker_loop, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stopping = true;
    }
    queue_cv.notify_all();

    for (std::thread& worker : workers) {
        worker.join();
    }
}

int ThreadPool::lanes() const {
    return static_cast<int>(workers.size()) + 1;
}

void ThreadPool::worker_loop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [this] { return stopping || !tasks.empty(); });
            if (stopping && tasks.empty()) return;
            task = std::move(tasks.front());
            tasks.pop();
        }
        task();
    }
}

void ThreadPool::parallel_for(int total, int max_chunks, const std::function<void(int, int)>& fn) {
    assert(total >= 0);
    assert(max_chunks >= 0);

    if (total == 0) return;

    int chunks = lanes();
    if (max_chunks > 0 && max_chunks < chunks) chunks = max_chunks;
    if (chunks > total) chunks = total;

    if (chunks == 1) {
        fn(0, total);
        return;
    }

    // completion latch for this call; fn and the latch live on the caller's stack,
    // which stays alive until every chunk reports done
    std::mutex done_mutex;
    std::condition_variable done_cv;
    int pending = chunks - 1;

    int chunk_size = total / chunks;
    int remainder = total % chunks;

    int start = 0;
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        for (int c = 0; c < chunks - 1; c++) {
            int end = start + chunk_size + (c < remainder ? 1 : 0);
            tasks.emplace([&fn, &done_mutex, &done_cv, &pending, start, end] {
                fn(start, end);
                {
                    std::lock_guard<std::mutex> done_lock(done_mutex);
                    pending--;
                }
                done_cv.notify_one();
            });
            start = end;
        }
    }
    queue_cv.notify_all();

    // calling thread takes the last chunk
    fn(start, total);

    std::unique_lock<std::mutex> done_lock(done_mutex);
    done_cv.wait(done_lock, [&pending] { return pending == 0; });
}

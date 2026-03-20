# ThreadManager Design Document

## 1. Overview

### 1.1 Purpose

nntrainer에 통합 스레드 관리자(ThreadManager)를 도입하여, 현재 분산된 4개의
스레딩 메커니즘(TaskExecutor, BS::thread_pool, ParallelBatch, OpenMP)을
하나의 lightweight thread pool로 대체한다.

### 1.2 Goals

- **통합**: 4개 스레딩 메커니즘 → 1개 ThreadManager
- **Lightweight**: BS::thread_pool(2,850줄) 대비 ~500줄 이하의 구현
- **Zero-overhead parallel_for**: 텐서 연산 시 할당(allocation) 없음
- **안전한 비동기 I/O**: FSU load/unload의 race condition 해결
- **Compute/I/O 분리**: GEMM 성능에 FSU I/O가 간섭하지 않음

### 1.3 Non-Goals

- GPU/OpenCL 스레딩 (기존 ClContext가 관리)
- NUMA topology 최적화 (향후 확장)
- 동적 스레드 수 조정 (향후 확장)

---

## 2. Background: Current State & Problems

### 2.1 Existing Threading Mechanisms

| Component | Location | Usage | Problem |
|-----------|----------|-------|---------|
| TaskExecutor | `tensor/task_executor.h` | FSU load/unload | Task ID 재사용 race, 7개 동기화 버그 |
| BS::thread_pool | `utils/bs_thread_pool.h` | GGML GEMM/GEMV | 2,850줄 외부 라이브러리, compute/I/O 미분리 |
| ParallelBatch | `utils/nntr_threads.h` | Conv2D, Pooling | 매 호출마다 std::thread 생성/파괴 |
| OpenMP | compiler directives | SIMD, GEMM fallback | fork-join 오버헤드, 런타임 비용 |

### 2.2 FSU Architecture Problems

현재 FSU 호출 스택 (6 레이어):

```
Manager → TensorPool → CacheLoader → CachePool → CacheElem → SwapDevice
```

- 실제 I/O 작업은 SwapDevice에서만 수행
- CacheLoader는 100% 글루 코드 (TaskExecutor 래핑 + 상태 관리)
- CachePool, TensorPool도 대부분 위임만 수행
- complete_callback 파라미터가 사용되지 않고 무시됨

### 2.3 Critical Bugs in Current FSU Threading

1. **inActive()가 wait 없이 releaseTask()** → Task ID 즉시 재사용 → future 꼬임
2. **Load/Unload race condition** → checkUnloadComplete()와 lock(state_mutex) 사이 gap
3. **wait()의 silent return** → release된 ID에 대해 아무것도 안 기다리고 리턴
4. **flushCacheExcept() deadlock** → callback 안에서 mutex 획득 시도

---

## 3. Architecture

### 3.1 High-Level Design

```
┌──────────────────────────────────────────────────────────┐
│                 ThreadManager (Singleton)                 │
│                                                          │
│  ┌──────────────────────────┐  ┌───────────────────────┐ │
│  │   Compute Workers [0~N]  │  │   I/O Workers [0~M]   │ │
│  │                          │  │                        │ │
│  │  - parallel_for 전용      │  │  - submit() 전용       │ │
│  │  - atomic chunk counter  │  │  - lock-free MPSC queue│ │
│  │  - barrier 동기화         │  │  - cond_var wait       │ │
│  │  - spin-wait (짧은 대기)  │  │  - CompletionToken 반환│ │
│  │  - CPU affinity (opt)    │  │  - 블로킹 I/O 허용     │ │
│  └──────────────────────────┘  └───────────────────────┘ │
│                                                          │
│  Config:                                                 │
│    compute_threads = hardware_concurrency()              │
│    io_threads = 3                                        │
│    spin_wait_ns = 1000                                   │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Design Principles

1. **물리적 분리, 논리적 통합**: Compute와 I/O worker는 별도 스레드이나 하나의 API
2. **Zero allocation on hot path**: parallel_for는 heap 할당 없음
3. **CompletionToken으로 동기화**: Task ID 재사용 문제 원천 차단
4. **Singleton**: 프로세스당 하나, 어디서든 `ThreadManager::Global()` 접근

### 3.3 Replacing Existing Mechanisms

```
Before                              After
──────────────────────────────────────────────────────
TaskExecutor::submit(cb)         →  ThreadManager::submit(fn)
BS::thread_pool::submit_loop()  →  ThreadManager::parallel_for()
ParallelBatch::run()            →  ThreadManager::parallel_for()
#pragma omp parallel for        →  ThreadManager::parallel_for()
```

### 3.4 Simplified FSU Architecture

```
Before (6 layers):
  Manager → TensorPool → CacheLoader → CachePool → CacheElem → SwapDevice

After (3 layers):
  Manager → CacheElem → SwapDevice
            (ThreadManager가 비동기 처리)
```

CacheElem이 직접 atomic 상태 머신과 CompletionToken으로 동기화:

```cpp
class CacheElem {
  std::atomic<State> state_;  // Idle → Loading → Loaded → Unloading → Idle
  CompletionToken load_token_;

  void loadAsync() {
    State expected = State::Idle;
    if (!state_.compare_exchange_strong(expected, State::Loading))
      return;  // lock-free 중복 방지
    load_token_ = ThreadManager::Global().submit([this] {
      device_->getBuffer(...);
      state_.store(State::Loaded, std::memory_order_release);
    });
  }

  void waitLoad() { load_token_.wait(); }

  void unload() {
    device_->putBuffer(...);
    state_.store(State::Idle, std::memory_order_release);
  }
};
```

---

## 4. Detailed API Design

### 4.1 ThreadManager Class

```cpp
// thread_manager.h
namespace nntrainer {

struct ThreadManagerConfig {
  unsigned int compute_threads = std::thread::hardware_concurrency();
  unsigned int io_threads = 3;
  unsigned int spin_wait_ns = 1000;  // compute worker spin 대기 시간
  bool enable_affinity = false;
};

class ThreadManager : public Singleton<ThreadManager> {
  friend class Singleton<ThreadManager>;

public:
  // ─── Compute API (parallel_for) ──────────────────────
  //
  // 모든 compute worker가 참여. barrier로 동기화.
  // heap 할당 없음. 호출 스레드도 참여 (N+1 way).
  //
  template <typename F>
  void parallel_for(size_t begin, size_t end, F &&fn);

  // 스레드 수 지정 버전
  template <typename F>
  void parallel_for(size_t begin, size_t end, size_t n_threads, F &&fn);

  // ─── I/O API (async submit) ──────────────────────────
  //
  // I/O worker에서 실행. CompletionToken 반환.
  // compute worker에 영향 없음.
  //
  CompletionToken submit(std::function<void()> task);

  // ─── Query ───────────────────────────────────────────
  unsigned int getComputeThreadCount() const;
  unsigned int getIOThreadCount() const;

protected:
  ThreadManager();
  ~ThreadManager();
  void initialize() noexcept override;

private:
  // Compute workers
  std::vector<std::thread> compute_workers_;
  std::atomic<size_t> chunk_counter_;
  Barrier barrier_;

  // Shared state for parallel_for
  std::function<void(size_t)> current_task_;
  std::atomic<size_t> task_begin_;
  std::atomic<size_t> task_end_;
  std::atomic<bool> has_work_;

  // I/O workers
  std::vector<std::thread> io_workers_;
  MPSCQueue<std::function<void()>> io_queue_;
  std::mutex io_mutex_;
  std::condition_variable io_cv_;
  std::atomic<bool> stop_;

  ThreadManagerConfig config_;
};

} // namespace nntrainer
```

### 4.2 CompletionToken

```cpp
// completion_token.h
namespace nntrainer {

class CompletionToken {
public:
  CompletionToken() : state_(std::make_shared<SharedState>()) {}

  // 완료 대기 - 반드시 완료를 보장하거나 예외 발생
  void wait() {
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->cv.wait(lock, [this] { return state_->done; });
    if (state_->exception)
      std::rethrow_exception(state_->exception);
  }

  // non-blocking 완료 확인
  bool is_done() const {
    return state_->done.load(std::memory_order_acquire);
  }

  // 타임아웃 대기
  template <typename Duration>
  bool wait_for(Duration timeout) {
    std::unique_lock<std::mutex> lock(state_->mutex);
    return state_->cv.wait_for(lock, timeout,
                               [this] { return state_->done.load(); });
  }

private:
  friend class ThreadManager;

  void complete() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->done.store(true, std::memory_order_release);
    state_->cv.notify_all();
  }

  void fail(std::exception_ptr e) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->exception = e;
    state_->done.store(true, std::memory_order_release);
    state_->cv.notify_all();
  }

  struct SharedState {
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> done{false};
    std::exception_ptr exception{nullptr};
  };

  std::shared_ptr<SharedState> state_;
};

} // namespace nntrainer
```

### 4.3 Barrier (Compute Worker 동기화)

```cpp
// barrier.h (내부 구현)
namespace nntrainer {

class Barrier {
public:
  explicit Barrier(unsigned int count) : threshold_(count), count_(count),
                                         generation_(0) {}

  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    auto gen = generation_;
    if (--count_ == 0) {
      ++generation_;
      count_ = threshold_;
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this, gen] { return gen != generation_; });
    }
  }

  void reset(unsigned int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    threshold_ = count;
    count_ = count;
    ++generation_;
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_;
  unsigned int threshold_;
  unsigned int count_;
  unsigned int generation_;
};

} // namespace nntrainer
```

---

## 5. Implementation Details

### 5.1 Compute Worker Loop

```cpp
void ThreadManager::compute_worker_loop(unsigned int worker_id) {
  while (!stop_.load(std::memory_order_relaxed)) {

    // Phase 1: 작업 대기 (spin-wait → yield → sleep)
    while (!has_work_.load(std::memory_order_acquire)) {
      if (stop_.load(std::memory_order_relaxed)) return;

      // Adaptive wait: spin → yield → condition variable
      for (unsigned int i = 0; i < config_.spin_wait_ns / 10; ++i) {
        if (has_work_.load(std::memory_order_acquire)) break;
        std::this_thread::yield();
      }
      if (!has_work_.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lock(compute_mutex_);
        compute_cv_.wait(lock, [this] {
          return has_work_.load() || stop_.load();
        });
      }
    }

    // Phase 2: 동적 chunk 분배 (atomic fetch_add)
    size_t begin = task_begin_.load(std::memory_order_relaxed);
    size_t end = task_end_.load(std::memory_order_relaxed);

    while (true) {
      size_t idx = chunk_counter_.fetch_add(1, std::memory_order_relaxed);
      if (idx >= end) break;
      current_task_(idx);
    }

    // Phase 3: barrier 동기화 (모든 worker 완료 대기)
    barrier_.wait();
  }
}
```

### 5.2 parallel_for Implementation

```cpp
template <typename F>
void ThreadManager::parallel_for(size_t begin, size_t end, F &&fn) {
  if (begin >= end) return;

  size_t range = end - begin;

  // 작업이 작으면 호출 스레드에서 직접 실행
  if (range == 1 || compute_workers_.empty()) {
    for (size_t i = begin; i < end; ++i) fn(i);
    return;
  }

  // Compute workers에 작업 배포
  current_task_ = std::forward<F>(fn);
  task_begin_.store(begin, std::memory_order_relaxed);
  task_end_.store(end, std::memory_order_relaxed);
  chunk_counter_.store(begin, std::memory_order_relaxed);

  // Workers 깨우기
  has_work_.store(true, std::memory_order_release);
  compute_cv_.notify_all();

  // 호출 스레드도 참여 (N+1 way parallelism)
  while (true) {
    size_t idx = chunk_counter_.fetch_add(1, std::memory_order_relaxed);
    if (idx >= end) break;
    fn(idx);
  }

  // 모든 worker 완료 대기
  barrier_.wait();
  has_work_.store(false, std::memory_order_release);
}
```

### 5.3 I/O Worker Loop

```cpp
void ThreadManager::io_worker_loop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(io_mutex_);
      io_cv_.wait(lock, [this] {
        return !io_queue_.empty() || stop_.load();
      });
      if (stop_.load() && io_queue_.empty()) return;
      task = std::move(io_queue_.front());
      io_queue_.pop();
    }
    task();
  }
}
```

### 5.4 submit Implementation

```cpp
CompletionToken ThreadManager::submit(std::function<void()> task) {
  CompletionToken token;
  auto state = token.state_;

  {
    std::lock_guard<std::mutex> lock(io_mutex_);
    io_queue_.push([task = std::move(task), state]() {
      try {
        task();
        state->done.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lk(state->mutex);
        state->cv.notify_all();
      } catch (...) {
        state->exception = std::current_exception();
        state->done.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lk(state->mutex);
        state->cv.notify_all();
      }
    });
  }
  io_cv_.notify_one();
  return token;
}
```

---

## 6. Migration Guide

### 6.1 BS::thread_pool → ThreadManager::parallel_for

```cpp
// Before (ggml_interface_bs_threadpool.cpp)
auto &bs_thread_pool = ThreadPoolManager::Global().getThreadPool();
int thread_num = bs_thread_pool.get_thread_count();
BS::multi_future<void> loop_future =
  bs_thread_pool.submit_loop(0, thread_num, [=](int i) {
    unsigned int start = (i * N) / thread_num;
    unsigned int end = ((i + 1) * N) / thread_num;
    nntr_gemv_q4_0_4x8_q8_0(K, C + start, N, B, QA.data(), M, end - start);
  });
loop_future.wait();

// After
auto &tm = ThreadManager::Global();
tm.parallel_for(0, N, [=](size_t col) {
  // 각 col에 대해 연산 (dynamic chunk 분배)
  nntr_gemv_q4_0_4x8_q8_0_single(K, C + col, N, B, QA.data(), M);
});
```

또는 기존 chunked 패턴 유지:

```cpp
auto &tm = ThreadManager::Global();
unsigned int n_threads = tm.getComputeThreadCount();
tm.parallel_for(0, n_threads, [=](size_t i) {
  unsigned int start = (i * N) / n_threads;
  unsigned int end = ((i + 1) * N) / n_threads;
  nntr_gemv_q4_0_4x8_q8_0(K, C + start, N, B, QA.data(), M, end - start);
});
```

### 6.2 ParallelBatch → ThreadManager::parallel_for

```cpp
// Before (conv2d_layer.cpp)
auto workers = ParallelBatch(forwarding_job, in_dim.batch(), nullptr);
if (workers.getNumWorkers() > 1) {
  workers.run();
} else {
  forwarding_job(0, in_dim.batch(), 0, nullptr);
}

// After
auto &tm = ThreadManager::Global();
tm.parallel_for(0, in_dim.batch(), [&](size_t b) {
  // batch b에 대한 forwarding
  forwarding_single_batch(b, user_data);
});
```

### 6.3 TaskExecutor → ThreadManager::submit + CompletionToken

```cpp
// Before (cache_loader.cpp)
load_task_executor = new TaskExecutor("loadPool", 2);
int task_id = load_task_executor->submit([this, id](void *) {
  pool->loadTensor(id);
  std::lock_guard<std::mutex> lock(state_mutex);
  states[id] = LoadState::Loaded;
}, nullptr);
load_task_executor->wait(task_id);

// After (cache_elem.cpp에서 직접)
auto token = ThreadManager::Global().submit([this] {
  device_->getBuffer(offset_, length_, memory_ptr_, id_, alloc_only_);
  state_.store(State::Loaded, std::memory_order_release);
});
token.wait();  // 반드시 완료 보장, silent return 없음
```

### 6.4 FSU Flow Migration

```cpp
// Before: neuralnet.cpp forwarding
model_graph.LoadTensors(i);            // → CacheLoader → TaskExecutor
model_graph.checkLoadComplete(f);      // → CacheLoader::wait
node->forwarding(training);
model_graph.inActive(f);               // → releaseTask (wait 안함!)
model_graph.LoadTensors(f + lookahead);

// After: neuralnet.cpp forwarding
model_graph.loadAsync(i);              // → CacheElem::loadAsync (direct)
model_graph.waitLoad(f);               // → CompletionToken::wait (보장)
node->forwarding(training);
model_graph.unload(f);                 // → CacheElem::unload (동기)
model_graph.loadAsync(f + lookahead);  // → prefetch
```

---

## 7. File Structure

```
nntrainer/utils/
├── thread_manager.h           # ThreadManager 클래스 선언
├── thread_manager.cpp         # ThreadManager 구현
├── completion_token.h         # CompletionToken 클래스
├── barrier.h                  # Barrier (내부용)
│
├── singleton.h                # (기존) 유지
├── bs_thread_pool.h           # (제거 예정)
├── bs_thread_pool_manager.hpp # (제거 예정)
├── bs_thread_pool_manager.cpp # (제거 예정)
├── nntr_threads.h             # (제거 예정)
└── nntr_threads.cpp           # (제거 예정)

nntrainer/tensor/
├── task_executor.h            # (제거 예정)
├── task_executor.cpp          # (제거 예정)
├── cache_loader.h             # (제거 예정)
├── cache_loader.cpp           # (제거 예정)
├── cache_elem.h               # (수정) 직접 ThreadManager 사용
└── cache_elem.cpp             # (수정) atomic 상태 머신 + CompletionToken
```

---

## 8. Safety Guarantees

### 8.1 vs Current Bugs

| Bug | 현재 | ThreadManager |
|-----|------|---------------|
| Task ID 재사용 race | releaseTask → 즉시 재사용 | CompletionToken (일회용, 재사용 없음) |
| wait() silent return | ID 없으면 그냥 리턴 | token.wait()는 반드시 완료 또는 예외 |
| Load/Unload race | mutex gap 존재 | atomic CAS로 상태 전이 (lock-free) |
| Deadlock in callback | callback 내 mutex 획득 | callback에서 mutex 안 잡음 |
| inActive가 wait 안함 | releaseTask만 호출 | unload()는 동기, 즉시 완료 |

### 8.2 Thread Safety Rules

1. `parallel_for`는 **재진입 불가** (한 번에 하나의 parallel_for만 실행)
2. `submit`은 **다중 스레드에서 안전** (MPSC queue)
3. `CompletionToken`은 **다중 스레드에서 wait 가능** (shared_ptr + cv)
4. Compute workers는 **I/O 태스크를 실행하지 않음** (간섭 없음)
5. I/O workers는 **parallel_for에 참여하지 않음** (deadlock 방지)

---

## 9. Performance Considerations

### 9.1 parallel_for Overhead

| 항목 | BS::thread_pool | ThreadManager |
|------|----------------|---------------|
| 태스크 제출 | std::function 할당 | atomic store (zero alloc) |
| 동기화 | multi_future vector 할당 | barrier (zero alloc) |
| chunk 분배 | 정적 블록 | atomic fetch_add (동적) |
| 호출 스레드 | 참여 안 함 | 참여 (N+1 way) |

### 9.2 I/O Submit Overhead

| 항목 | TaskExecutor | ThreadManager |
|------|-------------|---------------|
| 완료 추적 | map + shared_future | CompletionToken (shared_ptr 1개) |
| ID 관리 | map lookup + reusable queue | 없음 (token이 곧 핸들) |
| 상태 추적 | 별도 LoadState map | atomic<State> in CacheElem |

### 9.3 Compute vs I/O Isolation

```
GEMM 실행 중 FSU prefetch가 동시에 일어나는 경우:

Compute Workers [0..7]:  ▓▓▓▓ GEMM (8 threads 전부 사용) ▓▓▓▓
I/O Workers [0..2]:      ████ swapIn (disk read) ████

→ GEMM은 항상 모든 compute worker 사용 보장
→ I/O는 별도 스레드에서 실행, cache 오염 없음
```

---

## 10. Migration Plan

### Phase 1: ThreadManager Core
- `thread_manager.h/cpp`, `completion_token.h`, `barrier.h` 구현
- 단위 테스트 작성

### Phase 2: GGML Interface Migration
- `ggml_interface_bs_threadpool.cpp` → `ThreadManager::parallel_for` 전환
- BS::thread_pool 제거

### Phase 3: ParallelBatch Migration
- Conv2D, Pooling, LSTM 등 → `ThreadManager::parallel_for` 전환
- `nntr_threads.h/cpp` 제거

### Phase 4: FSU Simplification
- CacheElem에 atomic 상태 머신 + CompletionToken 직접 통합
- CacheLoader 제거
- Manager/TensorPool의 FSU 코드 단순화

### Phase 5: Cleanup
- TaskExecutor 제거
- OpenMP 의존성 점진적 축소 (SIMD 전용으로 유지 가능)
- 기존 테스트 마이그레이션

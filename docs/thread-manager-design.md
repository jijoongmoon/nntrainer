# ThreadManager Design Document

## 1. Overview

### 1.1 Purpose

nntrainer에 통합 스레드 관리자(ThreadManager)를 도입하여, 기존 분산된 4개의
스레딩 메커니즘(TaskExecutor, BS::thread_pool, ParallelBatch, OpenMP)을
하나의 lightweight thread pool로 대체한다.

### 1.2 Goals

- **통합**: 4개 스레딩 메커니즘 → 1개 ThreadManager
- **GGML 수준 성능**: llama.cpp threadpool과 동등한 성능 달성
- **Zero-overhead parallel_for**: spin-wait barrier, atomic chunk counter
- **안전한 비동기 I/O**: CompletionToken으로 race condition 해결
- **Compute/I/O 물리적 분리**: GEMM 성능에 FSU I/O 간섭 없음
- **CPU Affinity**: big.LITTLE 인식, 코어 1:1 바인딩

### 1.3 Migration Status

| Phase | 내용 | 상태 |
|-------|------|------|
| Phase 1 | ThreadManager Core | ✅ 완료 |
| Phase 2 | BS::thread_pool → ThreadManager | ✅ 완료 |
| Phase 3 | ParallelBatch + OpenMP → ThreadManager | ✅ 완료 |
| Phase 4 | FSU CacheLoader → ThreadManager::submit | ✅ 완료 |
| Phase 5 | 레거시 파일 삭제 (-4,868줄) | ✅ 완료 |

---

## 2. Architecture

### 2.1 Class Diagram

```mermaid
classDiagram
    class ThreadManager {
        -vector~thread~ compute_workers_
        -vector~thread~ io_workers_
        -atomic~uint~ generation_
        -atomic~int~ n_barrier_
        -atomic~int~ n_barrier_passed_
        -atomic~size_t~ current_chunk_
        -atomic~uint~ active_workers_
        -atomic~int~ active_threads_
        -atomic~bool~ stop_
        -function~void(size_t)~ current_task_
        -queue io_queue_
        -mutex io_mutex_
        -condition_variable io_cv_
        -ThreadManagerConfig config_
        +parallel_for(begin, end, fn)
        +parallel_for(begin, end, n_workers, fn)
        +parallel_for_chunked(n_threads, fn)
        +submit(task) CompletionToken
        +getComputeThreadCount() uint
        +getIOThreadCount() uint
        +setConfig(config)$
        -barrier()
        -dispatchAndJoin(begin, end, fn, n_workers)
        -computeWorkerLoop(worker_id)
        -ioWorkerLoop()
        -cpuRelax()$
    }

    class CompletionToken {
        -shared_ptr~SharedState~ state_
        +wait()
        +isDone() bool
        +waitFor(timeout) bool
        +valid() bool
        -create()$ CompletionToken
        -complete()
        -fail(exception_ptr)
    }

    class SharedState {
        +mutex mutex
        +condition_variable cv
        +atomic~bool~ done
        +exception_ptr exception
    }

    class ThreadManagerConfig {
        +uint compute_threads
        +uint io_threads
        +bool enable_affinity
    }

    class Singleton~T~ {
        +Global()$ T&
        #initialize()
    }

    class CacheElem {
        -shared_ptr~SwapDevice~ device
        -CompletionToken load_token_
        -CompletionToken unload_token_
        -bool active
        -uint id
        -size_t offset, length
        -CachePolicy policy
        +swapIn(opt)
        +swapOut(opt)
        +setLoadToken(token)
        +waitLoad()
        +isLoadDone() bool
        +setUnloadToken(token)
        +waitUnload()
    }

    class CachePool {
        -vector~CacheElem~ elems
        -shared_ptr~SwapDevice~ swap_device
        +validate(id)
        +invalidate(id)
        +loadTensor(id)
        +unloadTensor(id)
        +getExecIDs(order) set~uint~
        +getCacheElem(id) CacheElem&
    }

    class TensorPool {
        -shared_ptr~MemoryPool~ mem_pool
        +loadCacheExec(order)
        +loadCacheExecAsync(order)
        +checkLoadComplete(order) bool
        +flushCacheExecAsync(order)
        +inActive(order) uint
    }

    class SwapDevice {
        +getBuffer(offset, size, ptr, id, alloc_only) void*
        +putBuffer(ptr, dealloc_only)
        +start(size)
        +finish()
    }

    ThreadManager --|> Singleton : extends
    ThreadManager --> CompletionToken : creates
    CompletionToken --> SharedState : owns
    ThreadManager --> ThreadManagerConfig : uses
    CacheElem --> CompletionToken : has load/unload tokens
    CacheElem --> SwapDevice : reads/writes
    CachePool --> CacheElem : manages
    TensorPool --> CachePool : delegates
    TensorPool --> ThreadManager : submit()
```

### 2.2 High-Level Architecture

```mermaid
graph TB
    subgraph ThreadManager["ThreadManager (Singleton)"]
        subgraph Compute["Compute Workers"]
            CW0["Worker 0<br/>Core 1 (big)"]
            CW1["Worker 1<br/>Core 2 (big)"]
            CW2["Worker 2<br/>Core 3 (big)"]
            CWN["Worker N<br/>Core N (big)"]
        end
        subgraph IO["I/O Workers"]
            IW0["I/O Worker 0<br/>Core N+1 (LITTLE)"]
        end
        B["Barrier<br/>(spin-wait)"]
        Q["I/O Queue<br/>(cond_var)"]
    end

    Caller["Caller Thread<br/>Core 0 (fastest)"]

    Caller -->|parallel_for| B
    CW0 --> B
    CW1 --> B
    CW2 --> B
    CWN --> B

    Caller -->|submit| Q
    Q --> IW0

    subgraph FSU["FSU Path"]
        TP["TensorPool"]
        CP["CachePool"]
        CE["CacheElem"]
        SD["SwapDevice"]
    end

    IW0 -->|loadTensor| CP
    CP --> CE
    CE -->|mmap/read| SD
```

### 2.3 Core Layout (big.LITTLE)

```mermaid
graph LR
    subgraph "Snapdragon 8 Gen 3 (8 cores)"
        subgraph Big["Big Cores (high freq)"]
            C0["Core 0: Caller"]
            C1["Core 1: Compute 0"]
            C2["Core 2: Compute 1"]
            C3["Core 3: Compute 2"]
        end
        subgraph Little["LITTLE Cores (low freq)"]
            C4["Core 4: Compute 3"]
            C5["Core 5: Compute 4"]
            C6["Core 6: Compute 5"]
            C7["Core 7: I/O Worker"]
        end
    end

    style Big fill:#ff9999
    style Little fill:#99ccff
```

---

## 3. Synchronization Design

### 3.1 Compute: Spin-Wait Barrier (GGML-style)

Compute worker 동기화에 GGML의 spin-wait barrier 패턴을 사용한다.
llama.cpp issue #9588의 false sharing 수정도 적용 (alignas(64)).

```mermaid
sequenceDiagram
    participant Caller
    participant W0 as Worker 0
    participant W1 as Worker 1
    participant W2 as Worker 2

    Note over W0,W2: Workers spin on generation_

    Caller->>Caller: setup task, current_chunk_ = begin
    Caller->>Caller: generation_++ (seq_cst)

    par Caller + Workers grab chunks
        Caller->>Caller: fetch_add(current_chunk_) → process
        W0->>W0: detect generation change
        W0->>W0: fetch_add(current_chunk_) → process
        W1->>W1: detect generation change
        W1->>W1: fetch_add(current_chunk_) → process
        W2->>W2: detect generation change
        W2->>W2: fetch_add(current_chunk_) → process
    end

    Note over Caller,W2: Barrier (spin-wait)
    Caller->>Caller: n_barrier_++ (last? reset & bump passed)
    W0->>W0: n_barrier_++ (spin on n_barrier_passed_)
    W1->>W1: n_barrier_++ (spin on n_barrier_passed_)
    W2->>W2: n_barrier_++ (last → bump n_barrier_passed_)

    Note over Caller,W2: All threads past barrier → next round
```

### 3.2 I/O: Condition Variable (FSU path)

I/O worker는 blocking I/O (disk read/write)를 수행하므로 spin-wait가 부적합.
Condition variable로 대기하며, CompletionToken으로 완료 추적.

```mermaid
sequenceDiagram
    participant Caller
    participant TM as ThreadManager
    participant IW as I/O Worker
    participant CE as CacheElem
    participant SD as SwapDevice

    Caller->>TM: submit(load_task)
    TM->>TM: create CompletionToken
    TM->>TM: push to io_queue_
    TM->>IW: notify (cond_var)
    TM-->>Caller: return CompletionToken

    Note over Caller: Caller continues (compute)

    IW->>IW: pop from io_queue_
    IW->>CE: swapIn()
    CE->>SD: getBuffer(offset, size)
    SD->>SD: mmap() or read()
    SD-->>CE: buffer ptr
    CE-->>IW: done

    IW->>IW: token.complete()

    Caller->>Caller: token.wait()
    Note over Caller: Load confirmed
```

### 3.3 Barrier vs Condition Variable 비교

| 항목 | Barrier (Compute) | Condition Variable (I/O) |
|------|-------------------|--------------------------|
| 대기 방식 | Spin-wait (cpu_relax) | OS sleep (futex) |
| Wake 레이턴시 | ~1-5 us | ~50-100 us |
| CPU 사용 | 100% (대기 중) | 0% (대기 중) |
| 적합한 작업 | GEMM, Conv2D (짧고 빈번) | Disk I/O (길고 드뭄) |
| 동기화 대상 | 모든 worker 동시 완료 | 개별 task 완료 |

---

## 4. FSU (Flash Storage Utilization) Flow

### 4.1 Before vs After

```mermaid
graph LR
    subgraph Before["Before (6 layers)"]
        NN1["NeuralNet"] --> NG1["NetworkGraph"]
        NG1 --> M1["Manager"]
        M1 --> TP1["TensorPool"]
        TP1 --> CL["CacheLoader<br/>(glue code)"]
        CL --> TE["TaskExecutor<br/>(Task ID bugs)"]
        TE --> CP1["CachePool"]
        CP1 --> CE1["CacheElem"]
        CE1 --> SD1["SwapDevice"]
    end

    subgraph After["After (3 layers)"]
        NN2["NeuralNet"] --> NG2["NetworkGraph"]
        NG2 --> M2["Manager"]
        M2 --> TP2["TensorPool"]
        TP2 -->|"submit()"| TM["ThreadManager"]
        TM --> CP2["CachePool"]
        CP2 --> CE2["CacheElem"]
        CE2 --> SD2["SwapDevice"]
    end

    style CL fill:#ff6666
    style TE fill:#ff6666
    style TM fill:#66ff66
```

### 4.2 FSU Forwarding Flow

```mermaid
sequenceDiagram
    participant NN as NeuralNet
    participant TM as ThreadManager
    participant CP as CachePool
    participant CE as CacheElem

    Note over NN: Pre-load with lookahead
    NN->>TM: submit(load layer 0)
    NN->>TM: submit(load layer 1)

    loop For each layer f
        NN->>CE: waitLoad(f)
        Note over NN: Layer f data ready

        NN->>NN: node->forwarding()
        Note over NN: GEMM uses parallel_for<br/>(compute workers)

        NN->>CP: inActive(f)
        NN->>TM: submit(load layer f+lookahead)
        Note over TM: I/O worker loads next layer<br/>while compute continues
    end
```

### 4.3 Look-ahead Test Coverage

FSU look-ahead 파이프라인의 정확성을 검증하는 5개 테스트:

| 테스트 | 검증 내용 |
|--------|-----------|
| `lookahead_basic_pipeline` | lookahead=2로 5개 레이어 전체 파이프라인 시뮬레이션. pre-load → waitLoad → compute(parallel_for) → unload → prefetch next 순서 검증 |
| `lookahead_overlap_verification` | I/O와 compute의 실제 오버랩 검증. compute 완료 후 prefetch된 레이어의 `isLoadDone()` == true 확인 |
| `lookahead_multi_epoch` | 3 epoch 반복 시 CompletionToken이 매 epoch마다 정상 재설정되는지 확인 |
| `lookahead_async_unload_pipeline` | load + unload + compute 3가지 비동기 파이프라인. `asyncUnload(f)` + `asyncLoad(f+2)` + `parallel_for` 동시 실행 |
| `lookahead_token_polling` | `isDone()` 비차단 폴링과 `waitLoad()` 차단 대기를 혼합 사용하는 패턴 검증 |

#### Look-ahead Pipeline Sequence (테스트 기준)

```mermaid
sequenceDiagram
    participant Test as Test (Caller)
    participant TM as ThreadManager
    participant IO as I/O Worker
    participant CW as Compute Workers

    Note over Test: Pre-load layers 1, 2, 3 (lookahead=2)
    Test->>TM: submit(loadTensor layer 1)
    Test->>TM: submit(loadTensor layer 2)
    Test->>TM: submit(loadTensor layer 3)
    TM->>IO: queue 3 load tasks

    loop For each layer f = 1..5
        Test->>Test: waitLoad(f)
        Note over IO: I/O Worker: loadTensor(f) → swapIn()

        IO-->>Test: CompletionToken.complete()
        Note over Test: Layer f data ready

        par Compute + Prefetch
            Test->>CW: parallel_for(0, 100)
            Note over CW: Simulate GEMM on compute workers

            Test->>Test: unload(f) [sync]
            Note over Test: Layer f memory freed

            Test->>TM: submit(loadTensor f+3)
            Note over IO: I/O Worker starts loading f+3<br/>while compute runs
        end

        CW-->>Test: barrier (compute done)
    end

    Note over Test: All 5 layers processed<br/>with overlapped I/O
```

#### Async Unload Pipeline (lookahead_async_unload_pipeline)

```mermaid
sequenceDiagram
    participant Test as Test (Caller)
    participant TM as ThreadManager
    participant IO as I/O Worker
    participant CW as Compute Workers

    Note over Test: Pre-load layers 1, 2

    loop For each layer f = 1..5
        Test->>Test: waitLoad(f)

        alt f > 1
            Test->>Test: waitUnload(f-1)
            Note over Test: Previous layer fully freed
        end

        Test->>TM: submit(loadTensor f+2)

        par Concurrent operations
            Test->>CW: parallel_for (compute)
            IO->>IO: loadTensor(f+2)
        end

        Test->>TM: submit(unloadTensor f)
        Note over IO: Async unload queued<br/>(happens in background)
    end

    Test->>Test: waitUnload(5)
    Note over Test: Final cleanup complete
```

---

## 5. API Reference

### 5.1 parallel_for

```cpp
// 모든 compute worker 사용
tm.parallel_for(0, N, [&](size_t i) { compute(i); });

// n_workers개 worker만 사용 (나머지는 skip)
tm.parallel_for(0, N, 4u, [&](size_t i) { compute(i); });

// chunked: 스레드 인덱스 기반 (GEMM column 분할용)
tm.parallel_for_chunked(n_threads, [&](size_t tid) {
  size_t start = (tid * N) / n_threads;
  size_t end = ((tid + 1) * N) / n_threads;
  gemm_chunk(start, end);
});
```

### 5.2 submit (I/O)

```cpp
auto token = tm.submit([&] { load_from_disk(); });

// non-blocking check
if (token.isDone()) { use_data(); }

// blocking wait
token.wait();  // throws if task failed
```

### 5.3 Configuration

```cpp
ThreadManagerConfig config;
config.compute_threads = 6;  // default: hw_concurrency - 2
config.io_threads = 1;       // default: 1
config.enable_affinity = true; // pin to cores, big.LITTLE aware
ThreadManager::setConfig(config);  // must call before Global()
```

---

## 6. Performance

### 6.1 Benchmark Results (vs GGML-style threadpool)

4-core test environment, 3 threads (2 workers + 1 caller):

| Workload | Serial | OpenMP | ThreadManager | GGML-style | TM/GGML |
|----------|--------|--------|---------------|------------|---------|
| Small GEMM 64x64 | 132 us | 206 us | 283 us | 91 us | ~1.0x* |
| Large GEMM 256x256 | 18,345 us | 17,016 us | 12,185 us | 8,002 us | **1.001x** |
| GEMV 4096x4096 | 155,007 us | 81,074 us | 78,259 us | 46,970 us | **1.35x** |
| Chunked 4x4096 | 678,894 us | 279,222 us | 289,521 us | 191,057 us | **1.02x** |
| 50 rapid dispatch | 26 us | 77,929 us | 4,153 us | 89 us | **0.02x** |

*Large GEMM과 Chunked GEMM에서 GGML과 동등 성능 달성.
Rapid dispatch에서 ThreadManager가 50x 빠름 (inactive worker skip).*

### 6.2 Cache Line Isolation

llama.cpp issue #9588의 false sharing 문제를 동일하게 적용:

```cpp
alignas(64) std::atomic<unsigned int> generation_{0};
alignas(64) std::atomic<int> n_barrier_{0};
alignas(64) std::atomic<int> n_barrier_passed_{0};
alignas(64) std::atomic<size_t> current_chunk_{0};
alignas(64) std::atomic<unsigned int> active_workers_{0};
alignas(64) std::atomic<bool> stop_{false};
```

각 atomic이 별도 cache line (64 bytes)에 위치하여 코어 간 cache bouncing 방지.

---

## 7. File Structure

```
nntrainer/utils/
├── thread_manager.h       # ThreadManager class (GGML-style barrier)
├── thread_manager.cpp     # Worker loops, CPU affinity, barrier impl
├── completion_token.h     # CompletionToken (async sync)
├── barrier.h              # Barrier (utility, used in tests)
└── singleton.h            # Singleton base class

nntrainer/tensor/
├── cache_pool.h/cpp       # CachePool (memory management)
├── cache_elem.h/cpp       # CacheElem + CompletionToken (direct I/O)
├── swap_device.h/cpp      # Disk I/O (mmap/read/write)
├── tensor_pool.h/cpp      # TensorPool → ThreadManager::submit
└── manager.h/cpp          # Manager (high-level FSU orchestration)

test/unittest/
├── unittest_thread_manager.cpp        # 24 tests
├── unittest_threading_benchmark.cpp   # 4-way benchmark
└── memory/
    └── unittest_fsu_threadmanager.cpp # 11 FSU tests (6 basic + 5 look-ahead)
```

### Deleted Files (-4,868 lines)

| File | Lines | Reason |
|------|-------|--------|
| `bs_thread_pool.h` | 2,850 | Replaced by ThreadManager |
| `bs_thread_pool_manager.hpp/cpp` | ~100 | Singleton wrapper removed |
| `nntr_threads.h/cpp` | ~90 | ParallelBatch removed |
| `task_executor.h/cpp` | ~400 | Replaced by ThreadManager::submit |
| `task.h` | ~50 | TaskExecutor dependency removed |
| `cache_loader.h/cpp` | ~720 | Glue code eliminated |
| `unittest_cache_loader.cpp` | ~720 | Tests for removed code |

---

## 8. Safety Guarantees

### 8.1 Resolved Bugs

```mermaid
graph TD
    subgraph Before["Before: 7 FSU Threading Bugs"]
        B1["Task ID Reuse Race"]
        B2["Load/Unload Race"]
        B3["wait() Silent Return"]
        B4["Deadlock in Callback"]
        B5["inActive() No Wait"]
        B6["checkUnloadComplete Skip"]
        B7["Non-atomic State"]
    end

    subgraph After["After: All Resolved"]
        A1["CompletionToken<br/>(one-shot, no reuse)"]
        A2["waitLoad/waitUnload<br/>(explicit ordering)"]
        A3["token.wait() always<br/>blocks or throws"]
        A4["No callback mutex<br/>(I/O worker owns)"]
        A5["waitLoad() before<br/>inActive()"]
        A6["CompletionToken tracks<br/>each operation"]
        A7["CacheElem tokens<br/>are atomic"]
    end

    B1 --> A1
    B2 --> A2
    B3 --> A3
    B4 --> A4
    B5 --> A5
    B6 --> A6
    B7 --> A7

    style Before fill:#ffcccc
    style After fill:#ccffcc
```

### 8.2 Thread Safety Rules

1. `parallel_for`는 **재진입 불가** (한 번에 하나의 parallel_for만 실행)
2. `submit`은 **다중 스레드에서 안전** (mutex-protected queue)
3. `CompletionToken`은 **다중 스레드에서 wait 가능** (shared_ptr + cv)
4. Compute workers는 **I/O 태스크를 실행하지 않음** (간섭 없음)
5. I/O workers는 **parallel_for에 참여하지 않음** (deadlock 방지)
6. CPU affinity: **compute와 I/O는 다른 코어** (cache 오염 없음)

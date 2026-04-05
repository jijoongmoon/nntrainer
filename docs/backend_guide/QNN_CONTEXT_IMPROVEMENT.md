# QNN Context 개선 제안

## 현재 문제점

### 1. 안전하지 않은 `static_pointer_cast`

```cpp
// 현재: crash 위험
std::shared_ptr<QNNVar> getQNNVar(RunLayerContext &context) {
  return static_pointer_cast<QNNBackendVar>(context.getContextData())
    ->getVar();  // CPU context가 전달되면 undefined behavior!
}
```

### 2. QNN이 ComputeOps를 설정하지 않음

```cpp
// 현재: ComputeOps가 nullptr
void QNNContext::initialize() {
  init();
  setMemAllocator(...);
  // ComputeOps 설정 없음! → 일반 텐서 연산 불가
}
```

### 3. QNNVar가 monolithic struct

QNNVar에 backend handle, function pointers, RPC memory, IO tensor,
context-graph map이 모두 하나의 struct에 있음.

---

## 개선 방향

### 개선 1: ContextData에 type-safe 접근 메서드 추가

```cpp
// context_data.h
class ContextData {
public:
  // Type-safe downcast (returns nullptr if wrong type)
  template<typename T>
  T* as() { return dynamic_cast<T*>(this); }

  template<typename T>
  const T* as() const { return dynamic_cast<const T*>(this); }

  // ... existing members ...
};
```

QNN 레이어에서의 사용:
```cpp
// 개선: 안전한 타입 체크
std::shared_ptr<QNNVar> getQNNVar(RunLayerContext &context) {
  auto *qnn_data = context.getContextData()->as<QNNBackendVar>();
  NNTR_THROW_IF(!qnn_data, std::runtime_error)
    << "QNNGraph requires QNN context, got: " << context.getContextData()->getType();
  return qnn_data->getVar();
}
```

### 개선 2: 모든 Context가 ComputeOps를 설정하도록 강제

```cpp
// QNNContext::initialize()
void QNNContext::initialize() noexcept {
  // 1. CPU backend 초기화 (fallback ops)
  init_backend();

  // 2. ComputeOps 설정 — QNN에서도 필수
  //    QNN 전용 ops가 있으면 교체, 없으면 CPU fallback
  getContextData()->setComputeOps(g_compute_ops);

  // 3. QNN-specific 초기화
  init();
  setMemAllocator(std::make_shared<QNNRpcManager>());

  // 4. QNN 레이어 등록
  registerFactory(...);
}
```

이렇게 하면:
- QNN 모델에서 일반 텐서 연산 (전처리, 후처리)이 정상 동작
- QNN이 특정 ops를 가속하고 싶으면 ops table 교체 가능

### 개선 3: ContextData에 type 식별자 추가

```cpp
class ContextData {
public:
  virtual ~ContextData() = default;

  // Backend type identification
  virtual const char* getType() const { return "cpu"; }

  // ... existing members ...
};

class QNNBackendVar : public ContextData {
public:
  const char* getType() const override { return "qnn"; }
  // ...
};
```

### 개선 4: Context::initialize()에서 ComputeOps 설정을 기본 동작으로

```cpp
// context.h — Context base class
class Context {
protected:
  // 서브클래스가 override하지 않으면 CPU fallback ops 사용
  virtual void setupComputeOps() {
    if (auto cd = getContextData(); cd && !cd->getComputeOps()) {
      ensureComputeOps();  // init_backend() if needed
      cd->setComputeOps(g_compute_ops);
    }
  }
};
```

---

## 전체 아키텍처 (개선 후)

```
Context (base)
  │
  ├── AppContext ("cpu")
  │     └── ContextData { ComputeOps = arm_ops/x86_ops }
  │
  ├── ClContext ("gpu")
  │     └── ContextData { ComputeOps = opencl_ops }
  │
  └── QNNContext ("qnn")
        └── QNNBackendVar : ContextData {
              ComputeOps = cpu_fallback_ops (일반 텐서 연산용)
              QNNVar = { backend handle, graphs, sessions }
            }

레이어 실행:
  일반 레이어 → context.getComputeOps()->sgemm_fp32(...)
  QNN 레이어 → context.getContextData()->as<QNNBackendVar>()->getVar()
                → QNN 그래프 실행
```

## Dispatch 모델 정리

| Level | 메커니즘 | 사용자 | 예시 |
|-------|----------|--------|------|
| **Op-level** | ComputeOps table | 모든 레이어 | sgemm, ele_add, quantize |
| **Graph-level** | ContextData subclass | NPU 레이어만 | QNN graph execution |

Op-level은 **모든 Context가 제공**해야 합니다 (최소 CPU fallback).
Graph-level은 **NPU 전용 레이어만** 사용합니다.

## 구현 우선순위

1. ContextData에 `as<T>()` 메서드 추가 (type-safe cast)
2. ContextData에 `getType()` 가상 메서드 추가 (디버깅용)
3. QNNContext::initialize()에서 ComputeOps 설정
4. Context base에서 ComputeOps 기본 설정 로직 추가

# Unified Tensor API Design — Symbolic + Lazy + External Memory

## 1. Motivation

### 현재 문제점

**1) `tensor_api.h`의 `ml::train::Tensor` — Var_Grad 상속 문제**
- API Tensor가 `nntrainer::Var_Grad`를 직접 상속 → gradient, optimizer 상태 등 내부 구현 노출
- 사용자가 "데이터 컨테이너"로서 tensor를 쓰고 싶을 뿐인데, gradient 관련 메서드가 전부 딸려옴
- 사실상 빈 껍데기 — 생성자만 있고 연산 메서드가 전혀 없음

**2) `functions.h`의 `ml::train::Tensor` — IR Graph용 별도 클래스**
- 데이터를 전혀 갖지 않는 순수 그래프 노드 (TensorNode wrapper)
- `tensor_api.h`의 Tensor와 이름 충돌 (같은 namespace에 두 개의 Tensor 클래스)
- 실제 연산은 없고 그래프 구조만 기록

**3) 내부 `nntrainer::Tensor`**
- 13개 이상 데이터 타입, NCHW/NHWC, 풍부한 연산 지원
- 하지만 내부 구현에 직접 의존해야 해서 API 안정성 보장 불가

**4) LazyTensor**
- 체이닝 방식의 지연 계산 (`tensor.chain().add_i(2).multiply_i(3).run()`)
- 좋은 컨셉이지만 API로 노출되지 않음

### 핵심 설계 원칙

1. **내부 Tensor와 분리** — Var_Grad 상속 제거, 내부 `nntrainer::Tensor`를 Pimpl로 감싸기
2. **Lazy by default** — 텐서 생성/연산 시 즉시 실행하지 않고 그래프 구축 가능
3. **Eager mode 지원** — 즉시 계산도 가능 (디버깅, 데이터 전처리용)
4. **createLayer와 연동** — Tensor를 layer input/output으로 자연스럽게 연결
5. **생성 방식이 내부 매핑을 결정** — 별도 등록 API 없이 Tensor 생성 방법이 UNIQUE/PLACEHOLDER를 자동 결정

```
Tensor(dim)           → 심볼릭 → compile 시 UNIQUE      → MemoryPool 할당
Tensor::fromData(ptr) → 외부   → compile 시 PLACEHOLDER → MemoryPool 미할당, 외부 포인터 직접 사용
```

---

## 2. 내부 아키텍처 요약

### 현재 내부 메모리 스케줄링 파이프라인

```
User addLayer() → compile() → topological sort + execution order 설정
  → initialize() → layer.finalize(InitLayerContext)
    → context.requestWeight/requestTensor (스펙만 등록)
      → Manager → TensorPool.request() (빈 Tensor*, data=null)
        → TensorPool.finalize(MemoryPlanner) → 메모리 레이아웃 계획
          → TensorPool.allocate() → 실제 메모리 할당 + setData()
            → LayerNode.configureRunContext() → RunLayerContext에 포인터 세팅
              → layer.forwarding(RunLayerContext) → context.getInput(idx)로 접근
```

### API Tensor ↔ 내부 매핑의 핵심 문제

API Tensor는 사용자가 compile **전에** 만든다. 하지만 실제 메모리는 compile → initialize → allocate **이후에야** 존재한다.

```
사용자 코드 시점          내부 시점
─────────────          ─────────
auto x = Tensor(dim)   → 아직 아무것도 없음 (shape만 존재)
auto fc = createLayer()
auto y = fc(x)         → graph edge 기록만 (TensorNode)
model.compile()        → graph 구축, execution order, TensorPool.request()
model.initialize()     → layer.finalize(), 실제 내부 Tensor 생성
model.allocate()       → 메모리 할당, data pointer 세팅
model.train()/infer()  → RunLayerContext 통해 접근
```

**해결**: Symbolic Tensor (API) → compile 시 → Materialized Tensor (Internal)로 바인딩

---

## 3. Tensor API 설계

### 3.1 클래스 구조

```
ml::train::Tensor (Public API)
├── impl_: shared_ptr<Impl>        ← Pimpl 패턴
│   ├── dim: TensorDim             ← 차원 정보 (항상 존재)
│   ├── name: string               ← 이름
│   ├── dtype: Tdatatype           ← 데이터 타입
│   ├── node: shared_ptr<TensorNode> ← 연산 그래프 추적
│   ├── requires_grad: bool
│   ├── eager_data: shared_ptr<nntrainer::Tensor>  ← 경로 1: eager 모드
│   ├── bound_internal: Var_Grad*  ← 경로 2: compile 후 바인딩
│   ├── is_external: bool          ← fromData 플래그
│   └── call_chain: vector<function> ← lazy chain
└── (Var_Grad 상속 없음)
```

### 3.2 API Header

```cpp
namespace ml {
namespace train {

class Tensor {
public:
  // ──── 심볼릭 생성 (lazy, 메모리 없음) ────
  // compile 시 UNIQUE → MemoryPool 할당
  Tensor(const TensorDim &dim, const std::string &name = "");
  Tensor(const TensorDim &dim, Tdatatype dtype);

  // ──── 즉시 생성 (eager, 데이터 있음) ────
  // 데이터 전처리, 추론 입력 데이터 세팅 등에 사용
  static Tensor fromData(const TensorDim &dim, float *data);
  static Tensor fromData(const TensorDim &dim, void *data, Tdatatype dtype);
  static Tensor zeros(const TensorDim &dim);
  static Tensor ones(const TensorDim &dim);

  // ──── 속성 (항상 접근 가능) ────
  const TensorDim &shape() const;
  Tdatatype dtype() const;
  std::string name() const;
  bool is_materialized() const;  // 실제 메모리가 할당되었는가?
  bool is_external() const;      // fromData로 생성되었는가?

  // ──── 그래프 속성 ────
  bool requires_grad() const;
  void set_requires_grad(bool flag);
  bool is_leaf() const;           // 사용자가 직접 만든 텐서인가?

  // ──── 데이터 접근 (materialized 상태에서만) ────
  // compile+allocate 후, 또는 fromData/zeros/ones로 만든 경우
  template<typename T> const T *data() const;
  template<typename T> T *mutable_data();
  float getValue(unsigned b, unsigned c, unsigned h, unsigned w) const;
  void setValue(unsigned b, unsigned c, unsigned h, unsigned w, float v);

  // ──── 데이터 주입 ────
  void copyFrom(const float *src);     // 내부 버퍼에 데이터 복사
  void setData(float *new_data);       // 외부 포인터 교체 (fromData 텐서만)
  void setData(void *new_data, Tdatatype dtype);

  // ──── 연산 (새 심볼릭 텐서 반환, 그래프 확장) ────
  Tensor add(const Tensor &rhs) const;
  Tensor matmul(const Tensor &rhs) const;
  Tensor transpose(const std::string &direction) const;
  Tensor reshape(const TensorDim &new_dim) const;
  Tensor operator+(const Tensor &rhs) const;
  Tensor operator-(const Tensor &rhs) const;
  Tensor operator*(const Tensor &rhs) const;

  // ──── Lazy 체이닝 (materialized 텐서에 대한 최적화) ────
  Tensor &chain();                // lazy 모드 시작
  Tensor &add_i(float value);    // in-place lazy op
  Tensor &multiply_i(float value);
  Tensor eval();                  // 체인 실행

  // ──── 내부 변환 (framework 내부용, 사용자 호출 안함) ────
  void _bind(nntrainer::Var_Grad *internal);
  nntrainer::Var_Grad *_internal() const;

private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

} // namespace train
} // namespace ml
```

### 3.3 Impl 내부 상태

```cpp
struct Tensor::Impl {
  TensorDim dim;                              // 차원 (항상 존재)
  std::string name;                           // 이름
  Tdatatype dtype;                            // 데이터 타입

  // 그래프 추적 (compile 전)
  std::shared_ptr<TensorNode> node;           // 연산 그래프 노드
  bool requires_grad = false;

  // 실체화된 데이터 (두 가지 경로)
  // 경로 1: eager 모드 (fromData, zeros 등)
  std::shared_ptr<nntrainer::Tensor> eager_data;

  // 경로 2: compile 후 바인딩
  nntrainer::Var_Grad *bound_internal = nullptr;

  // 외부 데이터 플래그
  bool is_external = false;

  // Lazy chain
  std::vector<std::function<int(nntrainer::Tensor &)>> call_chain;

  bool is_materialized() const {
    return eager_data != nullptr ||
           (bound_internal != nullptr && !bound_internal->getVariable().empty());
  }
};
```

### 3.4 `setExternalData()` 제거 결정

기존 설계에 있던 `setExternalData(void *new_data)`는 **제거**하고 `setData()`로 통합.

**이유:**
- `setExternalData()`는 `fromData()` 텐서에 대한 thin wrapper에 불과
- `setData()`가 동일 기능을 제공하되 더 직관적인 이름
- KV cache swap이 당장 구현 범위에 없으므로, `fromData()` + `is_external()` + `setData()`만으로 충분
- 나중에 필요하면 `setData()`에 validation만 추가

```cpp
// setData()가 setExternalData()를 대체
void Tensor::setData(float *new_data) {
  NNTR_THROW_IF(!impl_->is_external, std::invalid_argument)
    << "setData(ptr) is only valid for fromData() tensors";
  // MemoryData를 외부 포인터로 래핑
  auto mem = std::make_shared<MemoryData>(new_data, impl_->dim.getDataLen());
  if (impl_->bound_internal) {
    impl_->bound_internal->getVariable().setData(mem, 0);
    // TensorPool에 등록된 경우 syncDependents() 필요
  } else {
    impl_->eager_data->setData(mem, 0);
  }
}
```

---

## 4. Layer 연동 메커니즘

### 4.1 Layer::operator() — 그래프 구축

```cpp
// layer.h에 추가
class Layer {
public:
  // 심볼릭 텐서를 받아 심볼릭 출력 텐서 반환
  // 내부적으로 그래프 edge만 기록
  Tensor operator()(const Tensor &input);
  std::vector<Tensor> operator()(const std::vector<Tensor> &inputs);
};
```

```cpp
// 구현
Tensor Layer::operator()(const Tensor &input) {
  // 1) 출력 차원 계산 (layer property 기반)
  TensorDim out_dim = this->computeOutputDim(input.shape());

  // 2) 심볼릭 출력 텐서 생성
  Tensor output(out_dim, this->getName() + "/out");

  // 3) 그래프 엣지 기록: input → [this layer] → output
  auto func = std::make_shared<LayerFunction>(
    this->shared_from_this(), input, output);
  output.impl_->node = std::make_shared<TensorNode>(func);
  output.impl_->node->is_leaf = false;

  return output;
}
```

### 4.2 텐서 연산 → 암묵적 레이어 생성

```cpp
// Tensor::add() 는 내부적으로 Addition 레이어를 자동 생성
Tensor Tensor::add(const Tensor &rhs) const {
  auto add_layer = createLayer("addition", {});
  return add_layer({*this, rhs})[0];  // 심볼릭 Addition layer
}

// 사용:
auto h2 = createLayer("fully_connected", {"unit=256"})(h);
auto h3 = h.add(h2);  // skip connection (심볼릭 Add layer 자동 생성)
```

---

## 5. Model에서의 매핑 과정

### 5.1 compile() 내부 동작

```cpp
void Model::compile(const Tensor &graph_input, const Tensor &graph_output) {
  // 1) 심볼릭 그래프를 순회하여 LayerNode 추출
  //    y.node->creator → fc2, fc2의 input → a
  //    a.node->creator → relu, relu의 input → h
  //    h.node->creator → fc1, fc1의 input → input (leaf)
  auto layers = traceGraph(graph_output, graph_input);

  // 2) 추출된 layer들을 기존 addLayer() 방식으로 등록
  for (auto &[layer, connections] : layers) {
    this->addLayer(layer);
    // input_layers 속성 자동 설정
  }

  // 3) 기존 NeuralNetwork::compile() 호출
  //    → GraphRealizer → topologicalSort → setExecutionOrder
  NeuralNetwork::compile(exec_mode);

  // 4) initialize()
  //    → finalizeContext() → Manager::requestTensors()
  //    → TensorPool::request() (빈 Tensor 생성)
  NeuralNetwork::initialize(exec_mode);

  // 5) API Tensor ↔ 내부 Var_Grad 바인딩
  //    심볼릭 텐서 이름으로 내부 텐서를 찾아 연결
  for (auto &[api_tensor, layer_name] : tensor_mapping) {
    auto layer_node = model_graph.getLayerNode(layer_name);
    auto *internal_vg = layer_node->getOutput(0);  // Var_Grad*
    api_tensor._bind(internal_vg);
    // 이제 api_tensor.data() → internal_vg->getVariable().getData()
  }

  // 6) allocate → TensorPool.finalize() + allocate()
  //    메모리 스케줄링 + 실제 할당
  model_graph.allocateTensors(exec_mode);
  // 이 시점부터 api_tensor.is_materialized() == true
}
```

### 5.2 매핑 관계 다이어그램

```
┌──────────────────────────────────────────────────────────────────┐
│  API Layer (사용자 영역)                                          │
│                                                                    │
│  Tensor("input")  ──fc1()──▶  Tensor("h")  ──relu()──▶  Tensor("y")│
│       │                          │                          │      │
│    [shape만]                  [shape만]                  [shape만]  │
│    [node: leaf]              [node: fc1→]              [node: fc2→]│
└────────┬─────────────────────────┬──────────────────────────┬──────┘
         │ compile() 시 매핑        │                          │
         ▼                         ▼                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  Internal Layer (프레임워크 영역)                                  │
│                                                                    │
│  LayerNode     LayerNode       LayerNode       LayerNode          │
│  "input_layer" "fc1"           "relu"          "fc2"              │
│       │           │               │               │               │
│       ▼           ▼               ▼               ▼               │
│  Var_Grad*    Var_Grad*       Var_Grad*       Var_Grad*           │
│  (output)     (output)        (output)        (output)            │
│       │           │               │               │               │
│       ▼           ▼               ▼               ▼               │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  TensorPool                                              │     │
│  │  request("input", dim, exec_order=[0], FORWARD_INFER)    │     │
│  │  request("fc1:out", dim, exec_order=[1,5], ITERATION)    │     │
│  │  request("relu:out", dim, exec_order=[2,4], FORWARD_FUNC)│     │
│  │  request("fc2:out", dim, exec_order=[3], FORWARD_FUNC)   │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                           │
│       ▼ finalize(OptimizedV1Planner)                              │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  MemoryPool                                              │     │
│  │  [relu:out과 fc2:out이 겹치지 않으면 메모리 재사용]          │     │
│  │  총 메모리: input + fc1:out + max(relu:out, fc2:out)      │     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                           │
│       ▼ allocate()                                                │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  실제 메모리 블록                                          │     │
│  │  [████ input ████][████ fc1:out ████][████ shared ████]   │     │
│  │                                      ↑relu:out OR fc2:out│     │
│  └─────────────────────────────────────────────────────────┘     │
│       │                                                           │
│       ▼ _bind()                                                   │
│  API Tensor.data() → 내부 Tensor의 data pointer로 직접 접근       │
└──────────────────────────────────────────────────────────────────┘
```

### 5.3 Lifespan 자동 추론

사용자가 API Tensor에 lifespan을 직접 지정할 필요 없음. 그래프 구조에서 자동 추론:

```
1) model input tensor   → FORWARD_INFER_LIFESPAN (추론 내내 유지)
2) 중간 activation      → FORWARD_FUNC_LIFESPAN (forward 후 해제 가능)
3) training 시 backward 필요한 tensor → ITERATION_LIFESPAN
4) weight tensor        → MAX_LIFESPAN
5) 사용자가 결과를 읽을 output        → FORWARD_INFER_LIFESPAN
```

---

## 6. 외부 메모리 관리 (KV Cache)

### 6.1 설계 결정: 레이어 입력으로 처리 (setExternalTensor 제거)

`model->setExternalTensor("mha/cache_key", ...)` 방식의 문제:
- **내부 이름 노출** — 사용자가 `"mha/cache_key"` 같은 내부 텐서명을 알아야 함
- **그래프와 분리** — 그래프 바깥에서 내부 텐서를 건드리는 backdoor
- **fragile** — 내부 이름이 바뀌면 사용자 코드가 깨짐
- **선언적이지 않음** — 데이터 흐름이 그래프에 보이지 않음

**대신, fromData 텐서를 레이어 입력으로 직접 전달:**

```cpp
auto input = Tensor({1, seq_len, embed_dim});               // 심볼릭 → UNIQUE
auto key_cache = Tensor::fromData(cache_dim, my_key_buf);    // 외부 → PLACEHOLDER
auto val_cache = Tensor::fromData(cache_dim, my_val_buf);    // 외부 → PLACEHOLDER

auto out = mha(input, key_cache, val_cache);
model->compile(input, out);
// key_cache는 fromData이므로 자동으로 PLACEHOLDER
// 별도 setExternalTensor() 호출 불필요
```

그래프 상의 모습:
```
input(UNIQUE) ──────────┐
key_cache(PLACEHOLDER) ──┼──▶ MHA ──▶ output(UNIQUE)
val_cache(PLACEHOLDER) ──┘
```

### 6.2 대안 비교

| 항목 | setExternalTensor (제거) | fromData + 레이어 입력 (채택) |
|------|------------------------|------------------------------|
| 내부 이름 노출 | `"mha/cache_key"` 필요 | 불필요 |
| 그래프 가시성 | 안 보임 | 보임 |
| 별도 API | 필요 | 불필요 (Tensor 생성 방식이 결정) |
| 메모리 매핑 | 수동 등록 | 자동 (fromData → PLACEHOLDER) |
| layer 코드 변경 | 불필요 | 입력 수 확장 필요 (최소) |
| 업계 표준 | 비표준 | PyTorch/ONNX와 동일 |

### 6.3 내부 매핑 흐름

```
fromData 텐서:
  → compile 시 is_external 플래그 확인
  → TensorPool::placeholder(name, dim)
  → SourceDetails{token=0, lifespan=UNMANAGED, exec_order={}}
  → MemoryPool 할당 안 함
  → fillPlaceholder(name, external_tensor)
  → spec.tensor->setData(external_ptr)
  → syncDependents(spec)  // view 텐서들 자동 갱신

일반 심볼릭 텐서:
  → TensorPool::request(name, dim, exec_order, lifespan)
  → 기존 UNIQUE 경로 그대로
```

### 6.4 데이터 포인터 교체 (추론 중 cache swap)

```cpp
// 다른 요청의 cache로 교체할 때
key_cache.setData(other_request_key_buf);
val_cache.setData(other_request_val_buf);
// → 내부적으로 fillPlaceholder 재호출 + syncDependents
```

**주의**: 외부 메모리의 lifetime은 사용자 책임. API 문서에 명시 필요.

---

## 7. MHA Layer 외부 캐시 지원

### 7.1 INOUT_INDEX 확장

```cpp
enum INOUT_INDEX {
  QUERY = 0,
  KEY = 1,
  VALUE = 2,
  MASK = 3,
  CACHE_KEY = 4,     // 새로 추가
  CACHE_VALUE = 5,   // 새로 추가
  OUTPUT = 0,
  RETURN_ATTENTION_WEIGHT = 1,
};
```

### 7.2 finalize() 변경

```cpp
void MultiHeadAttentionLayer::finalize(InitLayerContext &context) {
  auto num_inputs = context.getNumInputs();

  // 기존: 3-4개 입력 (query, key, value, [mask])
  // 변경: 3-6개 입력 지원
  //   3: query, key, value
  //   4: query, key, value, mask
  //   5: query, key, value, cache_key, cache_value
  //   6: query, key, value, mask, cache_key, cache_value
  NNTR_THROW_IF(num_inputs < 3 || num_inputs > 6, std::invalid_argument)
    << "MultiHeadAttention needs 3 to 6 inputs";

  const bool provide_attention_mask = (num_inputs == 4 || num_inputs == 6);

  if (num_inputs >= 5) {
    use_external_cache = true;
    // 외부 캐시가 입력으로 제공됨 → requestTensor() 불필요
  } else {
    use_external_cache = false;
    // 기존 방식: 내부 requestTensor()로 cache 할당
    weight_idx[AttentionParams::cache_key] = context.requestTensor(
      projected_key_dim, "cache_key", Initializer::NONE,
      true, TensorLifespan::MAX_LIFESPAN);
    weight_idx[AttentionParams::cache_value] = context.requestTensor(
      projected_value_dim, "cache_value", Initializer::NONE,
      true, TensorLifespan::MAX_LIFESPAN);
  }
  // 나머지 (projected_query, attention_weight 등) 동일
}
```

### 7.3 forwarding()/incremental_forwarding() 변경

```cpp
void MultiHeadAttentionLayer::incremental_forwarding(
    RunLayerContext &context, unsigned int from, unsigned int to,
    bool training) {
  // 캐시 텐서 접근 분기만 변경
  Tensor &cache_key = use_external_cache
    ? context.getInput(INOUT_INDEX::CACHE_KEY)
    : context.getTensor(weight_idx[AttentionParams::cache_key]);
  Tensor &cache_value = use_external_cache
    ? context.getInput(INOUT_INDEX::CACHE_VALUE)
    : context.getTensor(weight_idx[AttentionParams::cache_value]);

  // 이후 cache_key_step, cached_key 등 getSharedDataTensor 패턴은 100% 동일
}
```

### 7.4 하위 호환성 보장

| 시나리오 | num_inputs | use_external_cache | 동작 |
|----------|-----------|-------------------|------|
| 기존 MHA (query, key, value) | 3 | false | 내부 cache 할당 (기존과 동일) |
| 기존 MHA + mask | 4 | false | 내부 cache 할당 + mask (기존과 동일) |
| 새 API (q, k, v, ext_cache_k, ext_cache_v) | 5 | true | 외부 cache 사용 |
| 새 API + mask | 6 | true | 외부 cache + mask |

**기존 코드는 수정 없이 동작함.**

### 7.5 MHACoreLayer (CausalLM)도 동일 패턴

```cpp
// mha_core.cpp finalize()
void MHACoreLayer::finalize(InitLayerContext &context) {
  auto num_inputs = context.getNumInputs();

  // 기존: 3개 (query, key, value)
  // 변경: 3-5개 (query, key, value, [cache_key], [cache_value])
  if (num_inputs >= 4) {
    use_external_cache = true;
  } else {
    // 기존 방식
    tensor_idx[AttentionParams::cache_key] = context.requestTensor(
      cache_key_dim, "cache_key", Initializer::NONE,
      false, TensorLifespan::MAX_LIFESPAN);
    tensor_idx[AttentionParams::cache_value] = context.requestTensor(
      cache_value_dim, "cache_value", Initializer::NONE,
      false, TensorLifespan::MAX_LIFESPAN);
  }
}
```

---

## 8. Lazy Chaining

### 8.1 내부 LazyTensor 연동

```cpp
Tensor &Tensor::chain() {
  impl_->call_chain.clear();
  return *this;
}

Tensor &Tensor::add_i(float value) {
  impl_->call_chain.push_back([value](nntrainer::Tensor &t) {
    return t.add_i(value);
  });
  return *this;
}

Tensor &Tensor::multiply_i(float value) {
  impl_->call_chain.push_back([value](nntrainer::Tensor &t) {
    return t.multiply_i(value);
  });
  return *this;
}

Tensor Tensor::eval() {
  NNTR_THROW_IF(!is_materialized(), std::runtime_error)
    << "Cannot eval() on non-materialized tensor";
  auto &internal_tensor = impl_->bound_internal
    ? impl_->bound_internal->getVariable()
    : *impl_->eager_data;
  for (auto &op : impl_->call_chain) {
    op(internal_tensor);
  }
  impl_->call_chain.clear();
  return *this;
}
```

### 8.2 사용 예시

```cpp
// 데이터 전처리 (eager)
auto raw = Tensor::fromData({1, 3, 32, 32}, image_ptr);
auto normalized = raw.chain()
  .multiply_i(1.0f / 255.0f)
  .add_i(-0.5f)
  .eval();
```

---

## 9. 전체 사용 시나리오

### 9.1 심볼릭 그래프로 네트워크 구성

```cpp
using namespace ml::train;

auto x = Tensor({1, 784}, "input");
auto h = createLayer("fully_connected", {"unit=256"})(x);
h = createLayer("activation", {"activation=relu"})(h);
h = createLayer("dropout", {"dropout=0.5"})(h);

// 텐서 연산으로 residual connection
auto h2 = createLayer("fully_connected", {"unit=256"})(h);
auto h3 = h.add(h2);  // skip connection (심볼릭 Add layer 자동 생성)

auto y = createLayer("fully_connected", {"unit=10"})(h3);

auto model = createModel(ModelType::NEURAL_NET,
                          {"batch_size=32", "epochs=10"});
model->compile(x, y);   // 심볼릭 → 내부 그래프 + 메모리 스케줄링
model->initialize();
```

### 9.2 KV Cache를 사용한 추론

```cpp
using namespace ml::train;

// 1. 모델 구성
auto tokens = Tensor({1, 1, 1}, "tokens");
auto embed = createLayer("embedding", {"dim=4096"});

// 2. 외부 KV cache
float *key_buf = allocate_on_device(batch * max_seq * num_heads * head_dim);
float *val_buf = allocate_on_device(batch * max_seq * num_heads * head_dim);
auto key_cache = Tensor::fromData({1, 1, max_seq, num_heads * head_dim}, key_buf);
auto val_cache = Tensor::fromData({1, 1, max_seq, num_heads * head_dim}, val_buf);

auto h = embed(tokens);
auto attn_out = mha(h, key_cache, val_cache);  // cache는 그래프의 일부
auto out = ffn(attn_out);

auto model = createModel(ModelType::NEURAL_NET);
model->compile(tokens, out);
// tokens → UNIQUE, key_cache/val_cache → PLACEHOLDER (자동)

// 3. 추론 루프
for (int step = 0; step < generate_len; step++) {
  tokens.setValue(0, 0, 0, 0, next_token_id);

  // cache는 이미 바인딩되어 있으므로 별도 세팅 불필요
  // MHA가 cache에 write → 외부 버퍼에 직접 반영 (zero-copy)
  auto result = model->incremental_inference(step, step + 1);
  int next_token = argmax(out.data<float>());
}

// 4. 다른 요청의 cache로 교체 시
key_cache.setData(other_request_key_buf);
val_cache.setData(other_request_val_buf);
// → 내부적으로 fillPlaceholder 재호출 + syncDependents
```

### 9.3 추론 입력 데이터 흐름

```cpp
// compile 후 바인딩된 텐서에 데이터 주입
auto input = Tensor({1, 3, 224, 224}, "input");
// ... model 구성 + compile ...

// 방법 1: 이미 할당된 내부 텐서에 데이터 복사
float *image_data = load_image("cat.jpg");
input.copyFrom(image_data);
// → 내부: bound_internal->getVariable().copy(external_buffer)

// 방법 2: 외부 버퍼를 직접 매핑 (zero-copy, fromData 텐서만)
input.setData(image_data);
// → 내부: bound_internal->getVariable().setData(MemoryData(ptr))
```

---

## 10. Applications 마이그레이션

### 10.1 마이그레이션 전략

| 패턴 | 대상 | 변경 필요 여부 |
|------|------|---------------|
| **A. INI 기반** | MNIST, AlexNet, VGG, Resnet 등 | 변경 불필요 (Tensor API 미사용) |
| **B. createLayer API** | PicoGPT, LLaMA, Layers 등 | MHA 외부 캐시 옵트인 시만 변경 |
| **C. KV cache 직접 관리** | CausalLM | 가장 큰 변경 필요 |

### 10.2 CausalLM 마이그레이션 (가장 큰 변경)

```cpp
class CausalLM {
private:
  struct KVCacheBuffers {
    std::vector<float *> key_bufs;
    std::vector<float *> val_bufs;
  };
  KVCacheBuffers kv_cache;
  std::vector<nntrainer::Tensor> key_cache_tensors;
  std::vector<nntrainer::Tensor> val_cache_tensors;
};

void CausalLM::constructModel() {
  for (int i = 0; i < NUM_LAYERS; i++) {
    kv_cache.key_bufs.push_back(
      new float[BATCH * MAX_SEQ * NUM_KV_HEADS * HEAD_DIM]);
    kv_cache.val_bufs.push_back(
      new float[BATCH * MAX_SEQ * NUM_KV_HEADS * HEAD_DIM]);

    TensorDim cache_dim = {BATCH, 1, MAX_SEQ, NUM_KV_HEADS * HEAD_DIM};
    key_cache_tensors.push_back(
      Tensor::fromData(cache_dim, kv_cache.key_bufs[i],
                       "layer" + std::to_string(i) + "_ext_k_cache"));
    val_cache_tensors.push_back(
      Tensor::fromData(cache_dim, kv_cache.val_bufs[i],
                       "layer" + std::to_string(i) + "_ext_v_cache"));

    // MHA core에 5개 입력으로 전달
    model->addLayer(createLayer("mha_core", {
      "name=layer" + std::to_string(i) + "_attention",
      "input_layers=" + q_name + "," + k_name + "," + v_name + ","
        + key_cache_tensors.back().getName() + ","
        + val_cache_tensors.back().getName(),
      "num_heads=" + std::to_string(NUM_HEADS),
    }));
  }
}

// save/load 간소화 — 외부 버퍼에 직접 접근
void CausalLM::save_kvcache(std::string path, int to) {
  auto f = checkedOpenStream<std::ofstream>(path, std::ios::binary);
  for (int i = 0; i < NUM_LAYERS; i++) {
    size_t bytes = BATCH * to * NUM_KV_HEADS * HEAD_DIM * sizeof(float);
    f.write(reinterpret_cast<char*>(kv_cache.key_bufs[i]), bytes);
    f.write(reinterpret_cast<char*>(kv_cache.val_bufs[i]), bytes);
  }
}
```

### 10.3 PicoGPT / LLaMA — 변경 없음 (하위 호환)

기존 3-4개 입력 MHA 그대로 동작. 외부 cache 전환은 선택적.

### 10.4 마이그레이션 우선순위

| 순서 | 대상 | 변경 범위 | 이유 |
|------|------|-----------|------|
| 1 | CausalLM | 큼 | KV cache 직접 관리, 가장 큰 수혜자 |
| 2 | PicoGPT | 작음 | 내장 MHA 사용, 선택적 전환 |
| 3 | LLaMA | 작음 | 커스텀 어텐션, 선택적 전환 |
| 4 | 나머지 | 없음 | INI 기반 또는 Tensor API 미사용 |

---

## 11. TorchFXConverter 업데이트

### 11.1 변경 이유

TorchFXConverter가 생성하는 C++ 코드에서 외부 KV cache를 선택적으로 지원:

```bash
python converter.py --model Qwen/Qwen3-0.6B --external-kv-cache
```

### 11.2 변경 파일

| 파일 | 변경 내용 |
|------|-----------|
| `converter.py` | `--external-kv-cache` CLI 옵션 추가 |
| `emitter_cpp/header.py` | 외부 cache 멤버 변수 선언 |
| `emitter_cpp/source_construct.py` | `allocateKVCache()`, cache 텐서 생성 |
| `emitter_cpp/source_attention.py` | mha_core 5-input 모드 생성 |
| `emitter_ini/` | 변경 없음 |
| `weight_converter.py` | 변경 없음 |
| `patterns/` | 변경 없음 |

### 11.3 생성 코드 비교

```cpp
// 기존 (내부 cache) — mha_core에 3개 입력
layers.push_back(createLayer("mha_core", {
  withKey("name", A),
  withKey("num_heads", n_heads),
  withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
  withKey("input_layers", Q_norm + "," + K_norm + "," + V)
}));

// 새 API (--external-kv-cache) — mha_core에 5개 입력
layers.push_back(createLayer("mha_core", {
  withKey("name", A),
  withKey("num_heads", n_heads),
  withKey("input_layers", Q_norm + "," + K_norm + "," + V
          + "," + cache_key_name + "," + cache_value_name)
}));
```

---

## 12. 전체 변경 파일 요약

### Core (Phase 1)
| 파일 | 변경 |
|------|------|
| `nntrainer/tensor/tensor.h` | `fromData()`, `isExternal()`, `isMaterialized()`, `external_` 멤버 |
| `nntrainer/tensor/tensor.cpp` | 위 메서드 구현 |
| `nntrainer/tensor/tensor_pool.h` | `requestOrPlaceholder()` |
| `nntrainer/tensor/tensor_pool.cpp` | `requestOrPlaceholder()` 구현 |
| `api/ccapi/include/tensor_api.h` | Pimpl 기반 재구현, `fromData()`, `isExternal()`, 연산 메서드 |

### Layers (Phase 2)
| 파일 | 변경 |
|------|------|
| `nntrainer/layers/multi_head_attention_layer.h` | `use_external_cache` 멤버 |
| `nntrainer/layers/multi_head_attention_layer.cpp` | `finalize()`, `incremental_forwarding()`, `forwarding()` 분기 |
| `nntrainer/graph/network_graph.cpp` | `finalizeContext()` 외부 텐서 PLACEHOLDER 처리 |

### Applications (Phase 3)
| 파일 | 변경 |
|------|------|
| `Applications/CausalLM/causal_lm.h` | 외부 cache 버퍼/텐서 멤버 |
| `Applications/CausalLM/causal_lm.cpp` | `constructModel()`, `save/load_kvcache()` |
| `Applications/CausalLM/layers/mha_core.h` | `use_external_cache` 멤버 |
| `Applications/CausalLM/layers/mha_core.cpp` | `finalize()`, `incremental_forwarding()` 분기 |
| `Applications/PicoGPT/jni/main.cpp` | 변경 없음 (하위 호환) |
| `Applications/LLaMA/jni/main.cpp` | 변경 없음 (하위 호환) |

### TorchFXConverter (Phase 4)
| 파일 | 변경 |
|------|------|
| `Applications/TorchFXConverter/converter.py` | `--external-kv-cache` 옵션 |
| `Applications/TorchFXConverter/emitter_cpp/header.py` | 외부 cache 멤버 선언 |
| `Applications/TorchFXConverter/emitter_cpp/source_construct.py` | cache 할당/텐서 생성 코드 |
| `Applications/TorchFXConverter/emitter_cpp/source_attention.py` | 5-input mha_core 생성 |

---

## 13. 구현 순서

```
Phase 1: Core Tensor API (Pimpl 기반 재구현)
  ├── tensor_api.h: Var_Grad 상속 제거, Pimpl 패턴 도입
  ├── tensor.h/cpp: fromData(), isExternal(), isMaterialized()
  ├── tensor_pool: requestOrPlaceholder()
  └── Tensor 연산 메서드 (add, matmul, reshape 등)

Phase 2: Layer 연동
  ├── Layer::operator()(Tensor) — 심볼릭 그래프 구축
  ├── Model::compile(input, output) — 그래프 추출 + 내부 매핑
  └── Lazy chaining (chain/eval)

Phase 3: MHA 외부 캐시 지원
  ├── multi_head_attention_layer: finalize/forwarding 분기
  ├── mha_core (CausalLM): 동일 패턴
  └── network_graph: finalizeContext PLACEHOLDER 처리

Phase 4: Applications 마이그레이션
  ├── CausalLM: 외부 cache 버퍼 관리로 전환
  ├── PicoGPT: 하위 호환 확인 (변경 없음)
  └── LLaMA: 하위 호환 확인 (변경 없음)

Phase 5: TorchFXConverter
  ├── converter.py: --external-kv-cache 옵션
  ├── source_attention.py: 5-input mha_core
  ├── source_construct.py: cache 할당 코드
  └── header.py: 외부 cache 멤버 선언
```

---

## 14. 핵심 설계 결정 요약

| 항목 | 결정 |
|------|------|
| API Tensor 기반 | Pimpl (Var_Grad 상속 제거) |
| 메모리 매핑 | Tensor 생성 방식이 자동 결정 (UNIQUE vs PLACEHOLDER) |
| 외부 텐서 등록 | setExternalTensor 없음 — fromData 텐서를 레이어 입력으로 직접 전달 |
| 외부 포인터 교체 | `setData()` (setExternalData 대신 통합 API) |
| Layer 연동 | `Layer::operator()(Tensor)` — 심볼릭 그래프 edge 기록 |
| 텐서 연산 | `Tensor::add()` 등 → 암묵적 레이어 자동 생성 |
| Lifespan | 그래프 위치 + 학습/추론 모드에서 자동 추론 |
| Eager 텐서 | `fromData/zeros/ones` → `nntrainer::Tensor` 직접 생성 (Pool 미사용) |
| Lazy chain | `chain().op().eval()` → 내부 LazyTensor 연동 |
| KV cache | 레이어 입력으로 처리 (PyTorch/ONNX 표준) |
| 하위 호환 | 기존 3-4 input MHA 코드 수정 없이 동작 |

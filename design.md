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

### 10.1 핵심 원칙: 내부 Layer API는 변경하지 않는다

**변경되는 것:** Public API (`ml::train::Tensor`, `Model::compile` 패턴)
**변경되지 않는 것:** 내부 Layer API (`InitLayerContext`, `RunLayerContext`, `nntrainer::Tensor`)

```
변경 O (Public API)                변경 X (Internal API)
─────────────────                  ─────────────────────
ml::train::Tensor (Pimpl 재구현)    nntrainer::Tensor (그대로)
ml::train::Model::compile()        InitLayerContext::requestWeight()
ml::train::Layer::operator()       InitLayerContext::requestTensor()
                                   RunLayerContext::getInput/getOutput()
                                   RunLayerContext::getTensor/getWeight()
                                   nntrainer::Tensor::getSharedDataTensor()
                                   nntrainer::Tensor::multiply_i/add_i()
```

커스텀 레이어 (`mha_core`, `rms_norm`, `embedding_layer`, `qwen_moe_layer` 등)의 `finalize()`, `forwarding()`, `incremental_forwarding()` 내부 로직은 **수정할 필요 없음**. MHA 외부 캐시 지원을 위한 `finalize()` 입력 개수 분기만 추가.

### 10.2 영향도 분석

| 카테고리 | Applications | 영향도 | 이유 |
|----------|-------------|--------|------|
| **INI 기반** | MNIST, VGG, Resnet, TransferLearning | 없음 | Tensor API 미사용, INI 설정 파일로 모델 구성 |
| **createLayer+addLayer** | SimpleFC, Resnet (C++ API) | 없음 | `createLayer()` + `model->addLayer()` + 문자열 속성 패턴 유지 |
| **createLayer+addLayer+커스텀 레이어** | PicoGPT, LLaMA | 없음~최소 | 기존 3-4 입력 MHA 하위 호환. 외부 cache 전환은 선택적 |
| **KV cache 직접 관리** | CausalLM | **큼** | `forEachLayer()` 콜백으로 내부 텐서 접근 → 외부 버퍼 직접 관리로 전환 |
| **커스텀 옵티마이저** | Custom/momentum.cpp | 없음 | `RunOptimizerContext` API 변경 없음 |

### 10.3 변경 없는 Applications (확인용)

#### SimpleFC (`Applications/SimpleFC/jni/main.cpp`)
```cpp
// 현재 코드 — 변경 불필요
model->setProperty({withKey("batch_size", batch_size), withKey("epochs", epochs)});
auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
model->setOptimizer(std::move(optimizer));
status = model->compile();       // ← 기존 오버로드 유지
status = model->initialize();    // ← 기존 오버로드 유지
```

#### Resnet (`Applications/Resnet/jni/main.cpp`)
```cpp
// 현재 코드 — 변경 불필요
using ml::train::createLayer;
auto resnetBlock = [](/* ... */) {
  return createLayer("conv2d", {withKey("filters", filters), ...});
};
for (auto &layer : layers) model->addLayer(layer);
model->compile();
```

#### PicoGPT (`Applications/PicoGPT/jni/main.cpp`)
```cpp
// 현재 코드 — 변경 불필요
// MHA에 3개 입력: query, key, value
layers.push_back(createLayer("multi_head_attention", {
  withKey("name", "attention" + std::to_string(i)),
  withKey("num_heads", NUM_HEADS),
  withKey("input_layers", qkv_name)  // ← 3 inputs, 하위 호환
}));
```

#### LLaMA (`Applications/LLaMA/jni/main.cpp`)
```cpp
// 현재 코드 — 변경 불필요 (커스텀 MHA 사용)
layers.push_back(createLayer("custom_multi_head_attention", {
  withKey("name", attn_name),
  withKey("input_layers", q_name + "," + k_name + "," + v_name)
}));
```

#### Custom Optimizer (`Applications/Custom/momentum.cpp`)
```cpp
// 현재 코드 — 변경 불필요
void Momentum::applyGradient(nntrainer::RunOptimizerContext &context) {
  nntrainer::Tensor &x_grad = context.getGradient();        // 내부 API
  nntrainer::Tensor &accumulated = context.getOptimizerVariable(0);
  accumulated.multiply_i(m);
  accumulated.add_i(x_grad);
  x_grad.fill(accumulated);
}
```

### 10.4 CausalLM 마이그레이션 (가장 큰 변경)

#### 현재 구현 분석

**모델 구성** (`causal_lm.cpp:174-233`):
```cpp
// 현재: createLayer + addLayer 패턴 (문자열 기반)
void CausalLM::constructModel() {
  std::vector<LayerHandle> layers;
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  layers.push_back(createLayer("input", {
    withKey("name", "input0"),
    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))
  }));
  layers.push_back(createLayer(embedding_type, {
    "name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB), ...
  }));

  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto transformer = createTransformerDecoderBlock(i, ...);
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  for (auto &layer : layers) model->addLayer(layer);
}
```

**KV cache save/load** (`causal_lm.cpp:764-821`) — **가장 큰 변경 포인트**:
```cpp
// 현재: forEachLayer 콜백으로 내부 RunLayerContext 접근
void CausalLM::save_kvcache(std::string path, int to_) {
  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)> fn =
    [&f](ml::train::Layer &l, nntrainer::RunLayerContext &context, void *idx) {
      if (l.getType() == causallm::MHACoreLayer::type) {
        auto k_cache = context.getTensor(0);    // ← 내부 인덱스로 직접 접근
        auto v_cache = context.getTensor(1);
        ml::train::TensorDim k_dim = k_cache.getDim();
        k_dim.height(to);
        nntrainer::Tensor k_cache_prompt = k_cache.getSharedDataTensor(k_dim, 0, true);
        k_cache_prompt.save(f);
      }
    };
  model->forEachLayer(fn, arg);
}
```

**Attention 레이어** (`causal_lm.cpp:565-670`):
```cpp
// 현재: mha_core에 3개 입력 (Q, K, V)
std::vector<LayerHandle> CausalLM::createAttention(const int layer_id, ...) {
  layers.push_back(createLayer("fully_connected", v_params));
  layers.push_back(createLayer("fully_connected", k_params));
  layers.push_back(createLayer("fully_connected", q_params));
  layers.push_back(createLayer("mha_core", {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("input_layers", Q_norm + "," + K_norm + "," + V)  // ← 3 inputs
  }));
  layers.push_back(createLayer("fully_connected", o_params));
  return layers;
}
```

**Compile/Initialize** (`causal_lm.cpp:140-172`):
```cpp
// 현재: ExecutionMode 지정
model->setProperty({withKey("batch_size", BATCH_SIZE), ...});
model->compile(ml::train::ExecutionMode::INFERENCE);
model->initialize(ml::train::ExecutionMode::INFERENCE);
```

**Incremental Inference** (`causal_lm.cpp:396-420`):
```cpp
// 현재: incremental_inference API
output = model->incremental_inference(BATCH_SIZE, input, label,
                                      input_len, from, to, false);
```

#### 변경 후 CausalLM

**핵심 변경: KV cache를 외부 버퍼로 관리, mha_core에 5개 입력**

```cpp
// ──── causal_lm.h 변경 ────
class CausalLM {
private:
  // 새로 추가: 외부 KV cache 버퍼
  struct KVCacheBuffers {
    std::vector<float *> key_bufs;   // 사용자가 할당한 메모리
    std::vector<float *> val_bufs;
  };
  KVCacheBuffers kv_cache;
  std::vector<ml::train::Tensor> key_cache_tensors;  // fromData 텐서
  std::vector<ml::train::Tensor> val_cache_tensors;

  // 기존 멤버 유지
  ModelHandle model;
  // ...
};
```

```cpp
// ──── causal_lm.cpp constructModel() 변경 ────
void CausalLM::constructModel() {
  std::vector<LayerHandle> layers;
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  // 입력, 임베딩 등은 기존과 동일
  layers.push_back(createLayer("input", { ... }));
  layers.push_back(createLayer(embedding_type, { ... }));

  // 변경: 각 레이어마다 외부 KV cache 버퍼 할당
  for (int i = 0; i < NUM_LAYERS; i++) {
    size_t cache_size = BATCH_SIZE * MAX_SEQ * NUM_KV_HEADS * HEAD_DIM;
    kv_cache.key_bufs.push_back(new float[cache_size]());
    kv_cache.val_bufs.push_back(new float[cache_size]());

    ml::train::TensorDim cache_dim(
      {BATCH_SIZE, 1, MAX_SEQ, NUM_KV_HEADS * HEAD_DIM});
    key_cache_tensors.push_back(
      ml::train::Tensor::fromData(cache_dim, kv_cache.key_bufs[i]));
    val_cache_tensors.push_back(
      ml::train::Tensor::fromData(cache_dim, kv_cache.val_bufs[i]));
  }

  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto transformer = createTransformerDecoderBlock(i, ...);
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  for (auto &layer : layers) model->addLayer(layer);
}
```

```cpp
// ──── createAttention() 변경 ────
std::vector<LayerHandle> CausalLM::createAttention(const int layer_id, ...) {
  // V, K, Q projections — 동일
  layers.push_back(createLayer("fully_connected", v_params));
  layers.push_back(createLayer("fully_connected", k_params));
  layers.push_back(createLayer("fully_connected", q_params));

  // 변경: mha_core에 5개 입력 (Q, K, V, cache_key, cache_value)
  auto cache_k_name = key_cache_tensors[layer_id].name();
  auto cache_v_name = val_cache_tensors[layer_id].name();

  layers.push_back(createLayer("mha_core", {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads_kv),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("input_layers", Q_norm + "," + K_norm + "," + V
            + "," + cache_k_name + "," + cache_v_name)  // ← 5 inputs
  }));

  layers.push_back(createLayer("fully_connected", o_params));
  return layers;
}
```

```cpp
// ──── save/load_kvcache() 대폭 간소화 ────

// 현재: forEachLayer + RunLayerContext 콜백 (30줄+)
// 변경 후: 외부 버퍼 직접 접근 (10줄)

void CausalLM::save_kvcache(std::string path, int to) {
  auto f = checkedOpenStream<std::ofstream>(path, std::ios::binary);
  for (int i = 0; i < NUM_LAYERS; i++) {
    size_t bytes = BATCH_SIZE * to * NUM_KV_HEADS * HEAD_DIM * sizeof(float);
    f.write(reinterpret_cast<char*>(kv_cache.key_bufs[i]), bytes);
    f.write(reinterpret_cast<char*>(kv_cache.val_bufs[i]), bytes);
  }
}

void CausalLM::load_kvcache(std::string path, int to) {
  auto f = checkedOpenStream<std::ifstream>(path, std::ios::binary);
  for (int i = 0; i < NUM_LAYERS; i++) {
    size_t bytes = BATCH_SIZE * to * NUM_KV_HEADS * HEAD_DIM * sizeof(float);
    f.read(reinterpret_cast<char*>(kv_cache.key_bufs[i]), bytes);
    f.read(reinterpret_cast<char*>(kv_cache.val_bufs[i]), bytes);
  }
  // 내부 fillPlaceholder가 이미 연결되어 있으므로
  // model->allocate() 재호출 불필요
}
```

```cpp
// ──── compile/initialize — 기존 오버로드 유지 ────
// model->compile(ExecutionMode::INFERENCE) 와
// model->compile(input_tensor, output_tensor) 를 둘 다 지원
// CausalLM은 기존 addLayer 패턴이므로 기존 compile 사용 가능
model->compile(ml::train::ExecutionMode::INFERENCE);
model->initialize(ml::train::ExecutionMode::INFERENCE);
```

#### MHACoreLayer 변경 (`layers/mha_core.cpp`)

```cpp
// mha_core.cpp finalize() — 입력 수에 따른 분기 추가
void MHACoreLayer::finalize(nntrainer::InitLayerContext &context) {
  auto num_inputs = context.getNumInputs();

  // 기존: 3개 (query, key, value)
  // 변경: 3-5개 (query, key, value, [cache_key], [cache_value])
  if (num_inputs >= 4) {
    use_external_cache = true;
    // 외부 입력이 cache 역할 → requestTensor() 호출 안 함
  } else {
    use_external_cache = false;
    // 기존: 내부 requestTensor()
    ml::train::TensorDim cache_key_dim(
      {batch_size, 1, max_timestep, num_heads_KV * head_dim},
      {context.getFormat(), ml::train::TensorDim::DataType::FP16});
    tensor_idx[AttentionParams::cache_key] = context.requestTensor(
      cache_key_dim, "cache_key", nntrainer::Tensor::Initializer::NONE,
      false, TensorLifespan::MAX_LIFESPAN);
    tensor_idx[AttentionParams::cache_value] = context.requestTensor(
      cache_value_dim, "cache_value", nntrainer::Tensor::Initializer::NONE,
      false, TensorLifespan::MAX_LIFESPAN);
  }

  // 나머지 (projected_query, attention_weight 등) 동일
}
```

```cpp
// mha_core.cpp incremental_forwarding() — 캐시 접근 분기만 추가
void MHACoreLayer::incremental_forwarding(
    nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
    bool training) {
  // 캐시 텐서 참조 — 외부/내부 분기
  nntrainer::Tensor &cache_key = use_external_cache
    ? context.getInput(3)   // 4번째 입력 = 외부 cache_key
    : context.getTensor(tensor_idx[AttentionParams::cache_key]);
  nntrainer::Tensor &cache_value = use_external_cache
    ? context.getInput(4)   // 5번째 입력 = 외부 cache_value
    : context.getTensor(tensor_idx[AttentionParams::cache_value]);

  // 이후 로직 100% 동일:
  // cache_key_step = cache_key.getSharedDataTensor(step_dim, offset);
  // ...
}
```

### 10.5 하위 호환성 보장 요약

| Application | compile() 패턴 | 변경 | 이유 |
|-------------|----------------|------|------|
| CausalLM | `compile(ExecutionMode)` | 유지 | addLayer 패턴 그대로, 새 compile(input, output) 추가 |
| PicoGPT | `compile(ExecutionMode)` | 없음 | 3-input MHA 그대로 동작 |
| LLaMA | `compile(ExecutionMode)` | 없음 | 커스텀 MHA, 내부 cache |
| Resnet | `compile()` | 없음 | 기본 오버로드 유지 |
| SimpleFC | `compile()` | 없음 | 기본 오버로드 유지 |
| MNIST | INI 파일 | 없음 | API 미사용 |
| VGG | INI 파일 | 없음 | API 미사용 |
| test/ccapi | 둘 다 | 없음 | 기존 오버로드 유지 |

**`compile()` 오버로드 전략:**
```cpp
class Model {
  // 기존 (유지) — addLayer 패턴용
  int compile(ExecutionMode mode = ExecutionMode::TRAIN);
  int initialize(ExecutionMode mode = ExecutionMode::TRAIN);

  // 새로 추가 — 심볼릭 Tensor 그래프용
  int compile(const Tensor &input, const Tensor &output,
              ExecutionMode mode = ExecutionMode::TRAIN);
};
```

---

## 11. TorchFXConverter 업데이트

### 11.1 현재 생성 패턴 분석

TorchFXConverter는 HuggingFace 모델을 NNTrainer C++ 코드로 변환한다. 생성되는 코드의 핵심 패턴:

**헤더 (`emitter_cpp/header.py` 생성):**
```cpp
class Qwen3CausalLM {
public:
  void constructModel();
  void initialize();
  ModelHandle &getModel() { return model; }

protected:
  std::vector<LayerHandle> createTransformerDecoderBlock(
    const int layer_id, std::string input_name);
  std::vector<LayerHandle> createAttention(
    const int layer_id, int seq_len, int n_heads, int head_dim,
    std::string query_name, std::string key_name, std::string value_name);
  std::vector<LayerHandle> createMlp(
    const int layer_id, int dim, int hidden_dim, std::string input_name);
  void registerCustomLayers();

  ModelHandle model;

  // Model constants (HF config에서 추출)
  unsigned int NUM_VOCAB = 1000;
  int DIM = 64;
  int NUM_LAYERS = 2;
  // ...
};
```

**소스 — constructModel() (`emitter_cpp/source_construct.py` 생성):**
```cpp
void Qwen3CausalLM::constructModel() {
  std::vector<LayerHandle> layers;
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  layers.push_back(createLayer("input", {
    withKey("name", "input0"),
    withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))
  }));
  layers.push_back(createLayer("embedding_layer", { ... }));

  for (int i = 0; i < NUM_LAYERS; ++i) {
    auto block = createTransformerDecoderBlock(i, input_name);
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // 최종 norm + LM head
  layers.push_back(createLayer("rms_norm", { ... }));
  layers.push_back(createLayer("fully_connected", { ... }));

  for (auto &layer : layers) model->addLayer(layer);
}
```

**소스 — createAttention() (`emitter_cpp/source_attention.py` 생성):**
```cpp
// 현재: mha_core에 3개 입력, KV cache는 레이어 속성으로 암묵적 할당
layers.push_back(createLayer("mha_core", {
  withKey("name", A),
  withKey("num_heads", n_heads),
  withKey("num_heads_kv", n_heads / GQA_SIZE),
  withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
  withKey("sliding_window", SLIDING_WINDOW),
  withKey("rope_theta", ROPE_THETA),
  withKey("max_new_tokens", NUM_TO_GENERATE),
  withKey("input_layers", Q_norm + "," + K_norm + "," + V)  // ← 3 inputs
}));
```

**소스 — initialize() (`emitter_cpp/source_custom.py` 생성):**
```cpp
void Qwen3CausalLM::initialize() {
  registerCustomLayers();
  constructModel();
  model->setProperty({
    withKey("batch_size", 1),
    withKey("epochs", "1"),
    withKey("model_tensor_type", "FP32-FP32")
  });
  model->compile(ml::train::ExecutionMode::INFERENCE);
  model->initialize(ml::train::ExecutionMode::INFERENCE);
}
```

**코드 생성 유틸리티 (`emitter_cpp/helpers.py`):**
```python
def _cpp_layer(layer_type, props, indent=2):
    """createLayer() 호출 코드 생성"""
    pad = "  " * indent
    lines = []
    lines.append(pad + 'layers.push_back(createLayer("' + layer_type + '", {')
    for i, p in enumerate(props):
        comma = "," if i < len(props) - 1 else ""
        lines.append(pad + "  " + p + comma)
    lines.append(pad + "}));")
    return lines
```

### 11.2 변경 계획: `--external-kv-cache` 옵션

```bash
# 기존 (기본값) — 내부 cache (하위 호환)
python converter.py --model Qwen/Qwen3-0.6B

# 새 옵션 — 외부 cache
python converter.py --model Qwen/Qwen3-0.6B --external-kv-cache
```

### 11.3 변경 파일별 상세

#### `converter.py` — CLI 옵션 추가
```python
parser.add_argument("--external-kv-cache", action="store_true",
                    help="Generate external KV cache management code")
```

#### `emitter_cpp/header.py` — 멤버 변수 추가 (external-kv-cache 모드)
```cpp
// 기존 멤버에 추가
protected:
  // External KV cache (--external-kv-cache)
  struct KVCacheBuffers {
    std::vector<float *> key_bufs;
    std::vector<float *> val_bufs;
  };
  KVCacheBuffers kv_cache;
  std::vector<ml::train::Tensor> key_cache_tensors;
  std::vector<ml::train::Tensor> val_cache_tensors;

  void allocateKVCache();
```

#### `emitter_cpp/source_construct.py` — cache 할당 메서드 + constructModel 변경
```cpp
// 새로 생성되는 메서드
void Qwen3CausalLM::allocateKVCache() {
  ml::train::TensorDim cache_dim({1, 1,
    INIT_SEQ_LEN + NUM_TO_GENERATE,
    NUM_KV_HEADS * HEAD_DIM});

  for (int i = 0; i < NUM_LAYERS; i++) {
    size_t cache_size = cache_dim.getFeatureLen();
    kv_cache.key_bufs.push_back(new float[cache_size]());
    kv_cache.val_bufs.push_back(new float[cache_size]());

    key_cache_tensors.push_back(
      ml::train::Tensor::fromData(cache_dim, kv_cache.key_bufs[i]));
    val_cache_tensors.push_back(
      ml::train::Tensor::fromData(cache_dim, kv_cache.val_bufs[i]));
  }
}
```

#### `emitter_cpp/source_attention.py` — 5-input mha_core 생성
```cpp
// --external-kv-cache 모드에서 생성되는 코드
auto cache_k_name = key_cache_tensors[layer_id].name();
auto cache_v_name = val_cache_tensors[layer_id].name();

layers.push_back(createLayer("mha_core", {
  withKey("name", A),
  withKey("num_heads", n_heads),
  withKey("num_heads_kv", n_heads / GQA_SIZE),
  withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
  withKey("input_layers", Q_norm + "," + K_norm + "," + V
          + "," + cache_k_name + "," + cache_v_name)  // ← 5 inputs
}));
```

#### `emitter_cpp/source_custom.py` — initialize 변경
```cpp
void Qwen3CausalLM::initialize() {
  registerCustomLayers();
  allocateKVCache();    // ← 새로 추가 (--external-kv-cache 모드)
  constructModel();
  model->setProperty({ ... });
  model->compile(ml::train::ExecutionMode::INFERENCE);
  model->initialize(ml::train::ExecutionMode::INFERENCE);
}
```

### 11.4 변경되지 않는 파일

| 파일 | 이유 |
|------|------|
| `emitter_cpp/source_block.py` | 블록 구조는 동일 (norm → attention → residual → ffn → residual) |
| `emitter_cpp/source_ffn.py` | FFN은 KV cache와 무관 |
| `emitter_ini/` | INI 형식은 Tensor API와 무관 |
| `emitter_json.py` | JSON 메타데이터는 Tensor API와 무관 |
| `weight_converter.py` | 가중치 변환은 Tensor API와 무관 |
| `patterns/` | 패턴 감지는 Tensor API와 무관 |
| `node_mapper.py`, `module_mapper.py` 등 | FX 그래프 분석은 Tensor API와 무관 |

### 11.5 기존 생성 코드와의 호환성

```
--external-kv-cache 없음 (기본):
  → 현재와 100% 동일한 코드 생성
  → mha_core 3-input, 내부 cache

--external-kv-cache 있음:
  → header.py: KVCacheBuffers 멤버 추가
  → source_construct.py: allocateKVCache() 추가
  → source_attention.py: mha_core 5-input
  → source_custom.py: initialize()에 allocateKVCache() 호출 추가
```

---

## 12. 전체 변경 파일 요약

### Core (Phase 1) — Tensor API
| 파일 | 변경 |
|------|------|
| `api/ccapi/include/tensor_api.h` | Pimpl 기반 재구현 (Var_Grad 상속 제거), `fromData()`, `isExternal()`, 연산 메서드 |
| `nntrainer/tensor/tensor.h` | `fromData()`, `isExternal()`, `isMaterialized()`, `external_` 멤버 |
| `nntrainer/tensor/tensor.cpp` | 위 메서드 구현 |
| `nntrainer/tensor/tensor_pool.h` | `requestOrPlaceholder()` |
| `nntrainer/tensor/tensor_pool.cpp` | `requestOrPlaceholder()` 구현 |

### Core (Phase 2) — Layer 연동
| 파일 | 변경 |
|------|------|
| `api/ccapi/include/layer.h` | `operator()(Tensor)` 추가 |
| `api/ccapi/include/model.h` | `compile(Tensor, Tensor)` 오버로드 추가 |
| `nntrainer/graph/network_graph.cpp` | `finalizeContext()` 외부 텐서 PLACEHOLDER 처리 |

### Layers (Phase 3) — MHA 외부 캐시
| 파일 | 변경 |
|------|------|
| `nntrainer/layers/multi_head_attention_layer.h` | `use_external_cache` 멤버 |
| `nntrainer/layers/multi_head_attention_layer.cpp` | `finalize()` 입력 수 분기, `forwarding()`/`incremental_forwarding()` 캐시 접근 분기 |

### Applications (Phase 4) — CausalLM
| 파일 | 변경 |
|------|------|
| `Applications/CausalLM/causal_lm.h` | `KVCacheBuffers` 구조체, `key/val_cache_tensors` 멤버 |
| `Applications/CausalLM/causal_lm.cpp` | `constructModel()` 외부 cache 할당, `createAttention()` 5-input, `save/load_kvcache()` 간소화 |
| `Applications/CausalLM/layers/mha_core.h` | `use_external_cache` 멤버 |
| `Applications/CausalLM/layers/mha_core.cpp` | `finalize()` 입력 수 분기, `incremental_forwarding()` 캐시 접근 분기 |
| `Applications/PicoGPT/jni/main.cpp` | **변경 없음** (하위 호환) |
| `Applications/LLaMA/jni/main.cpp` | **변경 없음** (하위 호환) |
| `Applications/Resnet/jni/main.cpp` | **변경 없음** |
| `Applications/SimpleFC/jni/main.cpp` | **변경 없음** |
| `Applications/MNIST/jni/main.cpp` | **변경 없음** |
| `Applications/VGG/jni/main.cpp` | **변경 없음** |
| `Applications/Custom/momentum.cpp` | **변경 없음** |

### TorchFXConverter (Phase 5)
| 파일 | 변경 |
|------|------|
| `Applications/TorchFXConverter/converter.py` | `--external-kv-cache` CLI 옵션 |
| `Applications/TorchFXConverter/emitter_cpp/header.py` | 외부 cache 멤버 선언 (조건부) |
| `Applications/TorchFXConverter/emitter_cpp/source_construct.py` | `allocateKVCache()` 메서드 생성 (조건부) |
| `Applications/TorchFXConverter/emitter_cpp/source_attention.py` | mha_core 5-input 생성 (조건부) |
| `Applications/TorchFXConverter/emitter_cpp/source_custom.py` | `initialize()`에 cache 할당 호출 (조건부) |
| 기타 (ini, json, weight, patterns, mapper) | **변경 없음** |

### Tests
| 파일 | 변경 |
|------|------|
| `test/ccapi/unittest_ccapi.cpp` | **변경 없음** (기존 `compile()` 오버로드 유지) |
| `test/ccapi/unittest_ccapi_tensor.cpp` | 새 Tensor API 테스트 추가 (Pimpl, fromData, 연산 등) |
| 신규: `test/ccapi/unittest_tensor_graph.cpp` | 심볼릭 그래프 구축 + compile 테스트 |

---

## 13. 구현 순서 (세부 단계 + 테스트)

각 Step은 독립적으로 빌드/테스트 가능한 단위. 한 Step 완료 후 다음 Step 진행.

---

### Phase 1: Core Tensor API (Pimpl 기반 재구현)

#### Step 1-1: Pimpl 구조 + 기본 생성자

**구현:**
```
api/ccapi/include/tensor_api.h
  - Var_Grad 상속 제거
  - class Tensor { struct Impl; std::unique_ptr<Impl> impl_; }
  - 기본 생성자: Tensor()
  - 심볼릭 생성자: Tensor(TensorDim dim, std::string name = "")
  - 소멸자, 이동 생성자/대입 (unique_ptr 때문에 명시 필요)
  - 복사 생성자/대입 (shallow copy — 같은 그래프 노드 공유)
  - 기본 접근자: shape(), name(), dtype(), isValid()

api/ccapi/src/tensor_api.cpp (신규)
  - Impl 정의: { TensorDim dim; string name; bool external; ... }
  - 위 메서드 구현

api/ccapi/meson.build
  - ccapi_src에 'src/tensor_api.cpp' 추가
```

**테스트 (unittest_ccapi_tensor.cpp):**
```cpp
// 기존 tensor_01_p 유지 (하위 호환 확인)

TEST(nntrainer_ccapi_tensor, default_construct_p) {
  ml::train::Tensor t;
  EXPECT_FALSE(t.isValid());
}

TEST(nntrainer_ccapi_tensor, symbolic_construct_p) {
  ml::train::Tensor t({1, 1, 28, 28}, "input");
  EXPECT_TRUE(t.isValid());
  EXPECT_EQ(t.name(), "input");
  EXPECT_EQ(t.shape().batch(), 1);
  EXPECT_EQ(t.shape().width(), 28);
}

TEST(nntrainer_ccapi_tensor, move_construct_p) {
  ml::train::Tensor a({1, 1, 28, 28});
  ml::train::Tensor b(std::move(a));
  EXPECT_TRUE(b.isValid());
  EXPECT_FALSE(a.isValid());
}

TEST(nntrainer_ccapi_tensor, copy_construct_p) {
  ml::train::Tensor a({1, 1, 28, 28}, "shared");
  ml::train::Tensor b(a);
  EXPECT_EQ(a.name(), b.name());
}
```

**빌드 확인:** `meson test -C builddir unittest_ccapi`

---

#### Step 1-2: fromData, zeros, ones (Eager 텐서)

**구현:**
```
api/ccapi/include/tensor_api.h
  - static Tensor fromData(TensorDim dim, void *data, std::string name = "")
  - static Tensor zeros(TensorDim dim, std::string name = "")
  - static Tensor ones(TensorDim dim, std::string name = "")
  - bool isExternal() const
  - bool isMaterialized() const

api/ccapi/src/tensor_api.cpp
  - fromData: impl_->external = true, impl_->eager_data = shared_ptr<nntrainer::Tensor>(외부 포인터 매핑)
  - zeros/ones: impl_->eager_data = make_shared<nntrainer::Tensor>(dim, Initializer::ZEROS/ONES)
  - isMaterialized: eager_data != nullptr || bound_internal != nullptr
```

**테스트:**
```cpp
TEST(nntrainer_ccapi_tensor, from_data_p) {
  float buf[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  ml::train::Tensor t = ml::train::Tensor::fromData({1, 1, 3, 4}, buf);
  EXPECT_TRUE(t.isExternal());
  EXPECT_TRUE(t.isMaterialized());
  EXPECT_EQ(t.shape().height(), 3);
  EXPECT_EQ(t.shape().width(), 4);
}

TEST(nntrainer_ccapi_tensor, zeros_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 3});
  EXPECT_FALSE(t.isExternal());
  EXPECT_TRUE(t.isMaterialized());
}

TEST(nntrainer_ccapi_tensor, ones_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 3});
  EXPECT_TRUE(t.isMaterialized());
}

TEST(nntrainer_ccapi_tensor, symbolic_not_materialized_p) {
  ml::train::Tensor t({1, 1, 28, 28});
  EXPECT_FALSE(t.isMaterialized());  // 심볼릭 텐서는 compile 전까지 미실체화
  EXPECT_FALSE(t.isExternal());
}
```

---

#### Step 1-3: 데이터 접근 메서드

**구현:**
```
api/ccapi/include/tensor_api.h
  - template<typename T> const T *data() const
  - template<typename T> T *mutable_data()
  - float getValue(unsigned int batch, unsigned int c, unsigned int h, unsigned int w) const
  - void setValue(unsigned int batch, unsigned int c, unsigned int h, unsigned int w, float value)
  - void copyFrom(const void *src)
  - void setData(void *new_ptr)  // 외부 포인터 교체 (fromData 텐서만)
```

**테스트:**
```cpp
TEST(nntrainer_ccapi_tensor, data_access_from_data_p) {
  float buf[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 3}, buf);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 1.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 2), 6.0f);
  EXPECT_EQ(t.data<float>(), buf);  // zero-copy 확인
}

TEST(nntrainer_ccapi_tensor, set_value_p) {
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  t.setValue(0, 0, 1, 1, 42.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 42.0f);
}

TEST(nntrainer_ccapi_tensor, set_data_replace_ptr_p) {
  float buf1[4] = {1, 2, 3, 4};
  float buf2[4] = {5, 6, 7, 8};
  auto t = ml::train::Tensor::fromData({1, 1, 2, 2}, buf1);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 1.0f);
  t.setData(buf2);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 5.0f);
}

TEST(nntrainer_ccapi_tensor, set_data_on_symbolic_n) {
  ml::train::Tensor t({1, 1, 2, 2});
  EXPECT_THROW(t.setData(nullptr), std::runtime_error);
}

TEST(nntrainer_ccapi_tensor, copy_from_p) {
  float src[4] = {10, 20, 30, 40};
  auto t = ml::train::Tensor::zeros({1, 1, 2, 2});
  t.copyFrom(src);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 10.0f);
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 1, 1), 40.0f);
}

TEST(nntrainer_ccapi_tensor, data_access_unmaterialized_n) {
  ml::train::Tensor t({1, 1, 28, 28});
  EXPECT_THROW(t.data<float>(), std::runtime_error);
  EXPECT_THROW(t.getValue(0, 0, 0, 0), std::runtime_error);
}
```

---

#### Step 1-4: 기존 API 하위 호환

**구현:**
```
api/ccapi/include/tensor_api.h
  - setSrcLayer(shared_ptr<Layer>) — 기존 유지
  - getSrcLayer() — 기존 유지
  - clone() — Pimpl 기반 deep copy
```

**테스트:**
```cpp
// 기존 tensor_01_p 그대로 통과해야 함
TEST(nntrainer_ccapi, tensor_01_p) {
  ml::train::Tensor a;
  std::shared_ptr<ml::train::Layer> layer = ml::train::layer::Input(
    {ml::train::withKey("name", "input0"),
     ml::train::withKey("input_shape", "1:1:62720")});
  a.setSrcLayer(layer);
  EXPECT_EQ(a.getSrcLayer()->getName(), "input0");
}

TEST(nntrainer_ccapi_tensor, clone_eager_p) {
  auto orig = ml::train::Tensor::zeros({1, 1, 2, 2});
  orig.setValue(0, 0, 0, 0, 99.0f);
  auto cloned = orig.clone();
  cloned.setValue(0, 0, 0, 0, 1.0f);
  EXPECT_FLOAT_EQ(orig.getValue(0, 0, 0, 0), 99.0f);  // 원본 불변
  EXPECT_FLOAT_EQ(cloned.getValue(0, 0, 0, 0), 1.0f);
}
```

**Phase 1 완료 기준:**
- `meson test -C builddir unittest_ccapi` 전체 통과
- 기존 `unittest_ccapi.cpp` 테스트들 회귀 없음
- Pimpl Tensor가 `Var_Grad`에 의존하지 않음

---

### Phase 2: Layer 연동 + 심볼릭 그래프

#### Step 2-1: Layer::operator()(Tensor) — 그래프 edge 기록

**구현:**
```
api/ccapi/include/tensor_api.h (내부 구조)
  - Impl에 추가:
    struct GraphNode {
      std::shared_ptr<Layer> producing_layer;
      unsigned int output_index;
      std::vector<GraphNode *> inputs;
    };
    std::shared_ptr<GraphNode> graph_node;

api/ccapi/include/layer.h
  - Tensor operator()(const Tensor &input)
  - Tensor operator()(const std::vector<Tensor> &inputs)
  // LayerHandle = shared_ptr<Layer> 이므로 free function 또는 wrapper 필요
  // → LayerHandle을 감싸는 CallableLayer 또는 free function

api/ccapi/src/tensor_api.cpp
  - operator() 구현: 새 Tensor 생성, graph_node에 producing_layer + inputs 기록
```

**테스트 (unittest_ccapi_tensor.cpp에 추가):**
```cpp
TEST(nntrainer_ccapi_tensor, layer_call_symbolic_p) {
  using namespace ml::train;
  auto input = Tensor({1, 1, 784}, "input");
  auto fc = createLayer("fully_connected", {"unit=256", "name=fc1"});
  auto output = fc(input);
  EXPECT_TRUE(output.isValid());
  EXPECT_FALSE(output.isMaterialized());  // 심볼릭
  EXPECT_EQ(output.shape().width(), 256);
}

TEST(nntrainer_ccapi_tensor, layer_chain_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 784}, "x");
  auto h = createLayer("fully_connected", {"unit=128", "name=fc1"})(x);
  auto y = createLayer("fully_connected", {"unit=10", "name=fc2"})(h);
  EXPECT_TRUE(y.isValid());
}

TEST(nntrainer_ccapi_tensor, multi_input_layer_p) {
  using namespace ml::train;
  auto a = Tensor({1, 1, 1, 256}, "a");
  auto b = Tensor({1, 1, 1, 256}, "b");
  auto added = createLayer("Addition", {"name=add1"})({a, b});
  EXPECT_TRUE(added.isValid());
}
```

---

#### Step 2-2: Tensor 연산 → 암묵적 레이어

**구현:**
```
api/ccapi/include/tensor_api.h
  - Tensor add(const Tensor &other) const       // → Addition 레이어
  - Tensor multiply(const Tensor &other) const   // → Multiply 레이어
  - Tensor reshape(TensorDim new_shape) const     // → Reshape 레이어

api/ccapi/src/tensor_api.cpp
  - add: createLayer("Addition") 생성 → operator()({*this, other})
  - multiply: createLayer("Multiply") 생성
  - reshape: createLayer("Reshape", {"target_shape=..."}) 생성
```

**테스트:**
```cpp
TEST(nntrainer_ccapi_tensor, add_symbolic_p) {
  using namespace ml::train;
  auto a = Tensor({1, 1, 1, 256}, "a");
  auto b = Tensor({1, 1, 1, 256}, "b");
  auto c = a.add(b);
  EXPECT_TRUE(c.isValid());
  EXPECT_EQ(c.shape().width(), 256);
}

TEST(nntrainer_ccapi_tensor, residual_connection_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 256}, "x");
  auto h = createLayer("fully_connected", {"unit=256", "name=fc1"})(x);
  auto out = x.add(h);  // skip connection
  EXPECT_TRUE(out.isValid());
}
```

---

#### Step 2-3: Model::compile(Tensor, Tensor) — 그래프 추출

**구현:**
```
api/ccapi/include/model.h
  - virtual int compile(const Tensor &input, const Tensor &output,
                        ExecutionMode mode = ExecutionMode::TRAIN) = 0;
  // 기존 compile(ExecutionMode) 유지

nntrainer/models/neuralnet.h / neuralnet.cpp
  - compile(Tensor, Tensor) 구현:
    1. output의 graph_node에서 BFS/DFS로 모든 레이어 수집
    2. 위상 정렬
    3. 각 레이어에 대해 addLayer() 호출 + input_layers 설정
    4. 기존 compile(mode) 호출
    5. _bind(): API Tensor ↔ 내부 Var_Grad 바인딩
```

**테스트 (신규: test/ccapi/unittest_tensor_graph.cpp):**
```cpp
TEST(nntrainer_ccapi_graph, simple_fc_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 784}, "input");
  auto y = createLayer("fully_connected", {"unit=10", "name=fc"})(x);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
}

TEST(nntrainer_ccapi_graph, multi_layer_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 784}, "input");
  auto h = createLayer("fully_connected", {"unit=128", "name=fc1"})(x);
  h = createLayer("activation", {"activation=relu", "name=relu1"})(h);
  auto y = createLayer("fully_connected", {"unit=10", "name=fc2"})(h);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(), ML_ERROR_NONE);
}

TEST(nntrainer_ccapi_graph, residual_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 256}, "input");
  auto h = createLayer("fully_connected", {"unit=256", "name=fc1"})(x);
  auto out = x.add(h);
  auto y = createLayer("fully_connected", {"unit=10", "name=fc_out"})(out);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(x, y), ML_ERROR_NONE);
}

TEST(nntrainer_ccapi_graph, existing_add_layer_still_works_p) {
  // 기존 addLayer 방식 여전히 동작 확인
  using namespace ml::train;
  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  model->addLayer(createLayer("input", {"name=in", "input_shape=1:1:784"}));
  model->addLayer(createLayer("fully_connected", {"name=fc", "unit=10", "input_layers=in"}));
  EXPECT_EQ(model->compile(), ML_ERROR_NONE);
}
```

**meson.build 변경 (test/ccapi/meson.build):**
```meson
ccapi_targets = [
  'unittest_ccapi.cpp',
  'unittest_ccapi_tensor.cpp',
  'unittest_tensor_graph.cpp'     # 새로 추가
]
```

---

#### Step 2-4: _bind() + 데이터 주입/추출

**구현:**
```
api/ccapi/src/tensor_api.cpp
  - _bind(Var_Grad *internal): compile 후 API Tensor가 내부 Var_Grad를 참조
  - copyFrom: bound_internal이 있으면 내부 텐서에 복사
  - data<T>: bound_internal이 있으면 내부 텐서 포인터 반환
```

**테스트:**
```cpp
TEST(nntrainer_ccapi_graph, data_injection_after_compile_p) {
  using namespace ml::train;
  auto x = Tensor({1, 1, 1, 4}, "input");
  auto y = createLayer("fully_connected", {"unit=2", "name=fc"})(x);

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  model->compile(x, y);
  model->initialize();

  // compile 후 x는 materialized
  EXPECT_TRUE(x.isMaterialized());
  float input_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  x.copyFrom(input_data);
  EXPECT_FLOAT_EQ(x.getValue(0, 0, 0, 0), 1.0f);
}
```

---

#### Step 2-5: Lazy chaining (chain/eval)

**구현:**
```
api/ccapi/include/tensor_api.h
  - Tensor &chain()
  - Tensor &add_i(float value)
  - Tensor &multiply_i(float value)
  - Tensor eval()

api/ccapi/src/tensor_api.cpp
  - Impl에 std::vector<std::function<void(nntrainer::Tensor&)>> call_chain
  - chain(): call_chain.clear()
  - add_i/multiply_i: call_chain.push_back(lambda)
  - eval(): isMaterialized 확인 → 체인 순차 실행 → clear
```

**테스트:**
```cpp
TEST(nntrainer_ccapi_tensor, lazy_chain_p) {
  auto t = ml::train::Tensor::ones({1, 1, 2, 2});
  t.chain().multiply_i(2.0f).add_i(1.0f).eval();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 3.0f);  // 1*2+1
}

TEST(nntrainer_ccapi_tensor, lazy_chain_order_p) {
  auto t = ml::train::Tensor::ones({1, 1, 1, 1});
  t.chain().add_i(3.0f).multiply_i(2.0f).eval();
  EXPECT_FLOAT_EQ(t.getValue(0, 0, 0, 0), 8.0f);  // (1+3)*2
}

TEST(nntrainer_ccapi_tensor, lazy_eval_on_symbolic_n) {
  ml::train::Tensor t({1, 1, 2, 2});
  t.chain().add_i(1.0f);
  EXPECT_THROW(t.eval(), std::runtime_error);
}
```

**Phase 2 완료 기준:**
- `unittest_ccapi` 전체 통과 (기존 회귀 없음)
- `unittest_tensor_graph` 전체 통과
- 심볼릭 → compile → 내부 매핑 → 데이터 주입 full cycle 동작

---

### Phase 3: MHA 외부 캐시 지원

#### Step 3-1: MultiHeadAttentionLayer 입력 수 분기

**구현:**
```
nntrainer/layers/multi_head_attention_layer.h
  - bool use_external_cache = false;

nntrainer/layers/multi_head_attention_layer.cpp
  - finalize(): num_inputs >= 5 → use_external_cache = true, requestTensor() 스킵
  - forwarding(): cache 접근 분기 (getInput vs getTensor)
  - incremental_forwarding(): 동일 분기
```

**테스트 (test/unittest/ 기존 MHA 테스트에 추가):**
```cpp
// 기존 MHA 테스트가 여전히 통과하는지 확인 (회귀 테스트)
// → 기존 unittest_nntrainer_layers에서 MHA 관련 테스트 실행

// 새 테스트: 5-input MHA (외부 cache)
TEST(nntrainer_mha, external_cache_finalize_p) {
  // MHA 레이어에 5개 입력 설정 → use_external_cache = true 확인
  // (단위 테스트로 InitLayerContext mock 필요할 수 있음)
}
```

---

#### Step 3-2: TensorPool PLACEHOLDER 처리

**구현:**
```
nntrainer/tensor/tensor_pool.h
  - int requestOrPlaceholder(const std::string &name, const TensorDim &dim,
                             bool is_external)

nntrainer/tensor/tensor_pool.cpp
  - is_external=true → SourceDetails{lifespan=UNMANAGED}, MemoryPool 할당 안 함
  - fillPlaceholder(name, external_tensor) + syncDependents()

nntrainer/graph/network_graph.cpp
  - finalizeContext(): fromData 텐서 → requestOrPlaceholder(is_external=true)
```

**테스트 (test/unittest/unittest_nntrainer_tensor_pool.cpp에 추가):**
```cpp
TEST(nntrainer_tensor_pool, placeholder_request_p) {
  TensorPool pool;
  float ext_buf[12] = {};
  auto idx = pool.requestOrPlaceholder("ext_tensor", {1,1,3,4}, true);
  pool.fillPlaceholder("ext_tensor", ext_buf);
  // 실제 데이터가 외부 버퍼를 가리키는지 확인
}

TEST(nntrainer_tensor_pool, placeholder_no_alloc_p) {
  // PLACEHOLDER 텐서는 MemoryPool에서 할당하지 않는지 확인
}
```

---

#### Step 3-3: 통합 테스트 — fromData + MHA + Model

**테스트 (test/ccapi/unittest_tensor_graph.cpp에 추가):**
```cpp
TEST(nntrainer_ccapi_graph, external_cache_mha_compile_p) {
  using namespace ml::train;

  auto input = Tensor({1, 1, 4, 64}, "input");   // [batch, 1, seq, dim]

  float key_buf[1 * 1 * 32 * 64] = {};
  float val_buf[1 * 1 * 32 * 64] = {};
  auto key_cache = Tensor::fromData({1, 1, 32, 64}, key_buf);
  auto val_cache = Tensor::fromData({1, 1, 32, 64}, val_buf);

  auto q = createLayer("fully_connected", {"unit=64", "name=q_proj"})(input);
  auto k = createLayer("fully_connected", {"unit=64", "name=k_proj"})(input);
  auto v = createLayer("fully_connected", {"unit=64", "name=v_proj"})(input);

  auto attn = createLayer("multi_head_attention", {
    "name=mha", "num_heads=4"
  })({q, k, v, key_cache, val_cache});

  auto model = createModel(ModelType::NEURAL_NET, {"batch_size=1"});
  EXPECT_EQ(model->compile(input, attn, ExecutionMode::INFERENCE), ML_ERROR_NONE);
  EXPECT_EQ(model->initialize(ExecutionMode::INFERENCE), ML_ERROR_NONE);

  // key_cache는 외부 → PLACEHOLDER, 내부 할당 안 함
  EXPECT_TRUE(key_cache.isExternal());
  EXPECT_TRUE(key_cache.isMaterialized());
}
```

**Phase 3 완료 기준:**
- 기존 MHA 테스트 (3-4 input) 전체 통과
- 5-input MHA (외부 cache) compile + initialize 성공
- TensorPool placeholder 텐서가 MemoryPool 할당 안 함 확인

---

### Phase 4: Applications 마이그레이션

#### Step 4-1: MHACoreLayer (CausalLM) 외부 캐시 분기

**구현:**
```
Applications/CausalLM/layers/mha_core.h
  - bool use_external_cache = false;

Applications/CausalLM/layers/mha_core.cpp
  - finalize(): num_inputs >= 4 분기
  - incremental_forwarding(): getInput(3)/getInput(4) vs getTensor() 분기
```

**테스트:**
```
- CausalLM 빌드 성공 확인
- 기존 3-input 모드로 compile + initialize 테스트 (하위 호환)
```

---

#### Step 4-2: CausalLM constructModel + createAttention 변경

**구현:**
```
Applications/CausalLM/causal_lm.h
  - KVCacheBuffers 구조체, key/val_cache_tensors 멤버 추가

Applications/CausalLM/causal_lm.cpp
  - constructModel(): 외부 cache 버퍼 할당 + fromData 텐서 생성
  - createAttention(): 5-input mha_core
```

**테스트:**
```
- CausalLM 빌드 성공
- constructModel() + compile(INFERENCE) + initialize(INFERENCE) 성공
```

---

#### Step 4-3: CausalLM save/load_kvcache 간소화

**구현:**
```
Applications/CausalLM/causal_lm.cpp
  - save_kvcache: forEachLayer 제거 → 외부 버퍼 직접 write
  - load_kvcache: forEachLayer 제거 → 외부 버퍼 직접 read
```

**테스트:**
```
- save_kvcache → load_kvcache round-trip 테스트
- 저장한 데이터가 외부 버퍼에 올바르게 로드되는지 확인
```

---

#### Step 4-4: 하위 호환 검증

**테스트:**
```
- PicoGPT: 빌드 성공 확인 (코드 변경 없음)
- LLaMA: 빌드 성공 확인 (코드 변경 없음)
- Resnet: 빌드 성공 확인
- SimpleFC: 빌드 성공 확인
- unittest_ccapi 전체 통과 (기존 addLayer 패턴)
```

**Phase 4 완료 기준:**
- CausalLM 빌드 + 외부 cache 모드 동작
- 기타 Applications 전부 빌드 성공 (코드 변경 없음)
- `meson test -C builddir` 전체 통과

---

### Phase 5: TorchFXConverter

#### Step 5-1: --external-kv-cache CLI 옵션

**구현:**
```
Applications/TorchFXConverter/converter.py
  - argparse에 --external-kv-cache 추가
  - config dict에 'external_kv_cache' 플래그 전달
```

**테스트:**
```python
# test_converter.py 또는 수동 검증
# 기본 모드: 기존과 동일한 코드 출력
python converter.py --model test_model --output /tmp/test_default
diff /tmp/test_default /tmp/expected_default  # 차이 없어야 함
```

---

#### Step 5-2: header.py + source 파일 변경

**구현:**
```
emitter_cpp/header.py
  - external_kv_cache=True → KVCacheBuffers 멤버 + allocateKVCache() 선언 추가

emitter_cpp/source_construct.py
  - external_kv_cache=True → allocateKVCache() 메서드 생성

emitter_cpp/source_attention.py
  - external_kv_cache=True → mha_core input_layers에 cache 텐서 추가

emitter_cpp/source_custom.py
  - external_kv_cache=True → initialize()에 allocateKVCache() 호출
```

**테스트:**
```python
# --external-kv-cache 모드 출력 검증
python converter.py --model test_model --external-kv-cache --output /tmp/test_ext
# 생성된 코드에 KVCacheBuffers, allocateKVCache, 5-input mha_core 존재 확인
grep -q "KVCacheBuffers" /tmp/test_ext/header.h
grep -q "allocateKVCache" /tmp/test_ext/source.cpp
grep -q "cache_k_name" /tmp/test_ext/source.cpp
```

---

#### Step 5-3: 생성 코드 빌드 검증

**테스트:**
```
- --external-kv-cache로 생성된 C++ 코드가 NNTrainer와 링크하여 빌드 성공
- 기본 모드로 생성된 C++ 코드도 여전히 빌드 성공
```

**Phase 5 완료 기준:**
- 기본 모드: 기존과 100% 동일한 코드 생성
- --external-kv-cache: 올바른 외부 cache 코드 생성
- 두 모드 모두 생성 코드 빌드 성공

---

### 전체 완료 기준

```
✅ meson test -C builddir 전체 통과
✅ unittest_ccapi (기존 테스트 회귀 없음)
✅ unittest_ccapi_tensor (Pimpl, fromData, data 접근, lazy chain)
✅ unittest_tensor_graph (심볼릭 그래프, compile, bind, 외부 cache)
✅ unittest_nntrainer_tensor_pool (PLACEHOLDER)
✅ 기존 MHA 테스트 (3-4 input 하위 호환)
✅ CausalLM 빌드 + 외부 cache 동작
✅ PicoGPT, LLaMA, Resnet, SimpleFC 빌드 성공 (변경 없음)
✅ TorchFXConverter 기본 모드 동일 출력
✅ TorchFXConverter --external-kv-cache 모드 빌드 성공
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

# Tensor API with Lazy Evaluation — Comprehensive Design

## Overview

Tensor 생성 방식이 내부 메모리 매핑을 자동 결정하는 경량 API 확장.
기존 TensorPool/MemoryPool 파이프라인을 100% 재사용하며, 새로운 그래프 시스템 없이
외부 메모리(KV cache 등)를 zero-copy로 관리할 수 있게 한다.

### 핵심 원칙
```
Tensor(dim)           → UNIQUE      → MemoryPool 할당
Tensor::fromData(ptr) → PLACEHOLDER → MemoryPool 미할당, 외부 포인터 직접 사용
```

### 전체 작업 범위
1. **Phase 1**: Core Tensor API 확장
2. **Phase 2**: MultiHeadAttentionLayer / MHACoreLayer 외부 캐시 지원
3. **Phase 3**: Applications 예제 마이그레이션
4. **Phase 4**: TorchFXConverter C++ 에미터 업데이트

---

## Phase 1: Core Tensor API 확장

### 1.1 Tensor 클래스 변경

**파일**: `nntrainer/tensor/tensor.h`, `nntrainer/tensor/tensor.cpp`

#### 추가할 public API
```cpp
class Tensor {
public:
  // --- 기존 생성자 모두 유지 ---

  // 외부 데이터 바인딩 (compile 시 PLACEHOLDER → MemoryPool 미할당)
  static Tensor fromData(const TensorDim &dim, void *data,
                         const std::string &name = "");

  // 런타임 외부 포인터 교체 (추론 중 cache swap)
  void setExternalData(void *new_data);

  // 상태 확인
  bool isExternal() const;       // fromData로 생성되었는가?
  bool isMaterialized() const;   // getData() != nullptr?

private:
  bool external_ = false;        // fromData 플래그
  void *external_ptr_ = nullptr; // 외부 원시 포인터 보관 (lifetime은 사용자 책임)
};
```

#### fromData() 구현 세부사항
```cpp
Tensor Tensor::fromData(const TensorDim &dim, void *data,
                        const std::string &name) {
  Tensor t(name, dim.getFormat(), dim.getDataType());
  t.external_ = true;
  t.external_ptr_ = data;
  // 메모리 할당하지 않음 — compile 시 TensorPool::placeholder() 경로를 탐
  // dim 정보만 설정하여 그래프 빌드에 필요한 shape 정보 제공
  t.itensor->setTensorDim(dim);
  return t;
}
```

- Tensor를 생성하되 **메모리 할당은 하지 않음**
- `external_` 플래그로 compile 시 PLACEHOLDER 경로로 분류
- 실제 데이터 바인딩은 `fillPlaceholder()` 또는 `setExternalData()`에서 수행

#### setExternalData() 구현 세부사항
```cpp
void Tensor::setExternalData(void *new_data) {
  NNTR_THROW_IF(!external_, std::invalid_argument)
    << "setExternalData() is only valid for fromData() tensors";
  external_ptr_ = new_data;
  // MemoryData를 외부 포인터로 래핑하여 itensor에 설정
  auto mem = std::make_shared<MemoryData>(external_ptr_, dim.getDataLen());
  itensor->setMemoryData(mem, 0);
  // → TensorPool에 등록된 경우 syncDependents() 호출 필요
  //   (Phase 1에서는 TensorPool 측 연동도 함께 구현)
}
```

### 1.2 ml::train::Tensor (Public API) 확장

**파일**: `api/ccapi/include/tensor_api.h`

```cpp
namespace ml::train {
class Tensor : public nntrainer::Var_Grad {
public:
  // --- 기존 API 유지 ---

  // 외부 데이터 바인딩 (사용자 대면 API)
  static Tensor fromData(const TensorDim &dim, void *data,
                         const std::string &name = "");

  // 런타임 포인터 교체
  void setExternalData(void *new_data);

  // 상태 확인
  bool isExternal() const;
};
}
```

`ml::train::Tensor`는 내부 `nntrainer::Tensor`를 감싸는 래퍼이므로,
해당 메서드들은 내부 Tensor의 동일 메서드로 위임.

### 1.3 TensorPool 헬퍼 추가

**파일**: `nntrainer/tensor/tensor_pool.h`, `nntrainer/tensor/tensor_pool.cpp`

기존 `placeholder()` + `fillPlaceholder()`가 이미 필요한 기능을 제공.
추가할 것은 편의 메서드 하나:

```cpp
/// Tensor의 is_external 플래그에 따라 자동으로 request() 또는 placeholder() 호출
Tensor *TensorPool::requestOrPlaceholder(
    const std::string &name,
    const TensorDim &dim,
    const std::vector<unsigned int> &exec_order,
    TensorLifespan lifespan,
    bool is_external) {
  if (is_external) {
    return placeholder(name, dim);
    // → exec_order={}, UNMANAGED → MemoryPool 할당 안함
  }
  return request(name, dim, exec_order, lifespan);
  // → 기존 UNIQUE 경로
}
```

#### 기존 파이프라인 재사용 흐름
```
fromData 텐서:
  → requestOrPlaceholder(is_external=true)
  → placeholder(name, dim)
  → SourceDetails{token=0, lifespan=UNMANAGED, exec_order={}}
  → MemoryPool 할당 안 함
  → fillPlaceholder(name, external_tensor)
  → spec.tensor->setData(external_ptr)
  → syncDependents(spec)  // view 텐서들 자동 갱신

일반 텐서:
  → requestOrPlaceholder(is_external=false)
  → request(name, dim, exec_order, lifespan)
  → 기존 UNIQUE 경로 그대로
```

---

## Phase 2: MultiHeadAttentionLayer 외부 캐시 지원

### 2.1 Core MHA Layer (nntrainer 내장)

**파일**: `nntrainer/layers/multi_head_attention_layer.h`, `.cpp`

#### INOUT_INDEX 확장
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

#### 멤버 변수 추가
```cpp
// multi_head_attention_layer.h
private:
  bool use_external_cache = false;
```

#### finalize() 변경
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

  // mask 판별: 4개 또는 6개일 때 mask 있음
  const bool provide_attention_mask = (num_inputs == 4 || num_inputs == 6);

  // 외부 캐시 판별: 5개 또는 6개일 때 외부 캐시
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

#### incremental_forwarding() 변경
```cpp
void MultiHeadAttentionLayer::incremental_forwarding(
    RunLayerContext &context, unsigned int from, unsigned int to,
    bool training) {
  // ...기존 코드 동일...

  // 캐시 텐서 접근 분기만 변경
  Tensor &cache_key = use_external_cache
    ? context.getInput(INOUT_INDEX::CACHE_KEY)
    : context.getTensor(weight_idx[AttentionParams::cache_key]);
  Tensor &cache_value = use_external_cache
    ? context.getInput(INOUT_INDEX::CACHE_VALUE)
    : context.getTensor(weight_idx[AttentionParams::cache_value]);

  // 이후 cache_key_step, cached_key 등 getSharedDataTensor 패턴은 100% 동일
  // ...
}
```

#### forwarding() 변경
`forwarding()`에서도 동일한 분기 적용:
```cpp
void MultiHeadAttentionLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  // cache 텐서 접근 분기
  Tensor &cache_key = use_external_cache
    ? context.getInput(INOUT_INDEX::CACHE_KEY)
    : context.getTensor(weight_idx[AttentionParams::cache_key]);
  Tensor &cache_value = use_external_cache
    ? context.getInput(INOUT_INDEX::CACHE_VALUE)
    : context.getTensor(weight_idx[AttentionParams::cache_value]);
  // 나머지 동일
}
```

### 2.2 MHACoreLayer (CausalLM 커스텀 레이어)

**파일**: `Applications/CausalLM/layers/mha_core.h`, `mha_core.cpp`

MHACoreLayer도 동일한 패턴 적용:

```cpp
// mha_core.h
private:
  bool use_external_cache = false;

// mha_core.cpp finalize()
void MHACoreLayer::finalize(InitLayerContext &context) {
  auto num_inputs = context.getNumInputs();

  // 기존: 3개 (query, key, value)
  // 변경: 3-5개 (query, key, value, [cache_key], [cache_value])
  if (num_inputs >= 4) {
    use_external_cache = true;
    // CACHE_KEY = input[3], CACHE_VALUE = input[4]
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

// incremental_forwarding()
void MHACoreLayer::incremental_forwarding(...) {
  Tensor &cache_key = use_external_cache
    ? context.getInput(3)  // 외부 입력
    : context.getTensor(tensor_idx[AttentionParams::cache_key]);
  Tensor &cache_value = use_external_cache
    ? context.getInput(4)
    : context.getTensor(tensor_idx[AttentionParams::cache_value]);
  // 이후 동일
}
```

### 2.3 NetworkGraph에서 외부 텐서 처리

**파일**: `nntrainer/graph/network_graph.cpp`

`finalizeContext()`에서 입력 텐서가 외부(isExternal)인 경우 처리:

```cpp
// finalizeContext() 내부
// 입력 텐서 요청 시, 소스 텐서의 is_external 플래그 확인
for (auto &input_spec : input_specs) {
  if (input_spec.source_tensor && input_spec.source_tensor->isExternal()) {
    // PLACEHOLDER 경로 → TensorPool::placeholder()
    // exec_order = {}, lifespan = UNMANAGED
  } else {
    // 기존 UNIQUE 경로
  }
}
```

실제 데이터 바인딩은 `model->allocate()` 후 `fillPlaceholder()` 호출 시점에
외부 포인터가 연결됨.

### 2.4 하위 호환성 보장

| 시나리오 | num_inputs | use_external_cache | 동작 |
|----------|-----------|-------------------|------|
| 기존 MHA (query, key, value) | 3 | false | 내부 cache 할당 (기존과 동일) |
| 기존 MHA + mask | 4 | false | 내부 cache 할당 + mask (기존과 동일) |
| 새 API (q, k, v, ext_cache_k, ext_cache_v) | 5 | true | 외부 cache 사용 |
| 새 API + mask | 6 | true | 외부 cache + mask |

**기존 코드는 수정 없이 동작함.**

---

## Phase 3: Applications 마이그레이션

### 3.1 마이그레이션 전략

Applications은 3가지 패턴으로 분류:

| 패턴 | 대상 | 변경 필요 여부 |
|------|------|---------------|
| **A. INI 기반** | MNIST, AlexNet, VGG, Resnet 등 | 변경 불필요 (Tensor API 미사용) |
| **B. createLayer API** | PicoGPT, LLaMA, Layers 등 | MHA 외부 캐시 옵트인 시만 변경 |
| **C. KV cache 직접 관리** | CausalLM | 가장 큰 변경 필요 |

### 3.2 변경 불필요 Applications (패턴 A)

INI 설정 파일로 모델을 구성하는 예제들은 Tensor API 변경의 영향을 받지 않음:

- `Applications/MNIST/`
- `Applications/AlexNet/`
- `Applications/VGG/`
- `Applications/Resnet/`
- `Applications/TransferLearning/`
- `Applications/LogisticRegression/`
- `Applications/SimpleFC/`
- `Applications/YOLOv2/`, `Applications/YOLOv3/`
- `Applications/SimpleShot/`, `Applications/KNN/`
- `Applications/ReinforcementLearning/`

### 3.3 PicoGPT 마이그레이션

**파일**: `Applications/PicoGPT/jni/main.cpp`

**현재 상태**: `MultiHeadAttention` 레이어 사용, 내부 KV cache (변경 없이 동작).

**새 API 적용 시 (선택적 개선)**:
```cpp
// 현재: 내부 cache (변경 불필요, 하위 호환)
model->addLayer(ml::train::layer::MultiHeadAttention({
  "name=layer0/mha",
  "input_layers=q,k,v",
  "num_heads=12"
}));

// 새 API: 외부 cache로 전환 시 (선택적)
float *key_buf = new float[BATCH * MAX_SEQ * HEAD_DIM];
float *val_buf = new float[BATCH * MAX_SEQ * HEAD_DIM];
auto ext_key = nntrainer::Tensor::fromData(
  {BATCH, 1, MAX_SEQ, HEAD_DIM}, key_buf, "ext_key_cache");
auto ext_val = nntrainer::Tensor::fromData(
  {BATCH, 1, MAX_SEQ, HEAD_DIM}, val_buf, "ext_val_cache");

// MHA에 5개 입력으로 전달 → use_external_cache = true
model->addLayer(ml::train::layer::MultiHeadAttention({
  "name=layer0/mha",
  "input_layers=q,k,v,ext_key_cache,ext_val_cache",
  "num_heads=12"
}));
```

**결론**: PicoGPT는 기존 코드 그대로 동작. 외부 cache 전환은 선택적.

### 3.4 LLaMA 마이그레이션

**파일**: `Applications/LLaMA/jni/main.cpp`

LLaMA는 커스텀 `MultiHeadAttentionLayer`를 사용.
기존 코드 그대로 동작하되, 외부 KV cache 사용 시:

```cpp
// 현재: 커스텀 레이어 내부 cache
model->addLayer(createLayer("custom_multi_head_attention", {
  "name=layer0/attn",
  "input_layers=q,v,k",
  "num_heads=32"
}));

// 새 API: 외부 cache 전달 (선택적)
// 커스텀 레이어도 동일한 패턴으로 외부 cache 지원 가능
// → finalize()에서 num_inputs >= 5 체크만 추가
```

### 3.5 CausalLM 마이그레이션 (가장 큰 변경)

**파일**: `Applications/CausalLM/causal_lm.h`, `causal_lm.cpp`

#### 현재 KV cache 패턴 (제거 대상)
```cpp
// 현재: model->forEachLayer()로 내부 tensor에 직접 접근
void CausalLM::save_kvcache(std::string path, int to_) {
  model->forEachLayer([&f](Layer &l, RunLayerContext &context, void *idx) {
    if (l.getType() == MHACoreLayer::type) {
      auto k_cache = context.getTensor(0);  // 내부 tensor 직접 접근
      auto v_cache = context.getTensor(1);
      // ... save
    }
  });
}
```

#### 새 API 적용 후
```cpp
class CausalLM {
private:
  // 외부 KV cache 버퍼 (CausalLM이 소유)
  struct KVCacheBuffers {
    std::vector<float *> key_bufs;   // layer별 key cache
    std::vector<float *> val_bufs;   // layer별 value cache
  };
  KVCacheBuffers kv_cache;

  // 외부 Tensor handles
  std::vector<nntrainer::Tensor> key_cache_tensors;
  std::vector<nntrainer::Tensor> val_cache_tensors;
};

void CausalLM::constructModel() {
  // 각 레이어별 외부 KV cache 버퍼 할당
  for (int i = 0; i < NUM_LAYERS; i++) {
    kv_cache.key_bufs.push_back(
      new float[BATCH * MAX_SEQ * NUM_KV_HEADS * HEAD_DIM]);
    kv_cache.val_bufs.push_back(
      new float[BATCH * MAX_SEQ * NUM_KV_HEADS * HEAD_DIM]);

    // fromData로 외부 텐서 생성
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
      // ... other props
    }));
  }
}
```

#### save_kvcache / load_kvcache 간소화
```cpp
void CausalLM::save_kvcache(std::string path, int to) {
  auto f = checkedOpenStream<std::ofstream>(path, std::ios::binary);
  for (int i = 0; i < NUM_LAYERS; i++) {
    // 외부 버퍼에 직접 접근 (zero-copy)
    size_t bytes = BATCH * to * NUM_KV_HEADS * HEAD_DIM * sizeof(float);
    f.write(reinterpret_cast<char*>(kv_cache.key_bufs[i]), bytes);
    f.write(reinterpret_cast<char*>(kv_cache.val_bufs[i]), bytes);
  }
}

void CausalLM::load_kvcache(std::string path, int to) {
  auto f = checkedOpenStream<std::ifstream>(path, std::ios::binary);
  for (int i = 0; i < NUM_LAYERS; i++) {
    size_t bytes = BATCH * to * NUM_KV_HEADS * HEAD_DIM * sizeof(float);
    f.read(reinterpret_cast<char*>(kv_cache.key_bufs[i]), bytes);
    f.read(reinterpret_cast<char*>(kv_cache.val_bufs[i]), bytes);
  }
  // 포인터 교체 불필요 — 동일 버퍼에 읽기/쓰기
  // MHA가 cache에 write하면 외부 버퍼에 직접 반영 (zero-copy)
}
```

#### 멀티 요청 cache swap
```cpp
// 다른 요청의 cache로 교체할 때
void CausalLM::swapKVCache(KVCacheBuffers &other_cache) {
  for (int i = 0; i < NUM_LAYERS; i++) {
    key_cache_tensors[i].setExternalData(other_cache.key_bufs[i]);
    val_cache_tensors[i].setExternalData(other_cache.val_bufs[i]);
    // → 내부적으로 fillPlaceholder 재호출 + syncDependents
  }
}
```

### 3.6 Cached MoE Layer Variants

**파일**:
- `Applications/CausalLM/layers/qwen_moe_layer_cached.h/cpp`
- `Applications/CausalLM/layers/gpt_oss_moe_layer_cached.h/cpp`

이들도 MHACoreLayer와 동일한 패턴으로 외부 cache 지원 가능.
내부적으로 MHACoreLayer를 호출하는 구조이므로, MHACoreLayer 변경이
자동으로 전파될 수 있음. 필요 시 입력 연결만 변경.

### 3.7 Applications 마이그레이션 우선순위

| 순서 | 대상 | 변경 범위 | 이유 |
|------|------|-----------|------|
| 1 | CausalLM | 큼 | KV cache 직접 관리, 가장 큰 수혜자 |
| 2 | PicoGPT | 작음 | 내장 MHA 사용, 선택적 전환 |
| 3 | LLaMA | 작음 | 커스텀 어텐션, 선택적 전환 |
| 4 | 나머지 | 없음 | INI 기반 또는 Tensor API 미사용 |

---

## Phase 4: TorchFXConverter 업데이트

### 4.1 변경이 필요한 이유

TorchFXConverter는 PyTorch 모델을 nntrainer C++ 코드로 변환한다.
현재 생성되는 코드는:
1. `mha_core` 레이어에 3개 입력(Q, K, V)만 전달
2. KV cache를 레이어 내부에서 관리
3. `max_timestep`, `sliding_window` 등의 프로퍼티로 cache 크기 결정

새 Tensor API에서는 **외부 KV cache를 입력으로 전달하는 옵션**이 추가되므로,
Converter가 이를 선택적으로 생성할 수 있어야 한다.

### 4.2 변경 파일 및 내용

#### 4.2.1 source_attention.py — Attention 코드 생성

**파일**: `Applications/TorchFXConverter/emitter_cpp/source_attention.py`

```python
def emit_attention_method(cname, block, arch_type="decoder_only",
                          use_external_cache=False):  # 새 파라미터
    """Generate createAttention() method body."""
    attn = block.attention
    # ...기존 코드...

    # MHA core 레이어 생성
    mha_props = [
        'withKey("name", A)',
        'withKey("num_heads", n_heads)',
        'withKey("num_heads_kv", n_heads / GQA_SIZE)',
    ]

    if use_external_cache:
        # 외부 KV cache 입력 추가
        # cache 텐서 이름은 constructModel()에서 정의
        mha_props.append(
            'withKey("input_layers", {q_in} + "," + {k_in} + "," + V'
            ' + "," + cache_key_name + "," + cache_value_name)')
        # max_timestep은 외부 cache 크기로 결정되므로 불필요할 수 있음
    else:
        # 기존 방식: 내부 cache
        mha_props.append('withKey("max_timestep", ...)')
        mha_props.append(
            'withKey("input_layers", {q_in} + "," + {k_in} + "," + V)')

    L.extend(_cpp_layer("mha_core", mha_props))
```

#### 4.2.2 source_construct.py — 모델 생성 코드

**파일**: `Applications/TorchFXConverter/emitter_cpp/source_construct.py`

외부 KV cache 모드일 때 `constructModel()`에 cache 텐서 선언 추가:

```python
def emit_transformer_source(layers, structure, model_name=None,
                            use_external_cache=False):
    # ...기존 코드...

    if use_external_cache:
        # 각 레이어별 외부 cache 텐서 선언 생성
        L.append(f"  // External KV cache tensors")
        L.append(f"  for (int i = 0; i < {num_layers}; i++) {{")
        L.append(f"    auto cache_dim = ml::train::TensorDim("
                 f"{{1, 1, MAX_SEQ, NUM_KV_HEADS * HEAD_DIM}});")
        L.append(f'    auto k_name = "layer" + std::to_string(i) '
                 f'+ "_ext_k_cache";')
        L.append(f'    auto v_name = "layer" + std::to_string(i) '
                 f'+ "_ext_v_cache";')
        L.append(f"    key_cache_tensors.push_back("
                 f"nntrainer::Tensor::fromData(cache_dim, "
                 f"key_cache_bufs[i], k_name));")
        L.append(f"    val_cache_tensors.push_back("
                 f"nntrainer::Tensor::fromData(cache_dim, "
                 f"val_cache_bufs[i], v_name));")
        L.append(f"  }}")
```

#### 4.2.3 header.py — 헤더 생성

**파일**: `Applications/TorchFXConverter/emitter_cpp/header.py`

외부 cache 모드일 때 멤버 변수 선언 추가:

```python
def emit_header(layers, structure, model_name=None,
                use_external_cache=False):
    # ...기존 코드...

    if use_external_cache:
        L.append(f"  // External KV cache management")
        L.append(f"  std::vector<float *> key_cache_bufs;")
        L.append(f"  std::vector<float *> val_cache_bufs;")
        L.append(f"  std::vector<nntrainer::Tensor> key_cache_tensors;")
        L.append(f"  std::vector<nntrainer::Tensor> val_cache_tensors;")
        L.append(f"")
        L.append(f"  void allocateKVCache();")
        L.append(f"  void swapKVCache(float **new_key_bufs, "
                 f"float **new_val_bufs);")
```

#### 4.2.4 converter.py — CLI 옵션 추가

**파일**: `Applications/TorchFXConverter/converter.py`

```python
parser.add_argument('--external-kv-cache', action='store_true',
                    help='Generate code with external KV cache management')
```

이 옵션이 활성화되면:
- `emitter_cpp`에 `use_external_cache=True` 전달
- 생성되는 C++ 코드에 외부 cache 관리 코드 포함
- INI 에미터에서는 해당 없음 (프로그래밍 API만 지원)

### 4.3 생성되는 코드 비교

#### 기존 (내부 cache)
```cpp
void Qwen3Model::constructModel() {
  // ...
  auto attn_layers = createAttention(i, seq_len, n_heads, head_dim,
                                     pre_norm_name, pre_norm_name, pre_norm_name);
  for (auto &l : attn_layers) model->addLayer(l);
}

// mha_core에 3개 입력
layers.push_back(createLayer("mha_core", {
  withKey("name", A),
  withKey("num_heads", n_heads),
  withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
  withKey("input_layers", Q_norm + "," + K_norm + "," + V)
}));
```

#### 새 API (외부 cache, --external-kv-cache 옵션)
```cpp
void Qwen3Model::constructModel() {
  allocateKVCache();  // 외부 버퍼 할당 + fromData 텐서 생성
  // ...
  auto attn_layers = createAttention(i, seq_len, n_heads, head_dim,
                                     pre_norm_name, pre_norm_name, pre_norm_name,
                                     key_cache_tensors[i].getName(),
                                     val_cache_tensors[i].getName());
  for (auto &l : attn_layers) model->addLayer(l);
}

// mha_core에 5개 입력
layers.push_back(createLayer("mha_core", {
  withKey("name", A),
  withKey("num_heads", n_heads),
  withKey("input_layers", Q_norm + "," + K_norm + "," + V
          + "," + cache_key_name + "," + cache_value_name)
}));
```

### 4.4 TorchFXConverter 변경 범위 요약

| 파일 | 변경 내용 |
|------|-----------|
| `converter.py` | `--external-kv-cache` CLI 옵션 추가 |
| `emitter_cpp/header.py` | 외부 cache 멤버 변수 선언 |
| `emitter_cpp/source_construct.py` | `allocateKVCache()`, cache 텐서 생성 |
| `emitter_cpp/source_attention.py` | mha_core 5-input 모드 생성 |
| `emitter_ini/` | 변경 없음 (INI는 외부 cache 미지원) |
| `weight_converter.py` | 변경 없음 |
| `patterns/` | 변경 없음 |

---

## 전체 변경 파일 요약

### Core (Phase 1)
| 파일 | 변경 |
|------|------|
| `nntrainer/tensor/tensor.h` | `fromData()`, `setExternalData()`, `isExternal()`, `isMaterialized()`, `external_` 멤버 |
| `nntrainer/tensor/tensor.cpp` | 위 메서드 구현 |
| `nntrainer/tensor/tensor_pool.h` | `requestOrPlaceholder()` |
| `nntrainer/tensor/tensor_pool.cpp` | `requestOrPlaceholder()` 구현 |
| `api/ccapi/include/tensor_api.h` | public API에 `fromData()`, `setExternalData()`, `isExternal()` 추가 |

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
| `Applications/CausalLM/causal_lm.cpp` | `constructModel()`, `save/load_kvcache()`, `run()` |
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

## 사용자 최종 API 흐름

```cpp
using namespace ml::train;

// 1. 심볼릭 텐서 (compile 시 UNIQUE → MemoryPool 할당)
auto tokens = Tensor({1, 1, 1, vocab_dim}, false, Initializer::NONE, "tokens");

// 2. 외부 KV cache (compile 시 PLACEHOLDER → MemoryPool 미할당)
float *my_key_buf = new float[1 * max_seq * head_dim];
float *my_val_buf = new float[1 * max_seq * head_dim];
auto key_cache = Tensor::fromData({1, 1, max_seq, head_dim}, my_key_buf, "key_cache");
auto val_cache = Tensor::fromData({1, 1, max_seq, head_dim}, my_val_buf, "val_cache");

// 3. 모델 구성 (기존 createLayer API 그대로)
auto model = createModel(ModelType::NEURAL_NET);
model->addLayer(createLayer("embedding", {"dim=4096", ...}));
model->addLayer(createLayer("multi_head_attention", {
  "num_heads=32",
  "input_layers=embed,embed,embed,key_cache,val_cache"  // 5-input 모드
}));

model->compile();
model->initialize();
model->allocate();

// 4. 추론 — MHA가 cache에 write → 외부 버퍼에 직접 반영 (zero-copy)
for (int step = 0; step < gen_len; step++) {
  model->incremental_inference(1, {input}, {}, seq_len, step, step+1);
}

// 5. 다른 요청의 cache로 교체 시
key_cache.setExternalData(other_request_key_buf);
val_cache.setExternalData(other_request_val_buf);
// → 내부적으로 fillPlaceholder 재호출 + syncDependents
```

---

## 구현 순서

```
Phase 1: Core Tensor API
  ├── tensor.h/cpp: fromData(), setExternalData(), isExternal()
  ├── tensor_pool: requestOrPlaceholder()
  └── tensor_api.h: public API 확장

Phase 2: Layer 외부 캐시 지원
  ├── multi_head_attention_layer: finalize/forwarding 분기
  ├── mha_core (CausalLM): 동일 패턴
  └── network_graph: finalizeContext PLACEHOLDER 처리

Phase 3: Applications 마이그레이션
  ├── CausalLM: 외부 cache 버퍼 관리로 전환
  ├── PicoGPT: 하위 호환 확인 (변경 없음)
  └── LLaMA: 하위 호환 확인 (변경 없음)

Phase 4: TorchFXConverter
  ├── converter.py: --external-kv-cache 옵션
  ├── source_attention.py: 5-input mha_core
  ├── source_construct.py: cache 할당 코드
  └── header.py: 외부 cache 멤버 선언
```

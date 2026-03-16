# Tensor API with Lazy Evaluation - Implementation Plan

## Design Principle
Tensor 생성 방식이 내부 매핑을 결정:
- `Tensor(dim)` → 심볼릭, compile 시 UNIQUE → MemoryPool 할당
- `Tensor::fromData(dim, ptr)` → 외부 데이터, compile 시 PLACEHOLDER → MemoryPool 미할당, 외부 ptr 직접 사용

기존 TensorPool/MemoryPool 파이프라인을 100% 활용. 새로운 그래프 시스템 없음.

---

## Step 1: Tensor 클래스 확장
**파일**: `nntrainer/tensor/tensor.h`, `nntrainer/tensor/tensor.cpp`

### 추가할 API:
```cpp
class Tensor {
public:
  // 기존 생성자 유지

  // 외부 데이터 바인딩 (compile 시 PLACEHOLDER)
  static Tensor fromData(const TensorDim &dim, float *data,
                         const std::string &name = "");
  static Tensor fromData(const TensorDim &dim, void *data,
                         Tdatatype dtype, const std::string &name = "");

  // 런타임 외부 포인터 교체 (cache swap 용)
  void setExternalData(void *new_data);

  // 상태 확인
  bool isExternal() const;      // fromData로 생성되었는가?
  bool isMaterialized() const;  // 데이터 접근 가능한가?

private:
  bool external_ = false;       // fromData 플래그
  void *external_ptr_ = nullptr; // 외부 원시 포인터 보관
};
```

### 구현 세부사항:
- `fromData()`: Tensor 생성 후 `external_ = true`, `external_ptr_` 설정. 메모리 할당하지 않음. MemoryData를 외부 포인터로 래핑하여 setData() 호출
- `setExternalData()`: `external_ptr_` 갱신 + 새 MemoryData 생성 + itensor->setMemoryData() 호출. 이미 TensorPool에 등록된 경우 fillPlaceholder() 재호출 트리거
- `isExternal()`: `external_` 반환
- `isMaterialized()`: `getData() != nullptr` 반환

---

## Step 2: TensorPool에서 외부 텐서 자동 분류
**파일**: `nntrainer/tensor/tensor_pool.h`, `nntrainer/tensor/tensor_pool.cpp`

### 변경사항:
기존 `placeholder()` 메서드가 이미 정확히 필요한 동작을 제공:
- exec_order = {} (빈 실행 순서)
- TensorLifespan::UNMANAGED
- MemoryPool 할당 안 함

`fillPlaceholder()`도 이미 외부 데이터 바인딩 + syncDependents() 수행.

**추가할 것**: `requestOrPlaceholder()` 헬퍼 메서드
```cpp
// Tensor의 is_external 플래그에 따라 자동으로 request() 또는 placeholder() 호출
Tensor *requestOrPlaceholder(const std::string &name, const TensorDim &dim,
                             const std::vector<unsigned int> &exec_order,
                             TensorLifespan lifespan,
                             const Tensor *source_tensor = nullptr);
```
- `source_tensor->isExternal()` → `placeholder(name, dim)` 호출
- 아닌 경우 → 기존 `request(name, dim, exec_order, lifespan)` 호출

---

## Step 3: MultiHeadAttentionLayer 외부 캐시 지원
**파일**: `nntrainer/layers/multi_head_attention_layer.h`, `nntrainer/layers/multi_head_attention_layer.cpp`

### 3a. INOUT_INDEX 확장
```cpp
enum INOUT_INDEX {
  QUERY = 0,
  KEY = 1,
  VALUE = 2,
  MASK = 3,
  CACHE_KEY = 4,    // 새로 추가
  CACHE_VALUE = 5,  // 새로 추가
  OUTPUT = 0,
  RETURN_ATTENTION_WEIGHT = 1,
};
```

### 3b. 멤버 변수 추가
```cpp
private:
  bool use_external_cache = false;  // 외부 캐시 사용 여부
```

### 3c. finalize() 변경
```cpp
void MultiHeadAttentionLayer::finalize(InitLayerContext &context) {
  auto num_inputs = context.getNumInputs();

  // 기존: 3-4개 입력 (query, key, value, [mask])
  // 변경: 3-6개 입력 (query, key, value, [mask], [cache_key], [cache_value])
  NNTR_THROW_IF(num_inputs < 3 || num_inputs > 6, std::invalid_argument)
    << "Multi head Attention layer needs 3 to 6 inputs";

  const bool provide_attention_mask = (num_inputs == 4 || num_inputs == 6);

  if (num_inputs >= 5) {
    // 외부 캐시가 입력으로 들어온 경우
    // → 별도 requestTensor() 불필요
    // → context.getInput(4), context.getInput(5) 로 접근
    use_external_cache = true;
  } else {
    // 기존 방식: 내부 requestTensor()로 cache 할당
    weight_idx[AttentionParams::cache_key] = context.requestTensor(
      projected_key_dim, "cache_key", Initializer::NONE,
      true, TensorLifespan::MAX_LIFESPAN);
    weight_idx[AttentionParams::cache_value] = context.requestTensor(
      projected_value_dim, "cache_value", Initializer::NONE,
      true, TensorLifespan::MAX_LIFESPAN);
  }
  // 나머지 동일
}
```

### 3d. incremental_forwarding() 변경
```cpp
void MultiHeadAttentionLayer::incremental_forwarding(...) {
  // cache 접근 방식만 분기
  Tensor &cache_key = use_external_cache
    ? context.getInput(INOUT_INDEX::CACHE_KEY)
    : context.getTensor(weight_idx[AttentionParams::cache_key]);
  Tensor &cache_value = use_external_cache
    ? context.getInput(INOUT_INDEX::CACHE_VALUE)
    : context.getTensor(weight_idx[AttentionParams::cache_value]);

  // 이후 로직은 기존과 100% 동일
  // cache_key_step, cached_key 등 getSharedDataTensor 패턴 그대로
}
```

---

## Step 4: NetworkGraph에서 외부 텐서 처리
**파일**: `nntrainer/graph/network_graph.cpp`

### finalizeContext() 수정:
입력 텐서가 외부(isExternal)인 경우 requestInputs() 시 PLACEHOLDER로 처리되도록 함.

기존 `tensor_manager->requestInputs()` 호출 시 입력 소스가 외부 텐서인 경우를 식별하여 `placeholder()` 경로로 분기.

---

## Step 5: 사용자 API 흐름
**최종 사용 예시**:
```cpp
using namespace ml::train;

// 심볼릭 텐서 (UNIQUE → MemoryPool 할당)
auto tokens = Tensor({1, 1, 1, vocab_dim}, false, Initializer::NONE, "tokens");

// 외부 KV cache (PLACEHOLDER → MemoryPool 미할당)
auto key_cache = Tensor::fromData({1, 1, max_seq, head_dim}, my_key_buf, "key_cache");
auto val_cache = Tensor::fromData({1, 1, max_seq, head_dim}, my_val_buf, "val_cache");

// 모델 구성 (기존 createLayer API 그대로)
auto model = createModel(ModelType::NEURAL_NET);
model->addLayer(createLayer("embedding", {"dim=4096"}));
model->addLayer(createLayer("multi_head_attention", {
  "num_heads=32", "projected_key_dim=128"
}), {tokens_name, tokens_name, tokens_name, key_cache_name, val_cache_name});

model->compile();
model->initialize();

// 추론 — MHA가 cache에 write하면 외부 버퍼에 직접 반영 (zero-copy)
for (int step = 0; step < gen_len; step++) {
  model->incremental_inference(step, step+1);
}

// 다른 요청의 cache로 교체
key_cache.setExternalData(other_request_key_buf);
val_cache.setExternalData(other_request_val_buf);
```

---

## 변경 파일 요약
| 파일 | 변경 내용 |
|------|-----------|
| `nntrainer/tensor/tensor.h` | fromData(), setExternalData(), isExternal(), isMaterialized(), external_ 멤버 |
| `nntrainer/tensor/tensor.cpp` | 위 메서드 구현 |
| `nntrainer/tensor/tensor_pool.h` | requestOrPlaceholder() 헬퍼 |
| `nntrainer/tensor/tensor_pool.cpp` | requestOrPlaceholder() 구현 |
| `nntrainer/layers/multi_head_attention_layer.h` | use_external_cache 멤버 |
| `nntrainer/layers/multi_head_attention_layer.cpp` | finalize(), incremental_forwarding() 분기 로직 |
| `nntrainer/graph/network_graph.cpp` | finalizeContext()에서 외부 텐서 PLACEHOLDER 처리 |

## 기존 코드 영향 최소화
- 기존 3-4 입력 MHA는 `use_external_cache = false`로 동작 → 100% 하위 호환
- 기존 Tensor 생성자 변경 없음
- TensorPool의 placeholder()/fillPlaceholder() 재사용
- NetworkGraph 파이프라인 구조 변경 없음

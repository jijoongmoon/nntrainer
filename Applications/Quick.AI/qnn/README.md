# QNN Context 가이드: 커스텀 Context 만들기

## 개요

`QNNContext`는 nntrainer 프레임워크에서 Qualcomm Neural Network (QNN) 백엔드를 관리하는 핵심 클래스입니다. 이 클래스는 다음과 같은 역할을 수행합니다:

- **QNN 백엔드 초기화**: HTP 백엔드 라이브러리(`libQnnHtp.so`) 로드 및 설정
- **레이어 팩토리 관리**: QNN 전용 레이어(QNNLinear, QNNGraph 등)의 생성 팩토리 등록/조회
- **메모리 관리**: RPC 메모리 할당기(`QNNRpcManager`)를 통한 QNN 텐서 메모리 관리
- **바이너리 컨텍스트 로드**: QNN 바이너리 파일(`.bin`)로부터 그래프를 로드하고 실행 준비

이 문서는 `QNNContext`의 구조를 참고하여 **사용자가 자신만의 커스텀 Context를 만드는 방법**을 안내합니다.

---

## 아키텍처 개요

### 클래스 상속 구조

```
Context (context.h)              Singleton<QNNContext> (singleton.h)
    │                                      │
    └──────────────┬───────────────────────┘
                   │
            QNNContext (qnn_context.h)
```

### 데이터 구조

```
ContextData (context.h)
    │
QNNBackendVar (qnn_context_var.h)
    │
    └── QNNVar (qnn_context_var.h)    ← QNN 백엔드 핸들, 디바이스, 컨텍스트 맵 등
```

- `Context`: 레이어/옵티마이저 팩토리와 `ContextData`를 관리하는 베이스 클래스
- `Singleton<T>`: 싱글톤 패턴을 제공하는 템플릿 클래스 (`Global()` 메서드로 접근)
- `ContextData`: 백엔드별 데이터를 저장하는 컨테이너 (메모리 할당기 포함)
- `QNNBackendVar`: `ContextData`를 상속하여 `QNNVar`를 래핑
- `QNNVar`: QNN 백엔드 핸들, 디바이스 핸들, 프로파일링, 바이너리 컨텍스트 맵 등 실제 QNN 상태를 보관

---

## 커스텀 Context 만들기

`QNNContext`를 참고하여 새로운 백엔드 Context를 만드는 단계별 가이드입니다.

### Step 1: ContextData 상속 - 백엔드 데이터 클래스 작성

백엔드에서 사용할 상태와 핸들을 담는 데이터 클래스를 만듭니다.

```cpp
// my_backend_var.h
#include <context.h>

namespace nntrainer {

// 백엔드 고유 상태를 보관하는 구조체
struct MyBackendVar {
  void *backend_handle = nullptr;
  bool is_initialized = false;
  // ... 백엔드별 핸들, 설정 등
};

// ContextData를 상속하여 래핑
class MyBackendData : public ContextData {
public:
  MyBackendData() : data(std::make_shared<MyBackendVar>()) {}
  std::shared_ptr<MyBackendVar> &getVar() { return data; }

private:
  std::shared_ptr<MyBackendVar> data;
};

} // namespace nntrainer
```

> **참고**: `QNNContext`에서는 `QNNBackendVar`가 `ContextData`를 상속하고, 내부에 `QNNVar`를 `shared_ptr`로 보관합니다. (`qnn_context_var.h` 참조)

### Step 2: Context 클래스 상속

`Context`를 상속받아 필수 메서드를 구현합니다. 싱글톤이 필요하면 `Singleton<T>`도 함께 상속합니다.

```cpp
// my_context.h
#include <context.h>
#include "singleton.h"
#include "my_backend_var.h"

namespace nntrainer {

class MyContext : public Context, public Singleton<MyContext> {
public:
  MyContext() : Context(std::make_shared<MyBackendData>()) {}
  ~MyContext();

  // [필수] 백엔드 초기화
  int init() override;

  // [필수] Context 이름 반환 (Engine에서 식별용)
  std::string getName() override { return "my_backend"; }

  // [필수] 레이어 생성 - 문자열 키
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const std::string &type,
                    const std::vector<std::string> &properties) override;

  // [필수] 레이어 생성 - 정수 키
  std::unique_ptr<nntrainer::Layer>
  createLayerObject(const int int_key,
                    const std::vector<std::string> &properties = {}) override;

  // [선택] 모델 바이너리 로드
  int load(const std::string &file_path) override;

private:
  // Singleton에서 호출하는 초기화 메서드
  void initialize() noexcept override;

  // 레이어 팩토리 맵
  FactoryMap<nntrainer::Layer> factory_map;
};

} // namespace nntrainer
```

### Step 3: 필수 메서드 구현

#### `initialize()` - 최초 초기화 및 레이어 등록

`Singleton<T>::Global()` 호출 시 `initializeOnce()`를 통해 한 번만 실행됩니다. 여기서 백엔드를 초기화하고 레이어 팩토리를 등록합니다.

```cpp
// my_context.cpp
#include "my_context.h"
#include <MyCustomLayer.h>

namespace nntrainer {

void MyContext::initialize() noexcept {
  try {
    // 1. 백엔드 초기화
    init();

    // 2. 메모리 할당기 설정 (필요 시)
    setMemAllocator(std::make_shared<MyMemAllocator>());

    // 3. 커스텀 레이어 팩토리 등록
    registerFactory(nntrainer::createLayer<MyCustomLayer>,
                    MyCustomLayer::type,
                    ml::train::LayerType::LAYER_FC);

  } catch (std::exception &e) {
    ml_loge("MyContext initialization failed: %s", e.what());
  }
}
```

> **참고**: `QNNContext::initialize()`에서는 `QNNLinear`, `WeightLayer`, `TensorLayer`, `QNNGraph` 네 가지 레이어를 등록합니다. (`qnn_context.cpp:40-58` 참조)

#### `init()` - 백엔드 세부 초기화

백엔드 라이브러리 로드, 핸들 생성 등 구체적인 초기화 로직을 구현합니다.

```cpp
int MyContext::init() {
  auto backend_data = std::static_pointer_cast<MyBackendData>(getContextData());
  auto var = backend_data->getVar();

  // 백엔드 라이브러리 로드
  // 디바이스 핸들 생성
  // 프로파일링 설정
  // ...

  var->is_initialized = true;
  return 0;
}
```

#### `createLayerObject()` - 레이어 객체 생성

팩토리 맵에서 키로 검색하여 레이어 객체를 생성합니다.

```cpp
std::unique_ptr<nntrainer::Layer>
MyContext::createLayerObject(const std::string &type,
                             const std::vector<std::string> &properties) {
  return createObject<nntrainer::Layer>(type, properties);
}

std::unique_ptr<nntrainer::Layer>
MyContext::createLayerObject(const int int_key,
                             const std::vector<std::string> &properties) {
  return createObject<nntrainer::Layer>(int_key, properties);
}
```

#### `load()` - 모델 바이너리 로드 (선택)

QNN 바이너리 등 백엔드 고유 모델 파일을 로드합니다.

```cpp
int MyContext::load(const std::string &file_path) {
  // 백엔드 고유 바이너리 로드 로직
  return 0;
}
```

### Step 4: 레이어 팩토리 등록

`registerFactory` 메서드를 사용하여 커스텀 레이어를 등록합니다.

```cpp
// 함수 포인터로 등록
registerFactory(nntrainer::createLayer<MyLayer>, "my_layer_type", -1);

// 정수 키를 함께 지정
registerFactory(nntrainer::createLayer<MyLayer>,
                MyLayer::type,
                ml::train::LayerType::LAYER_FC);
```

**파라미터 설명:**

| 파라미터 | 설명 |
|---------|------|
| `factory` | 레이어를 생성하는 팩토리 함수 (`std::unique_ptr<T>(const PropsType&)`) |
| `key` | 문자열 키 (빈 문자열이면 `factory({})->getType()`으로 자동 결정) |
| `int_key` | 정수 키 (`-1`이면 자동 할당) |

> **주의**: 이미 등록된 키를 다시 등록하면 `std::invalid_argument` 예외가 발생합니다.

### Step 5: Pluggable Context로 빌드

`PLUGGABLE` 매크로를 정의하여 공유 라이브러리(`.so`)로 빌드하면, 런타임에 동적으로 로드할 수 있습니다.

```cpp
// my_context.cpp 하단
#ifdef PLUGGABLE
nntrainer::Context *create_my_context() {
  nntrainer::MyContext *ctx = new nntrainer::MyContext();
  ctx->Global();
  return ctx;
}

void destroy_my_context(nntrainer::Context *ct) { delete ct; }

extern "C" {
nntrainer::ContextPluggable ml_train_context_pluggable{
  create_my_context, destroy_my_context};
}
#endif
```

> **핵심**: `extern "C"` 블록에서 `ml_train_context_pluggable` 심볼을 반드시 정의해야 합니다. Engine이 이 심볼을 통해 Context를 로드합니다.

---

## Engine 등록

생성한 Context를 Engine에 등록하여 사용합니다.

```cpp
#include "engine.h"

auto &engine = nntrainer::Engine::Global();

// 공유 라이브러리로 등록 (Pluggable 방식)
engine.registerContext("libmy_context.so", "");

// 등록된 Context 사용
auto *ctx = engine.getRegisteredContext("my_backend");
```

Engine은 `registerContext()`를 호출할 때 공유 라이브러리를 `dlopen`하고, 내부의 `ml_train_context_pluggable` 심볼을 찾아 Context를 생성합니다.

레이어 생성 시 `engine` 속성으로 Context를 지정할 수 있습니다:

```cpp
auto layer = createLayer("my_layer_type", {withKey("engine", "my_backend")});
```

---

## QNNContext 초기화 흐름

`QNNContext`의 `init()` 메서드가 실행되는 내부 흐름입니다. 커스텀 Context 작성 시 참고할 수 있습니다.

```
init()
  │
  ├── 1. 로깅 초기화 (QNN Logger)
  │
  ├── 2. 백엔드 라이브러리 로드
  │     └── dynamicloadutil::getQnnFunctionPointers("libQnnHtp.so")
  │
  ├── 3. 시스템 라이브러리 로드
  │     └── dynamicloadutil::getQnnSystemFunctionPointers("libQnnSystem.so")
  │
  ├── 4. 로그 핸들 생성
  │     └── qnnInterface.logCreate()
  │
  ├── 5. 백엔드 확장(Backend Extensions) 설정
  │     ├── BackendExtensions 생성 (htp_backend_ext_config.json)
  │     └── beforeBackendInitialize() → backendCreate() → afterBackendInitialize()
  │
  ├── 6. 디바이스 생성
  │     └── isDevicePropertySupported() → createDevice()
  │
  ├── 7. 프로파일링 초기화
  │     └── initializeProfiling() (OFF / BASIC / DETAILED)
  │
  └── 8. Op 패키지 등록
        └── registerOpPackages()
```

---

## 바이너리 컨텍스트 로드

`QNNContext::load(file_path)` 호출 시 `QNNVar::makeContext()`가 실행되어 QNN 바이너리를 메모리에 매핑하고 컨텍스트를 생성합니다.

```
load(bin_path)
  │
  └── QNNVar::makeContext(bin_path)
        │
        ├── 1. 기존 컨텍스트 확인 (이미 로드되었으면 스킵)
        │
        ├── 2. 백엔드 확장: beforeCreateFromBinary()
        │
        ├── 3. 바이너리 파일 mmap
        │     └── mmapBinaryFile() → mmap(PROT_READ, MAP_PRIVATE)
        │
        ├── 4. 바이너리 메타데이터 추출
        │     ├── systemContextCreate()
        │     ├── systemContextGetBinaryInfo()
        │     └── copyMetadataToGraphsInfo()
        │
        ├── 5. QNN 컨텍스트 생성
        │     └── contextCreateFromBinary()
        │
        ├── 6. 백엔드 확장: afterCreateFromBinary()
        │
        ├── 7. 그래프 정보 맵 설정
        │     └── setGraphInfoMap() → graph_map에 그래프 이름-정보 매핑
        │
        └── 8. 컨텍스트 맵에 저장
              └── ct_map[bin_path] = context_i
```

이후 `QNNVar::graphRetrieve(bin_path, graphName)`을 호출하여 특정 그래프를 가져올 수 있습니다. 바이너리 파일 하나에 여러 그래프가 포함될 수 있으며, `ct_map`을 통해 바이너리 파일별로 컨텍스트를 관리합니다.

---

## 관련 파일 목록

| 파일 | 설명 |
|------|------|
| `nntrainer/qnn_context.h` | QNNContext 클래스 선언 |
| `nntrainer/qnn_context.cpp` | QNNContext 구현 |
| `nntrainer/context.h` | 베이스 Context 클래스 |
| `nntrainer/qnn/jni/qnn_context_var.h` | QNNVar, QNNBackendVar 데이터 구조체 |
| `nntrainer/engine.h` | Engine 클래스 (Context 등록/관리) |
| `nntrainer/utils/singleton.h` | Singleton 템플릿 |
| `nntrainer/qnn/jni/qnn_rpc_manager.h` | QNN RPC 메모리 관리자 |

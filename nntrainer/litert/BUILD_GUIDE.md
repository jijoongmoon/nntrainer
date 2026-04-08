# LiteRT-LM Context Plugin Build Guide

## Overview

이 가이드는 `liblitert_context.so` (nntrainer GPU2 backend plugin)를 빌드하기 위한
LiteRT-LM C++ 라이브러리 빌드 및 연동 방법을 설명합니다.

## Prerequisites

- Ubuntu 22.04+ (x86_64 또는 arm64)
- Clang 18+ / GCC 13+
- CMake 3.25+ 또는 Bazel 7.6.1
- Android NDK r28b (Android 빌드 시)
- Protobuf compiler (protoc)
- 인터넷 접근 (외부 의존성 다운로드 필요)

## Step 1: 소스 코드 준비

```bash
# LiteRT-LM 클론
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
git checkout v0.10.1  # stable 태그 사용 권장

# LiteRT SDK 클론 (헤더 필요)
git clone https://github.com/google-ai-edge/LiteRT.git ../LiteRT

# Abseil 클론 (헤더 필요)
git clone --branch 20250512.0 https://github.com/abseil/abseil-cpp.git ../abseil-cpp

# Git LFS 파일 가져오기 (prebuilt GPU .so)
git lfs pull --include="prebuilt/*"
```

## Step 2: LiteRT-LM 라이브러리 빌드

### Option A: Bazel (권장)

```bash
# Host (Linux x86_64) 빌드
bazel build //runtime/engine:litert_lm_lib

# Android arm64 빌드
export ANDROID_NDK_HOME=/path/to/android-ndk-r28b
bazel build --config=android_arm64 //runtime/engine:litert_lm_lib

# GPU 지원 포함 시
bazel build --config=android_arm64 \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false \
  //runtime/engine:litert_lm_lib
```

빌드 결과물:
- `bazel-bin/runtime/engine/liblitert_lm_lib.a` (정적 라이브러리)
- 또는 `bazel-bin/runtime/engine/liblitert_lm_lib.so` (동적 라이브러리)

### Option B: CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Android 크로스 컴파일
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-29 \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Step 3: Proto 파일 생성

> **Note:** Bazel 빌드(Step 2-2 Option A)를 사용한 경우 이 단계는 **불필요**합니다.
> Bazel이 proto 파일 생성을 자동으로 처리합니다.
> CMake 빌드 또는 Step 5에서 NDK 직접 컴파일 시에만 필요합니다.

```bash
cd LiteRT-LM
protoc --cpp_out=. --proto_path=. \
  runtime/proto/engine.proto \
  runtime/proto/llm_metadata.proto \
  runtime/proto/llm_model_type.proto \
  runtime/proto/sampler_params.proto \
  runtime/proto/token.proto \
  runtime/executor/proto/constrained_decoding_options.proto
```

## Step 4: build_config.h 생성

LiteRT SDK에서 빌드 시 자동 생성되는 파일이 없으면 빈 파일 생성:

```bash
mkdir -p LiteRT/litert/build_common
echo "// Generated stub" > LiteRT/litert/build_common/build_config.h
```

## Step 5: liblitert_context.so 빌드

### Include paths 설정

```
LITERT_LM_DIR   = /path/to/LiteRT-LM
LITERT_SDK_DIR  = /path/to/LiteRT
ABSEIL_DIR      = /path/to/abseil-cpp
NNTRAINER_DIR   = /path/to/nntrainer
```

### 컴파일 명령

#### Host (Linux x86_64) - 개발/테스트용

```bash
clang++ -std=c++20 -shared -fPIC -DPLUGGABLE -DENABLE_LITERT_LM \
  -Wno-deprecated-declarations \
  -I${NNTRAINER_DIR} \
  -I${NNTRAINER_DIR}/nntrainer \
  -I${NNTRAINER_DIR}/api/ccapi/include \
  -I${LITERT_LM_DIR} \
  -I${LITERT_SDK_DIR} \
  -I${ABSEIL_DIR} \
  ${NNTRAINER_DIR}/nntrainer/litert/litert_context.cpp \
  ${NNTRAINER_DIR}/nntrainer/litert/litert_graph.cpp \
  -L/path/to/litert_lm_lib \
  -llitert_lm_lib \
  -lprotobuf \
  -o liblitert_context.so
```

#### Android arm64 - 실제 배포용

```bash
# NDK 크로스 컴파일러 경로
NDK_TOOLCHAIN=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64
NDK_CC=$NDK_TOOLCHAIN/bin/aarch64-linux-android29-clang++
NDK_SYSROOT=$NDK_TOOLCHAIN/sysroot

$NDK_CC -std=c++20 -shared -fPIC -DPLUGGABLE -DENABLE_LITERT_LM \
  -Wno-deprecated-declarations \
  --sysroot=$NDK_SYSROOT \
  -I${NNTRAINER_DIR} \
  -I${NNTRAINER_DIR}/nntrainer \
  -I${NNTRAINER_DIR}/api/ccapi/include \
  -I${LITERT_LM_DIR} \
  -I${LITERT_SDK_DIR} \
  -I${ABSEIL_DIR} \
  ${NNTRAINER_DIR}/nntrainer/litert/litert_context.cpp \
  ${NNTRAINER_DIR}/nntrainer/litert/litert_graph.cpp \
  -L/path/to/litert_lm_lib_android_arm64 \
  -llitert_lm_lib \
  -lprotobuf \
  -llog \
  -o liblitert_context.so

# 빌드 결과 확인
file liblitert_context.so
# ELF 64-bit LSB shared object, ARM aarch64, ...
```

> **주의:** Host 빌드는 x86_64 바이너리를 생성하므로 Android 디바이스에서 실행할 수 없습니다.
> APK에 포함하려면 반드시 NDK 크로스 컴파일러(`aarch64-linux-android29-clang++`)를 사용하세요.
> LiteRT-LM 라이브러리도 동일하게 `--config=android_arm64`로 빌드된 것을 사용해야 합니다.

### 핵심 컴파일 플래그

| 플래그 | 설명 |
|--------|------|
| `-DPLUGGABLE` | `ContextPluggable ml_train_context_pluggable` 심볼 export |
| `-DENABLE_LITERT_LM` | LiteRT-LM API 호출 코드 활성화 |
| `-shared -fPIC` | 동적 라이브러리 생성 |
| `-std=c++20` | LiteRT-LM이 C++20 필요 |

## Step 6: 필요 파일 배포

### Android 디바이스 배포 파일

```
/data/local/tmp/nntrainer/causallm/
├── libcausallm_api.so          # CausalLM API
├── libnntrainer.so             # nntrainer core
├── liblitert_context.so        # LiteRT-LM context plugin (우리 코드)
├── libLiteRtGpuAccelerator.so  # prebuilt/android_arm64/
├── libLiteRtOpenClAccelerator.so
├── libGemmaModelConstraintProvider.so
├── libLiteRtTopKOpenClSampler.so
└── models/
    └── gemma4-e2b/
        └── gemma-4-E2B-it.litertlm  # HuggingFace에서 다운로드
```

### 모델 다운로드

```bash
# HuggingFace에서 Gemma4-E2B LiteRT-LM 모델 다운로드
pip install huggingface_hub
huggingface-cli download litert-community/gemma-4-E2B-it-litert-lm \
  --local-dir ./models/gemma4-e2b/
```

## Step 7: 동작 검증

### 기본 테스트 (litert_lm_main)

```bash
# prebuilt 바이너리로 먼저 확인
adb push litert_lm_main /data/local/tmp/
adb push prebuilt/android_arm64/*.so /data/local/tmp/
adb push models/gemma4-e2b/*.litertlm /data/local/tmp/

adb shell "cd /data/local/tmp && \
  LD_LIBRARY_PATH=. ./litert_lm_main \
  --backend=cpu \
  --model_path=gemma-4-E2B-it.litertlm \
  --input_prompt='What is AI?'"
```

### nntrainer 연동 테스트 (libcausallm_api.so)

```cpp
#include "causal_lm_api.h"

setModelBasePath("/data/local/tmp/nntrainer/causallm/models/");
ErrorCode err = loadModel(CAUSAL_LM_BACKEND_GPU2,
                          CAUSAL_LM_MODEL_GEMMA4_E2B,
                          CAUSAL_LM_QUANTIZATION_UNKNOWN);
// → Engine::registerContext("liblitert_context.so")
// → context("gpu2")->load("gemma-4-E2B-it.litertlm")

const char *output = nullptr;
err = runModel("What is AI?", &output);
printf("Output: %s\n", output);
```

## Architecture

```
libcausallm_api.so
  │
  ├─ loadModel(GPU2, GEMMA4_E2B)
  │    └─ Engine::registerContext("liblitert_context.so")
  │         └─ LiteRTContext::load("model.litertlm")
  │              └─ litert::lm::EngineFactory::CreateAny(settings)
  │
  └─ runModel("prompt")
       └─ LiteRTContext → Session::GenerateContent(InputText(prompt))
            └─ litert_lm_ext.so 내부에서 전체 추론 실행
```

## Current Implementation Status

### GPU2 코드 경로 (causal_lm_api.cpp)

```
loadModel(GPU2, GEMMA4_E2B, ...)
  │
  ├─ [1] Transformer 생성 안 함 (g_model = nullptr)
  │      GPU2/NPU는 CPU/GPU와 달리 nntrainer Transformer 모델을 사용하지 않음.
  │      loadModel() 함수 초반에서 GPU2/NPU를 먼저 분기하여
  │      Transformer Factory 호출 이전에 처리함.
  │
  ├─ [2] Engine::Global().registerContext("liblitert_context.so")
  │      dlopen으로 liblitert_context.so 로딩
  │      → ml_train_context_pluggable 심볼에서 createfunc 호출
  │      → LiteRTContext 인스턴스 생성
  │      → LiteRTContext::initialize() → LiteRTGraph 레이어 팩토리 등록
  │      → Engine에 "gpu2" 이름으로 context 등록
  │      ※ 이미 등록된 경우 예외 catch하여 무시 (재등록 방지)
  │
  ├─ [3] engine.getRegisteredContext("gpu2")->load(model.litertlm)
  │      LiteRTContext::load() 호출:
  │      #ifdef ENABLE_LITERT_LM:
  │        → ModelAssets::Create(file_path)
  │        → EngineSettings::CreateDefault(assets, Backend::GPU)
  │           GPU 실패 시 Backend::CPU로 fallback
  │        → EngineFactory::CreateAny(settings) → engine_ 생성
  │      #else:
  │        → 에러 리턴 (-1)
  │
  ├─ [4] g_initialized = true, g_loaded_backend = GPU2
  │
  └─ return CAUSAL_LM_ERROR_NONE


runModel("prompt")
  │
  ├─ g_loaded_backend == GPU2 확인
  │
  └─ [현재] placeholder 텍스트 반환
     [TODO] 실제 LiteRT-LM 추론 연결:
        auto &engine = nntrainer::Engine::Global();
        auto *ctx = dynamic_cast<LiteRTContext*>(
            engine.getRegisteredContext("gpu2"));
        auto session = ctx->createSession();
        auto responses = session->GenerateContent(
            {litert::lm::InputText(std::string(prompt))});
        output = responses->GetTexts()[0];


unloadModel()
  │
  └─ g_initialized = false, g_model.reset()
     GPU2/NPU에서는 g_model이 이미 nullptr이므로 정상 동작
     Context 자원은 Engine::release()에서 관리


getPerformanceMetrics()
  │
  ├─ GPU2/NPU → placeholder 메트릭 반환 (init duration만 유효)
  │  [TODO] LiteRT-LM BenchmarkInfo에서 메트릭 추출:
  │    session->GetBenchmarkInfo() → prefill_tok/s, decode_tok/s 등
  │
  └─ CPU/GPU → g_model->getPerformanceMetrics() (기존 동작)
```

### 핵심 빌드 플래그

| 플래그 | 설명 | 적용 대상 |
|--------|------|-----------|
| `-DPLUGGABLE` | `extern "C" ml_train_context_pluggable` 심볼 export | liblitert_context.so |
| `-DENABLE_LITERT_LM` | LiteRT-LM C++ API 호출 코드 활성화 | liblitert_context.so |

**`-DENABLE_LITERT_LM` 없이 빌드하면:**
- `LiteRTContext::load()` → 에러 리턴 (-1)
- `LiteRTContext::init()` → 경고 로그만 출력, 성공(0) 리턴
- `LiteRTGraph::forwarding()` → input을 output으로 복사만 함 (pass-through)

### 남은 TODO (실제 동작을 위해)

1. **runModel() GPU2 실제 추론 연결**
   - 현재: placeholder 텍스트 반환
   - 필요: `LiteRTContext::createSession()` → `session->GenerateContent()` 호출
   - `#ifdef ENABLE_LITERT_LM`으로 감싸야 함
   - `causal_lm_api.cpp`에 `#include "litert_context.h"` 추가 필요

2. **getPerformanceMetrics() GPU2 메트릭**
   - 현재: 0으로 채운 placeholder
   - 필요: `session->GetBenchmarkInfo()` → prefill/decode tok/s 추출
   - `BenchmarkInfo::GetPrefillTurn()`, `GetDecodeTurn()` 사용

3. **liblitert_context.so 실제 빌드**
   - LiteRT-LM 라이브러리 (.a/.so) 빌드 필요 (Bazel/CMake)
   - 헤더: LiteRT-LM + LiteRT SDK + Abseil
   - 프록시 없는 환경에서 빌드해야 함

4. **.litertlm 모델 파일 확보**
   - HuggingFace: `litert-community/gemma-4-E2B-it-litert-lm`
   - `huggingface-cli download` 명령으로 다운로드

## Verified Items

- [x] LiteRT-LM C++ API 헤더 확인 (Engine, EngineFactory, Session, InputText, Responses)
- [x] LiteRT-LM .so 심볼 확인 (ModelAssets::Create, EngineSettings::CreateDefault, EngineFactory::Register 등)
- [x] nntrainer Context plugin 구조 (QNNContext 패턴과 동일)
- [x] litert_context.h/cpp - LiteRT-LM API 연동 코드 작성 완료
- [x] litert_graph.h/cpp - GenerateContent/BenchmarkInfo 연동 코드 작성 완료
- [x] ContextPluggable export (#ifdef PLUGGABLE)
- [x] causal_lm_api.cpp - GPU2 loadModel() Engine::registerContext() 경로 구현
- [x] causal_lm_api.cpp - GPU2 runModel() 분기 구현 (placeholder)
- [x] causal_lm_api.cpp - GPU2 unloadModel() / getPerformanceMetrics() 처리
- [ ] runModel() GPU2 실제 LiteRT-LM 추론 연결 (ENABLE_LITERT_LM 빌드 후)
- [ ] getPerformanceMetrics() GPU2 BenchmarkInfo 연동
- [ ] 실제 컴파일 및 링크 (LiteRT-LM 라이브러리 빌드 필요)
- [ ] .litertlm 모델 로딩 테스트
- [ ] 추론 동작 검증

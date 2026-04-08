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

```bash
# liblitert_context.so 빌드
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

## Verified Items

- [x] LiteRT-LM C++ API 헤더 확인 (Engine, EngineFactory, Session, InputText, Responses)
- [x] LiteRT-LM .so 심볼 확인 (ModelAssets::Create, EngineSettings::CreateDefault, EngineFactory::Register 등)
- [x] nntrainer Context plugin 구조 (QNNContext 패턴과 동일)
- [x] litert_context.h/cpp - LiteRT-LM API 연동 코드 작성 완료
- [x] litert_graph.h/cpp - GenerateContent/BenchmarkInfo 연동 코드 작성 완료
- [x] ContextPluggable export (#ifdef PLUGGABLE)
- [ ] 실제 컴파일 및 링크 (LiteRT-LM 라이브러리 빌드 필요)
- [ ] .litertlm 모델 로딩 테스트
- [ ] 추론 동작 검증

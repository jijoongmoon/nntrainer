# Quick.AI Service - Build & Run Guide

현재 브랜치 (`claude/merge-pr-3849-Sld3Y`)에서 APK를 빌드하고 Android 디바이스에서
구동하기 위한 전체 순서입니다.

## Prerequisites

- Android NDK r28b+ (ndk-build + LiteRT-LM 빌드)
- Android SDK (API 35)
- Bazel 7.6.1 (LiteRT-LM 빌드용)
- Gradle 8.7+
- Android 디바이스 (arm64-v8a, Android 10+, Snapdragon 8 Gen3+ 권장)
- adb 연결 확인

```bash
# 환경변수 설정
export ANDROID_NDK_HOME=/path/to/android-ndk-r28b
export ANDROID_HOME=/path/to/android-sdk
export PATH=$PATH:$ANDROID_HOME/platform-tools
```

## Step 1: nntrainer 네이티브 라이브러리 빌드

CausalLM 앱의 기존 빌드 스크립트를 사용하여 arm64-v8a 네이티브 라이브러리를 빌드합니다.

```bash
cd Applications/CausalLM

# Step 1-1: 코어 라이브러리 빌드
# → libnntrainer.so, libccapi-nntrainer.so, libcausallm_core.so
./build_android.sh

# Step 1-2: API 라이브러리 빌드
# → libcausallm_api.so
./build_api_lib.sh
```

빌드 결과물 확인:
```bash
ls jni/libs/arm64-v8a/
# libcausallm_api.so
# libcausallm_core.so
# libnntrainer.so
# libccapi-nntrainer.so
# nntrainer_causallm (실행파일)
```

## Step 2: LiteRT-LM 라이브러리 빌드 (GPU2 백엔드)

GPU2 백엔드(Gemma4-E2B)를 사용하려면 LiteRT-LM 라이브러리와 liblitert_context.so를
빌드해야 합니다. CPU 백엔드만 사용할 경우 이 단계를 건너뛸 수 있습니다.

### Step 2-1: LiteRT-LM 소스 준비

```bash
# LiteRT-LM 클론
git clone https://github.com/google-ai-edge/LiteRT-LM.git
cd LiteRT-LM
git checkout v0.10.1

# Git LFS로 prebuilt GPU .so 다운로드
git lfs pull --include="prebuilt/android_arm64/*"

# LiteRT SDK 클론 (헤더 필요)
git clone https://github.com/google-ai-edge/LiteRT.git ../LiteRT

# Abseil 클론 (헤더 필요)
git clone --branch 20250512.0 https://github.com/abseil/abseil-cpp.git ../abseil-cpp
```

### Step 2-2: LiteRT-LM 엔진 라이브러리 빌드

```bash
cd LiteRT-LM

# Android arm64 빌드 (Bazel)
bazel build --config=android_arm64 //runtime/engine:litert_lm_lib

# GPU 지원 포함 시
bazel build --config=android_arm64 \
  --define=litert_link_capi_so=true \
  --define=resolve_symbols_in_exec=false \
  //runtime/engine:litert_lm_lib
```

### Step 2-3: Proto 파일 생성

```bash
cd LiteRT-LM
protoc --cpp_out=. --proto_path=. \
  runtime/proto/engine.proto \
  runtime/proto/llm_metadata.proto \
  runtime/proto/llm_model_type.proto \
  runtime/proto/sampler_params.proto \
  runtime/proto/token.proto \
  runtime/executor/proto/constrained_decoding_options.proto

# LiteRT SDK build_config.h (빌드 시 자동 생성 안 된 경우)
mkdir -p ../LiteRT/litert/build_common
echo "// Generated stub" > ../LiteRT/litert/build_common/build_config.h
```

### Step 2-4: liblitert_context.so 빌드

```bash
NNTRAINER_DIR=/path/to/nntrainer
LITERT_LM_DIR=/path/to/LiteRT-LM
LITERT_SDK_DIR=/path/to/LiteRT
ABSEIL_DIR=/path/to/abseil-cpp

# Android arm64 크로스 컴파일
$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ \
  -std=c++20 -shared -fPIC -DPLUGGABLE -DENABLE_LITERT_LM \
  -Wno-deprecated-declarations \
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
  -o liblitert_context.so
```

빌드 결과물 확인:
```bash
ls -lh liblitert_context.so
file liblitert_context.so
# ELF 64-bit LSB shared object, ARM aarch64, ...
```

> 상세 빌드 옵션은 `nntrainer/litert/BUILD_GUIDE.md` 참조

## Step 3: .so 파일을 Service APK로 복사

```bash
# 대상 디렉토리
JNILIBS=Applications/QuickAI/service-app/app/src/main/jniLibs/arm64-v8a
mkdir -p $JNILIBS

# === 필수: CausalLM 빌드 결과물 (CPU 백엔드) ===
cp Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_api.so    $JNILIBS/
cp Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_core.so   $JNILIBS/
cp Applications/CausalLM/jni/libs/arm64-v8a/libnntrainer.so       $JNILIBS/
cp Applications/CausalLM/jni/libs/arm64-v8a/libccapi-nntrainer.so $JNILIBS/

# === 필수: NDK C++ 런타임 ===
cp $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $JNILIBS/

# === GPU2 백엔드 (LiteRT-LM) - Step 2를 수행한 경우 ===
cp liblitert_context.so $JNILIBS/

# LiteRT-LM prebuilt GPU 라이브러리
LITERT_PREBUILT=LiteRT-LM/prebuilt/android_arm64
cp $LITERT_PREBUILT/libLiteRtGpuAccelerator.so          $JNILIBS/
cp $LITERT_PREBUILT/libLiteRtOpenClAccelerator.so        $JNILIBS/
cp $LITERT_PREBUILT/libLiteRtTopKOpenClSampler.so        $JNILIBS/
cp $LITERT_PREBUILT/libGemmaModelConstraintProvider.so   $JNILIBS/

# 확인
ls -lh $JNILIBS/
# 필수 (5개): libcausallm_api.so, libcausallm_core.so, libnntrainer.so,
#              libccapi-nntrainer.so, libc++_shared.so
# GPU2 (5개): liblitert_context.so, libLiteRtGpuAccelerator.so,
#              libLiteRtOpenClAccelerator.so, libLiteRtTopKOpenClSampler.so,
#              libGemmaModelConstraintProvider.so
```

## Step 4: local.properties 설정

```bash
cd Applications/QuickAI

# Android SDK/NDK 경로 설정
cat > local.properties << EOF
sdk.dir=$ANDROID_HOME
ndk.dir=$ANDROID_NDK_HOME
EOF
```

## Step 5: APK 빌드

```bash
cd Applications/QuickAI

# Service APK 빌드
./gradlew :service-app:app:assembleDebug

# Client APK 빌드
./gradlew :client-app:app:assembleDebug

# 빌드 결과물 확인
ls -lh service-app/app/build/outputs/apk/debug/app-debug.apk
ls -lh client-app/app/build/outputs/apk/debug/app-debug.apk
```

## Step 6: 모델 파일 준비

### CPU 백엔드: Qwen3-0.6B

```bash
# 모델 디렉토리 구조:
#   qwen3-0.6b-w4a32/
#   ├── qwen3-0.6b-q40-fp32-arm.bin  (가중치 파일)
#   ├── tokenizer.json
#   ├── config.json (외부 설정 사용 시)
#   └── nntr_config.json (외부 설정 사용 시)
```

### GPU2 백엔드: Gemma4-E2B (LiteRT-LM)

```bash
# HuggingFace에서 모델 다운로드
pip install huggingface_hub
huggingface-cli download litert-community/gemma-4-E2B-it-litert-lm \
  --local-dir ./gemma4-e2b/

# 모델 디렉토리 구조:
#   gemma4-e2b/
#   └── gemma-4-E2B-it-litert-lm.task  (.litertlm 모델 파일)
```

### 디바이스 배포 경로

```
/data/data/com.quickai.service/files/models/
├── qwen3-0.6b-w4a32/          ← CPU 백엔드
│   ├── qwen3-0.6b-q40-fp32-arm.bin
│   └── tokenizer.json
├── gemma4-e2b/                 ← GPU2 백엔드
│   └── gemma-4-E2B-it-litert-lm.task
└── liblitert_context.so        ← GPU2 context plugin (Step 2에서 빌드)
```

## Step 7: APK 설치

```bash
# Service APK 설치
adb install -r service-app/app/build/outputs/apk/debug/app-debug.apk

# Client APK 설치
adb install -r client-app/app/build/outputs/apk/debug/app-debug.apk
```

## Step 8: 모델 파일을 앱 디렉토리에 배포

```bash
APP_FILES=/data/data/com.quickai.service/files
MODELS=$APP_FILES/models

# CPU 모델: Qwen3-0.6B
adb push qwen3-0.6b-w4a32/ /data/local/tmp/qwen3-0.6b-w4a32/
adb shell "run-as com.quickai.service mkdir -p $MODELS/qwen3-0.6b-w4a32"
adb shell "run-as com.quickai.service cp -r /data/local/tmp/qwen3-0.6b-w4a32/* $MODELS/qwen3-0.6b-w4a32/"

# GPU2 모델: Gemma4-E2B (Step 2를 수행한 경우)
adb push gemma4-e2b/ /data/local/tmp/gemma4-e2b/
adb shell "run-as com.quickai.service mkdir -p $MODELS/gemma4-e2b"
adb shell "run-as com.quickai.service cp -r /data/local/tmp/gemma4-e2b/* $MODELS/gemma4-e2b/"

# GPU2 context plugin을 모델 디렉토리에 배포
# (loadModel에서 g_model_base_path + "liblitert_context.so" 경로로 로딩)
adb push liblitert_context.so /data/local/tmp/
adb shell "run-as com.quickai.service cp /data/local/tmp/liblitert_context.so $MODELS/"
```

## Step 9: 실행 및 테스트

### Service 앱 실행
1. 디바이스에서 "Quick.AI Service" 앱 실행
2. "Start Service" 버튼 터치
3. 알림바에 "LLM service running on port 8080" 확인

### Client 앱으로 테스트
1. 디바이스에서 "Quick.AI Client" 앱 실행
2. **API Test** 탭 → "GET /v1/health" → `{"status":"ok"}` 확인
3. **Models** 탭 → "qwen3-0.6b" 선택 → Backend "cpu" → "Load" 터치
4. **Chat** 탭 → 프롬프트 입력 → Send

### adb로 직접 REST API 테스트

```bash
# 디바이스 포트 포워딩
adb forward tcp:8080 tcp:8080

# Health check
curl http://localhost:8080/v1/health

# 모델 목록
curl http://localhost:8080/v1/models

# 모델 로드 (CPU, Qwen3-0.6B, W4A32)
curl -X POST http://localhost:8080/v1/engine/load \
  -H "Content-Type: application/json" \
  -d '{"backend":"cpu", "model_id":"qwen3-0.6b", "quant_type":1}'

# 텍스트 생성
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is artificial intelligence?", "use_chat_template":true}'

# 성능 메트릭
curl http://localhost:8080/v1/metrics

# 모델 언로드
curl -X POST http://localhost:8080/v1/engine/unload

# === GPU2 백엔드 테스트 (LiteRT-LM + Gemma4-E2B) ===

# 모델 로드 (GPU2)
curl -X POST http://localhost:8080/v1/engine/load \
  -H "Content-Type: application/json" \
  -d '{"backend":"gpu2", "model_id":"gemma4-e2b", "quant_type":0}'

# 텍스트 생성
curl -X POST http://localhost:8080/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello, what can you do?", "use_chat_template":true}'

# 언로드
curl -X POST http://localhost:8080/v1/engine/unload
```

## Troubleshooting

### "Service not starting"
- logcat 확인: `adb logcat -s LlmService QuickAI`
- 권한 확인: Android 13+ 에서 알림 권한 필요

### "Model load failed"
- 모델 파일 경로 확인: `adb shell "run-as com.quickai.service ls files/models/"`
- logcat에서 에러 확인: `adb logcat | grep causallm`

### "libcausallm_api.so not found"
- jniLibs 디렉토리에 .so 파일이 있는지 확인
- `adb shell "run-as com.quickai.service ls lib/arm64/"` 로 설치된 라이브러리 확인

### Client 앱에서 연결 실패
- Service 앱이 실행 중인지 확인
- 같은 디바이스에서 실행 중인지 확인 (localhost:8080)
- `adb forward tcp:8080 tcp:8080` (PC에서 테스트 시)

## Architecture

```
[Client App]                    [Service App]
     │                               │
     │  HTTP (localhost:8080)         │
     ├──────────────────────────────►│
     │                               │
     │                          LlmHttpServer (NanoHTTPD)
     │                               │
     │                          NativeEngine.kt
     │                               │ JNI
     │                          native_engine_jni.cpp
     │                               │
     │                          libcausallm_api.so
     │                               │
     │                 ┌─────────────┼──────────────┐
     │                 │             │              │
     │              [CPU]         [NPU]          [GPU2]
     │            AppContext    QNNContext     LiteRTContext
     │                │             │              │
     │          libnntrainer.so  libqnn_      liblitert_
     │                │          context.so    context.so
     │          libcausallm_        │              │
     │           core.so       QNNGraph       LiteRTGraph
     │                │        Layer          Layer
     │          Transformer       │              │
     │          decode loop   QNN binary    LiteRT-LM
     │                        execution      Engine
     │                                    (end-to-end)
```

### GPU2 (LiteRT-LM) 로딩 흐름

```
loadModel(GPU2, GEMMA4_E2B)
  │
  ├─ Engine::registerContext("liblitert_context.so")
  │    └─ dlopen → ml_train_context_pluggable 심볼 로드
  │    └─ create_litert_context() → LiteRTContext 생성
  │    └─ LiteRTContext::initialize() → LiteRTGraph 레이어 팩토리 등록
  │    └─ Engine에 "gpu2" context로 등록
  │
  └─ Engine::getRegisteredContext("gpu2")->load("gemma4-e2b.litertlm")
       └─ LiteRTContext::load()
            └─ ModelAssets::Create(path)
            └─ EngineSettings::CreateDefault(assets, Backend::GPU)
            └─ EngineFactory::CreateAny(settings) → Engine 생성 완료

runModel("Hello")
  │
  └─ LiteRTContext::createSession()
       └─ engine_->CreateSession(SessionConfig::CreateDefault())
       └─ session->GenerateContent(InputText("Hello"))
       └─ responses->GetTexts()[0] → 생성된 텍스트 반환
```

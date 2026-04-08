# Quick.AI Service - Build & Run Guide

현재 브랜치 (`claude/merge-pr-3849-Sld3Y`)에서 APK를 빌드하고 Android 디바이스에서
구동하기 위한 전체 순서입니다.

## Prerequisites

- Android NDK r26+ (ndk-build 사용)
- Android SDK (API 35)
- Gradle 8.7+
- Android 디바이스 (arm64-v8a, Android 10+)
- adb 연결 확인

```bash
# 환경변수 설정
export ANDROID_NDK_HOME=/path/to/android-ndk
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

## Step 2: .so 파일을 Service APK로 복사

```bash
# 대상 디렉토리
JNILIBS=Applications/QuickAI/service-app/app/src/main/jniLibs/arm64-v8a

# CausalLM 빌드 결과물 복사
cp Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_api.so   $JNILIBS/
cp Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_core.so  $JNILIBS/
cp Applications/CausalLM/jni/libs/arm64-v8a/libnntrainer.so      $JNILIBS/
cp Applications/CausalLM/jni/libs/arm64-v8a/libccapi-nntrainer.so $JNILIBS/

# NDK C++ 런타임 복사
cp $ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so $JNILIBS/

# 확인
ls -lh $JNILIBS/
```

## Step 3: local.properties 설정

```bash
cd Applications/QuickAI

# Android SDK/NDK 경로 설정
cat > local.properties << EOF
sdk.dir=$ANDROID_HOME
ndk.dir=$ANDROID_NDK_HOME
EOF
```

## Step 4: APK 빌드

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

## Step 5: 모델 파일 준비

디바이스에 모델 파일을 미리 배포합니다.

```bash
# Qwen3-0.6B 모델 (CPU 백엔드)
# HuggingFace 또는 별도 경로에서 다운로드한 모델 디렉토리 구조:
#   qwen3-0.6b-w4a32/
#   ├── qwen3-0.6b-q40-fp32-arm.bin
#   ├── tokenizer.json
#   ├── config.json (외부 설정 사용 시)
#   └── nntr_config.json (외부 설정 사용 시)

# 디바이스 경로:
# /data/data/com.quickai.service/files/models/qwen3-0.6b-w4a32/
```

## Step 6: APK 설치

```bash
# Service APK 설치
adb install -r service-app/app/build/outputs/apk/debug/app-debug.apk

# Client APK 설치
adb install -r client-app/app/build/outputs/apk/debug/app-debug.apk
```

## Step 7: 모델 파일을 앱 디렉토리에 배포

```bash
# Service 앱의 내부 저장소에 모델 복사
MODEL_DIR=/data/data/com.quickai.service/files/models/qwen3-0.6b-w4a32

adb shell "run-as com.quickai.service mkdir -p $MODEL_DIR"

# adb push는 앱 내부 디렉토리에 직접 못 쓰므로 tmp 경유
adb push qwen3-0.6b-w4a32/ /data/local/tmp/qwen3-0.6b-w4a32/
adb shell "run-as com.quickai.service cp -r /data/local/tmp/qwen3-0.6b-w4a32/* $MODEL_DIR/"
```

## Step 8: 실행 및 테스트

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
     │                          libcausallm_core.so
     │                               │
     │                          libnntrainer.so
     │                               │
     │                          [CPU] nntrainer inference
     │                          [NPU] libqnn_context.so (TODO)
     │                          [GPU2] liblitert_context.so (TODO)
```

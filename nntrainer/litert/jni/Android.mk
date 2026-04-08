LOCAL_PATH := $(call my-dir)

# ndk path
ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

ifndef LITERT_LM_ROOT
$(error LITERT_LM_ROOT is not defined! Set to LiteRT-LM source root.)
endif

ifndef LITERT_SDK_ROOT
$(error LITERT_SDK_ROOT is not defined! Set to LiteRT SDK source root.)
endif

ifndef ABSEIL_ROOT
$(error ABSEIL_ROOT is not defined! Set to abseil-cpp source root.)
endif

# Protobuf headers - /usr/include 직접 사용하면 호스트 glibc 헤더 충돌
# google/protobuf 헤더만 별도 디렉토리에 복사하여 사용
ifndef PROTOBUF_INCLUDES
PROTOBUF_INCLUDES := $(LOCAL_PATH)/protobuf_headers
endif

ML_API_COMMON_INCLUDES := $(NNTRAINER_ROOT)/ml_api_common/include
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
	$(NNTRAINER_ROOT)/nntrainer/dataset \
	$(NNTRAINER_ROOT)/nntrainer/models \
	$(NNTRAINER_ROOT)/nntrainer/layers \
	$(NNTRAINER_ROOT)/nntrainer/compiler \
	$(NNTRAINER_ROOT)/nntrainer/graph \
	$(NNTRAINER_ROOT)/nntrainer/optimizers \
	$(NNTRAINER_ROOT)/nntrainer/tensor \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
	$(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
	$(NNTRAINER_ROOT)/nntrainer/utils \
	$(NNTRAINER_ROOT)/api \
	$(NNTRAINER_ROOT)/api/ccapi/include \
	$(ML_API_COMMON_INCLUDES)

####################################################################
# Prebuilt: nntrainer
####################################################################
include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

####################################################################
# Prebuilt: ccapi-nntrainer
####################################################################
include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

####################################################################
# Prebuilt shared: litert_lm_lib (LiteRT-LM engine)
# Bazel: bazel-bin/runtime/engine/liblitert_lm_lib.so
# .so를 사용해야 Bazel 의존성(SessionConfig 등)이 모두 포함됨
# .a는 thin archive라 직접 소스만 포함, deps 심볼 누락
####################################################################
include $(CLEAR_VARS)
LOCAL_MODULE := litert_lm_lib

ifndef LITERT_LM_LIB_PATH
LITERT_LM_LIB_PATH := $(LITERT_LM_ROOT)/bazel-bin/runtime/engine
endif

LOCAL_SRC_FILES := $(LITERT_LM_LIB_PATH)/liblitert_lm_lib.so
LOCAL_EXPORT_C_INCLUDES := $(LITERT_LM_ROOT) $(LITERT_SDK_ROOT) $(ABSEIL_ROOT) $(PROTOBUF_INCLUDES)
include $(PREBUILT_SHARED_LIBRARY)

####################################################################
# liblitert_context.so - LiteRT-LM context plugin for nntrainer
#
# litert_lm_lib.a (protobuf 포함) → liblitert_context.so
# 배포 시 .a 파일 불필요, .so 하나에 모두 포함됨
####################################################################
include $(CLEAR_VARS)

LOCAL_MODULE        := litert_context
LOCAL_MODULE_TAGS   := optional
LOCAL_ARM_NEON      := true
LOCAL_ARM_MODE      := arm

LOCAL_SRC_FILES     := \
	$(NNTRAINER_ROOT)/nntrainer/litert/litert_context.cpp \
	$(NNTRAINER_ROOT)/nntrainer/litert/litert_graph.cpp

LOCAL_C_INCLUDES    := \
	$(NNTRAINER_INCLUDES) \
	$(NNTRAINER_ROOT)/nntrainer/litert \
	$(LITERT_LM_ROOT) \
	$(LITERT_SDK_ROOT) \
	$(ABSEIL_ROOT) \
	$(PROTOBUF_INCLUDES) \
	$(LITERT_LM_ROOT)/LiteRT

LOCAL_CFLAGS        += -pthread -fexceptions -Wno-deprecated-declarations
LOCAL_CXXFLAGS      += -std=c++17 -frtti -fexceptions -DPLUGGABLE -DENABLE_LITERT_LM
LOCAL_LDLIBS        := -llog -landroid
LOCAL_LDFLAGS       += "-Wl,-z,max-page-size=16384"

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer litert_lm_lib

include $(BUILD_SHARED_LIBRARY)

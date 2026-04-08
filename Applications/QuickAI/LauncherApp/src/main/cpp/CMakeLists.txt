# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# QuickAI JNI shim build
#
# Produces libquickai_jni.so, which forwards JNI calls from Kotlin
# (NativeCausalLm.kt) to the C entry points in Applications/CausalLM/api/
# causal_lm_api.h. The causal_lm_api.so itself (plus its transitive
# dependencies) is shipped as a prebuilt shared library in
# LauncherApp/src/main/jniLibs/${ANDROID_ABI}/ — see build_api_lib.sh in
# Applications/CausalLM to regenerate it.
cmake_minimum_required(VERSION 3.22.1)
project(quickai_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to the CausalLM headers so we can #include "causal_lm_api.h".
# The NDK build is invoked from LauncherApp/; the CausalLM sources live
# four directories up (..  /.. /.. /.. /CausalLM/api).
set(CAUSALLM_API_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../CausalLM/api")

# Register the prebuilt causal_lm_api shared library so the linker can
# resolve its symbols. It's shipped in jniLibs/${ANDROID_ABI}/.
add_library(causallm_api SHARED IMPORTED)
set_target_properties(causallm_api PROPERTIES
    IMPORTED_LOCATION
        "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libcausallm_api.so")

add_library(quickai_jni SHARED
    quickai_jni.cpp)

target_include_directories(quickai_jni PRIVATE
    ${CAUSALLM_API_DIR})

find_library(log-lib log)

target_link_libraries(quickai_jni
    causallm_api
    ${log-lib})

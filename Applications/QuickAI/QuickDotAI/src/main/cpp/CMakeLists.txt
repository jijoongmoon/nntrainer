# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# QuickDotAI JNI shim build
#
# Produces libquickai_jni.so, which forwards JNI calls from Kotlin
# (com.example.quickdotai.NativeCausalLm) to the C entry points in
# quick_dot_ai_api.h. The causal_lm_api.so itself (plus its transitive
# dependencies libnntrainer.so, libccapi-nntrainer.so, libcausallm_core.so,
# libc++_shared.so) is shipped as a set of prebuilt shared libraries in
# QuickDotAI/prebuilt_libs/ — see build_api_lib.sh in Applications/CausalLM
# to regenerate them. API headers are bundled in src/main/cpp/include/ so
# the AAR module is fully self-contained. A Gradle copy task in
# QuickDotAI/build.gradle.kts copies the prebuilt .so files into the AAR's
# jniLibs/arm64-v8a at build time, so they are discoverable at runtime via
# System.loadLibrary from whichever app hosts the AAR.
cmake_minimum_required(VERSION 3.22.1)
project(quickai_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to the QuickDotAI API headers so we can #include "quick_dot_ai_api.h".
# Bundled locally in src/main/cpp/include/ so the AAR is self-contained.
set(CAUSALLM_API_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Path to the prebuilt native libraries. Layout is flat:
#   QuickDotAI/prebuilt_libs/lib*.so
# (only arm64-v8a is supported at the moment — see build_api_lib.sh).
set(PREBUILT_LIBS_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../prebuilt_libs")

# Register the prebuilt causal_lm_api shared library so the linker can
# resolve its symbols when linking libquickai_jni.so.
add_library(quick_dot_ai_api SHARED IMPORTED)
set_target_properties(quick_dot_ai_api PROPERTIES
    IMPORTED_LOCATION
        "${PREBUILT_LIBS_DIR}/libquick_dot_ai_api.so")

add_library(quickai_jni SHARED
    quickai_jni.cpp)

target_include_directories(quickai_jni PRIVATE
    ${CAUSALLM_API_DIR})

find_library(log-lib log)

target_link_libraries(quickai_jni
    quick_dot_ai_api
    ${log-lib})

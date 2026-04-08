cmake_minimum_required(VERSION 3.22)
project(quickai_jni)

# JNI wrapper that bridges NativeEngine.kt to libcausallm_api.so
add_library(quickai_jni SHARED native_engine_jni.cpp)

# CausalLM API headers
target_include_directories(quickai_jni PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../CausalLM/api
)

# Link against pre-built libcausallm_api.so from jniLibs
find_library(CAUSALLM_API_LIB causallm_api
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}
    NO_DEFAULT_PATH
)

target_link_libraries(quickai_jni
    ${CAUSALLM_API_LIB}
    log
)

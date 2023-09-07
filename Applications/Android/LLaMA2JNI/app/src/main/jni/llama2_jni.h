// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   llama2_jni.h
 * @date   07 Sept 2023
 * @todo   move llama2 model creating to separate sourcefile
 * @brief  task runner for the llama2
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seung Back Hong <sb92.hong@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <jni.h>

#ifndef _Included_com_applications_llama2_MainActivity
#define _Included_com_applications_llama2_MainActivity
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create Model
 *
 * @param jint number of output ( number of classes )
 * @return model pointer
 */
JNIEXPORT jlong JNICALL Java_com_applications_llama2_MainActivity_createLLaMA2(
  JNIEnv *, jobject, jstring);

/**
 * @brief Inference Model
 * @param jlong Model pointer
 * @param jobeject bmp from android java
 * @return string inference result
 */
JNIEXPORT jstring JNICALL
Java_com_applications_llama2_MainActivity_inferLLaMA2(JNIEnv *, jobject,
                                                           jstring, jstring,
                                                           jlong);
/**
 * @brief check model destoryed
 * @return bool true if model is destoryed successfully
 */
JNIEXPORT jboolean JNICALL
Java_com_applications_llama2_MainActivity_modelDestroyed(JNIEnv *, jobject);

JNIEXPORT jstring JNICALL
Java_com_applications_llama2_MainActivity_getInferResult(JNIEnv *,
							     jobject);

#ifdef __cplusplus
}
#endif
#endif

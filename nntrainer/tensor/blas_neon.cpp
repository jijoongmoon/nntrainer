// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   blas_neon.cpp
 * @date   4 Aug 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Source for blas neon implementation
 *
 */

#include <blas_neon.h>
#include <iostream>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer::neon {

void sgemv_neon(const float *A, const float *X, float *Y, uint32_t rows,
                uint32_t cols, float alpha, float beta) {
  const float *__restrict x;

  for (unsigned int i = 0; i < rows; ++i) {
    Y[i] = Y[i] * beta;
  }

  float32x4_t v_alpha = vmovq_n_f32(alpha);

  if (cols % 16 == 0) {
    for (unsigned i = 0; i < cols; i += 16) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);
      float32x4_t x4_7 = vld1q_f32(&X[i + 4]);
      float32x4_t x8_11 = vld1q_f32(&X[i + 8]);
      float32x4_t x12_15 = vld1q_f32(&X[i + 12]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
        x4_7 = vmulq_f32(x4_7, v_alpha);
        x8_11 = vmulq_f32(x8_11, v_alpha);
        x12_15 = vmulq_f32(x12_15, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);
        wvec8_11 = vld1q_f32(&w[8]);
        wvec12_15 = vld1q_f32(&w[12]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);
        y0 = vmlaq_f32(y0, wvec4_7, x4_7);
        y0 = vmlaq_f32(y0, wvec8_11, x8_11);
        y0 = vmlaq_f32(y0, wvec12_15, x12_15);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }

  } else if (cols % 8 == 0) {
    for (unsigned i = 0; i < cols; i += 8) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);
      float32x4_t x4_7 = vld1q_f32(&X[i + 4]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
        x4_7 = vmulq_f32(x4_7, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);
        y0 = vmlaq_f32(y0, wvec4_7, x4_7);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }
  } else if (cols % 4 == 0) {
    for (unsigned i = 0; i < cols; i += 4) {
      float32x4_t x0_3 = vld1q_f32(&X[i]);

      if (alpha != 1.0) {
        x0_3 = vmulq_f32(x0_3, v_alpha);
      }

      float32x4_t wvec0_3, wvec4_7;

      const float *__restrict w;

      float32x4_t y0;

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f32(0);

        float r[4];
        wvec0_3 = vld1q_f32(&w[0]);

        y0 = vmlaq_f32(y0, wvec0_3, x0_3);

        vst1q_f32(r, y0);
        for (unsigned int k = 0; k < 4; ++k) {
          Y[j] = Y[j] + r[k];
        }
      }
    }
  }
}

void sgemv_transpose_neon(const float *A, const float *X, float *Y,
                          uint32_t rows, uint32_t cols, float alpha,
                          float beta) {
  const float *__restrict x;

  const float32x4_t v_beta = vdupq_n_f32(beta);
  const float32x4_t v_alpha = vdupq_n_f32(alpha);

  if (cols % 16 == 0) {
    unsigned int n = cols / 16;
    bool *initialized = (bool *)malloc(sizeof(bool) * n);

    unsigned int step;
    for (unsigned int i = 0; i < cols / 16; ++i) {
      initialized[i] = false;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 16) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        float32x4_t y4_7 = vld1q_f32(&y[4]);
        float32x4_t y8_11 = vld1q_f32(&y[8]);
        float32x4_t y12_15 = vld1q_f32(&y[12]);
        step = j / 16;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          y4_7 = vmulq_f32(y4_7, v_beta);
          y8_11 = vmulq_f32(y8_11, v_beta);
          y12_15 = vmulq_f32(y12_15, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);
        wvec8_11 = vld1q_f32(&w[8]);
        wvec12_15 = vld1q_f32(&w[12]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        y4_7 = vmlaq_f32(y4_7, wvec4_7, x);
        y8_11 = vmlaq_f32(y8_11, wvec8_11, x);
        y12_15 = vmlaq_f32(y12_15, wvec12_15, x);

        vst1q_f32(&y[0], y0_3);
        vst1q_f32(&y[4], y4_7);
        vst1q_f32(&y[8], y8_11);
        vst1q_f32(&y[12], y12_15);
      }
    }
    free(initialized);
    return;
  } else if (cols % 8 == 0) {
    unsigned int n = cols / 8;
    bool *initialized = (bool *)malloc(sizeof(bool) * n);
    if (initialized == nullptr) {
      ml_loge("failed to malloc");
      return;
    }

    unsigned int step;
    for (unsigned int i = 0; i < cols / 8; ++i) {
      initialized[i] = false;
    }

    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 8) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        float32x4_t y4_7 = vld1q_f32(&y[4]);

        step = j / 8;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          y4_7 = vmulq_f32(y4_7, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3, wvec4_7;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);
        wvec4_7 = vld1q_f32(&w[4]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        y4_7 = vmlaq_f32(y4_7, wvec4_7, x);
        vst1q_f32(&y[0], y0_3);
        vst1q_f32(&y[4], y4_7);
      }
    }
    free(initialized);
    return;
  } else if (cols % 4 == 0) {
    unsigned int n = cols / 4;
    auto initialized = std::make_unique<bool[]>(n);
    if (initialized == nullptr) {
      ml_loge("Error : Memory allocation failed");
      return;
    }

    unsigned int step;
    for (unsigned int i = 0; i < cols / 4; ++i) {
      initialized[i] = false;
    }
    for (unsigned int i = 0; i < rows; ++i) {
      float32x4_t x = vld1q_dup_f32(&X[i]);
      x = vmulq_f32(x, v_alpha);

      for (unsigned int j = 0; j < cols; j += 4) {
        float *__restrict y = &Y[j];

        float32x4_t y0_3 = vld1q_f32(&y[0]);
        step = j / 4;
        if (!initialized[step]) {
          y0_3 = vmulq_f32(y0_3, v_beta);
          initialized[step] = true;
        }

        float32x4_t wvec0_3;
        const float *__restrict w;

        w = &A[i * cols + j];

        wvec0_3 = vld1q_f32(&w[0]);

        y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
        vst1q_f32(&y[0], y0_3);
      }
    }
  }

  return;
}

#ifdef ENABLE_FP16
void sgemv_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t rows,
                     uint32_t cols, float alpha, float beta) {
  const __fp16 *__restrict x;

  // std::cout << rows << " X " << cols << std::endl;

  float16x8_t v_beta = vmovq_n_f16(beta);

  for (unsigned int i = 0; i < rows; i += 8) {
    float16x8_t y = vld1q_f16(&Y[i]);
    y = vmulq_f16(v_beta, y);
    vst1q_f16(&Y[i], y);
  }

  float16x8_t v_alpha = vmovq_n_f16(alpha);

  float16x8_t x0_7 = vld1q_f16(&X[0]);
  for (int i = 0; i < 8; ++i) {
  }
  if (cols % 32 == 0) {
    for (unsigned i = 0; i < cols; i += 32) {
      float16x8_t x0_7 = vld1q_f16(&X[i]);
      float16x8_t x8_15 = vld1q_f16(&X[i + 8]);
      float16x8_t x16_23 = vld1q_f16(&X[i + 16]);
      float16x8_t x24_31 = vld1q_f16(&X[i + 24]);

      if (alpha != 1.0) {
        x0_7 = vmulq_f16(x0_7, v_alpha);
        x8_15 = vmulq_f16(x8_15, v_alpha);
        x16_23 = vmulq_f16(x16_23, v_alpha);
        x24_31 = vmulq_f16(x24_31, v_alpha);
      }

      float16x8_t wvec0_7, wvec8_15, wvec16_23, wvec24_31;

      const __fp16 *__restrict w;

      float16x8_t y0;
      __fp16 r[4];

      float16x4_t y0_high;
      float16x4_t y0_low;
      if (i == 0) {
        w = &A[0 * cols + i];
        wvec0_7 = vld1q_f16(&w[0]);
        for (int z = 0; z < 8; ++z) {
        }
      }

      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f16(0);

        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);
        wvec16_23 = vld1q_f16(&w[16]);
        wvec24_31 = vld1q_f16(&w[24]);

        y0 = vfmaq_f16(y0, wvec0_7, x0_7);
        if (i == 0 && j == 0) {
          for (int z = 0; z < 8; ++z) {
          }
        }
        y0 = vfmaq_f16(y0, wvec8_15, x8_15);
        y0 = vfmaq_f16(y0, wvec16_23, x16_23);
        y0 = vfmaq_f16(y0, wvec24_31, x24_31);

        y0_high = vget_high_f16(y0);
        y0_low = vget_low_f16(y0);

        y0_low = vadd_f16(y0_high, y0_low);
        vst1_f16(r, y0_low);

        Y[j] += r[0] + r[1] + r[2] + r[3];
      }
    }

  } else if (cols % 16 == 0) {

    for (unsigned i = 0; i < cols; i += 16) {
      float16x8_t x0_7 = vld1q_f16(&X[i]);
      float16x8_t x8_15 = vld1q_f16(&X[i + 8]);

      if (alpha != 1.0) {
        x0_7 = vmulq_f16(x0_7, v_alpha);
        x8_15 = vmulq_f16(x8_15, v_alpha);
      }

      float16x8_t wvec0_7, wvec8_15;

      const __fp16 *__restrict w;

      float16x8_t y0;
      __fp16 r[4];

      float16x4_t y0_high;
      float16x4_t y0_low;
      for (unsigned int j = 0; j < rows; ++j) {
        w = &A[j * cols + i];
        y0 = vmovq_n_f16(0);

        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);

        y0 = vfmaq_f16(y0, wvec0_7, x0_7);
        y0 = vfmaq_f16(y0, wvec8_15, x8_15);

        y0_high = vget_high_f16(y0);
        y0_low = vget_low_f16(y0);

        y0_low = vadd_f16(y0_high, y0_low);
        vst1_f16(r, y0_low);

        Y[j] += r[0] + r[1] + r[2] + r[3];
      }
    }
  } else if (cols % 8 == 0) {
    for (unsigned i = 0; i < cols; i += 8) {
      float16x8_t x0_7 = vld1q_f16(&X[i]);

      if (alpha != 1.0) {
        x0_7 = vmulq_f16(x0_7, v_alpha);
      }

      float16x8_t wvec0_7;

      float16x8_t y0;
      __fp16 r[4];

      float16x4_t y0_high;
      float16x4_t y0_low;
      for (unsigned int j = 0; j < rows; ++j) {

        wvec0_7 = vld1q_f16(&A[j * cols + i]);

        y0 = vmulq_f16(wvec0_7, x0_7);

        y0_high = vget_high_f16(y0);
        y0_low = vget_low_f16(y0);

        y0_low = vadd_f16(y0_high, y0_low);
        vst1_f16(r, y0_low);

        Y[j] += r[0] + r[1] + r[2] + r[3];
      }
    }
  }
}

void sgemv_transpose_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y,
                               uint32_t rows, uint32_t cols, float alpha,
                               float beta) {

  const float16x8_t v_beta = vmovq_n_f16(beta);
  const float16x8_t v_alpha = vmovq_n_f16(alpha);

  if (cols % 32 == 0) {
    for (unsigned int j = 0; j < cols; j += 8) {
      float16x8_t y0_7 = vld1q_f16(&Y[j]);
      y0_7 = vmulq_f16(y0_7, v_beta);
      vst1q_f16(&Y[j], y0_7);
    }
    float16x8_t wvec0_7 = vld1q_f16(&A[0]);
    for (int z = 0; z < 8; ++z) {
    }

    for (unsigned int i = 0; i < rows; ++i) {
      __fp16 x = alpha * X[i];

      for (unsigned int j = 0; j < cols; j += 32) {
        __fp16 *__restrict y = &Y[j];

        float16x8_t y0_7 = vld1q_f16(&y[0]);
        float16x8_t y8_15 = vld1q_f16(&y[8]);
        float16x8_t y16_23 = vld1q_f16(&y[16]);
        float16x8_t y24_31 = vld1q_f16(&y[24]);

        float16x8_t wvec0_7, wvec8_15, wvec16_23, wvec24_31;
        const __fp16 *__restrict w;

        w = &A[i * cols + j];

        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);
        wvec16_23 = vld1q_f16(&w[16]);
        wvec24_31 = vld1q_f16(&w[24]);

        y0_7 = vfmaq_n_f16(y0_7, wvec0_7, x);
        y8_15 = vfmaq_n_f16(y8_15, wvec8_15, x);
        y16_23 = vfmaq_n_f16(y16_23, wvec16_23, x);
        y24_31 = vfmaq_n_f16(y24_31, wvec24_31, x);

        if (i == 0 && j == 0) {
          for (int z = 0; z < 8; ++z) {
          }
        }

        vst1q_f16(&y[0], y0_7);
        vst1q_f16(&y[8], y8_15);
        vst1q_f16(&y[16], y16_23);
        vst1q_f16(&y[24], y24_31);
      }
    }
    return;
  } else if (cols % 16 == 0) {

    for (unsigned int j = 0; j < cols; j += 8) {
      float16x8_t y0_7 = vld1q_f16(&Y[j]);
      y0_7 = vmulq_f16(y0_7, v_beta);
      vst1q_f16(&Y[j], y0_7);
    }

    for (unsigned int i = 0; i < rows; ++i) {
      __fp16 x = alpha * X[i];

      for (unsigned int j = 0; j < cols; j += 16) {
        __fp16 *__restrict y = &Y[j];

        float16x8_t y0_7 = vld1q_f16(&y[0]);
        float16x8_t y8_15 = vld1q_f16(&y[8]);

        float16x8_t wvec0_7, wvec8_15;
        const __fp16 *__restrict w;

        w = &A[i * cols + j];

        wvec0_7 = vld1q_f16(&w[0]);
        wvec8_15 = vld1q_f16(&w[8]);

        y0_7 = vfmaq_n_f16(y0_7, wvec0_7, x);
        y8_15 = vfmaq_n_f16(y8_15, wvec8_15, x);

        vst1q_f16(&y[0], y0_7);
        vst1q_f16(&y[8], y8_15);
      }
    }
    return;
  } else if (cols % 8 == 0) {

    for (unsigned int j = 0; j < cols; j += 8) {
      float16x8_t y0_7 = vld1q_f16(&Y[j]);
      y0_7 = vmulq_f16(y0_7, v_beta);
      vst1q_f16(&Y[j], y0_7);
    }

    for (unsigned int i = 0; i < rows; ++i) {

      __fp16 x = alpha * X[i];

      for (unsigned int j = 0; j < cols; j += 8) {

        float16x8_t y0_7 = vld1q_f16(&Y[j]);
        float16x8_t wvec0_7 = vld1q_f16(&A[i * cols + j]);

        y0_7 = vfmaq_n_f16(y0_7, wvec0_7, x);

        vst1q_f16(&Y[j], y0_7);
      }
    }
    return;
  }
}

void saxpy_neon_fp16(const unsigned int N, const float alpha, const __fp16 *X,
                     __fp16 *Y) {

  const float16x8_t v_alphaX8 = vmovq_n_f16(alpha);
  const float16x4_t v_alphaX4 = vmov_n_f16(alpha);

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);
    float16x8_t y = vld1q_f16(&Y[idx]);

    // alpha*X + Y -> mulacc
    float16x8_t mulacc = vfmaq_f16(y, v_alphaX8, x);
    vst1q_f16(&Y[idx], mulacc);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);
    float16x4_t y = vld1_f16(&Y[idx]);

    // alpha*X + Y -> mulacc
    float16x4_t mulacc = vfma_f16(y, v_alphaX4, x);
    vst1_f16(&Y[idx], mulacc);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    Y[idx] = Y[idx] + alpha * X[idx];
}

__fp16 sdot_neon_fp16(const unsigned int N, const __fp16 *X, const __fp16 *Y) {

  float16x8_t accX8 = vmovq_n_f16(0);
  float16x4_t accX4 = vmov_n_f16(0);

  unsigned int idx = 0;
  __fp16 ret = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);
    float16x8_t y = vld1q_f16(&Y[idx]);

    // x*y + accX8 -> accX8
    accX8 = vfmaq_f16(accX8, x, y);
  }

  // check at least one batch of 8 is processed
  if (N - 8 >= 0) {
    __fp16 result[8];
    vst1q_f16(result, accX8);
    for (unsigned int i = 0; i < 8; i++)
      ret += result[i];
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);
    float16x4_t y = vld1_f16(&Y[idx]);

    // x*y + accX4 -> accX4
    accX4 = vfma_f16(accX4, x, y);
  }

  // check at least one batch of 4 is processed
  if (N % 8 >= 4) {
    __fp16 result[4];
    vst1_f16(result, accX4);
    ret += result[0] + result[1] + result[2] + result[3];
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    ret += X[idx] * Y[idx];

  return ret;
}

__fp16 snrm2_neon_fp16(const unsigned int N, const __fp16 *X) {

  float16x8_t accX8 = vmovq_n_f16(0);
  float16x4_t accX4 = vmov_n_f16(0);

  unsigned int idx = 0;
  __fp16 ret = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);

    // x*x + accX8 -> accX8
    accX8 = vfmaq_f16(accX8, x, x);
  }

  // check at least one batch of 8 is processed
  if (N - 8 >= 0) {
    __fp16 result[8];
    vst1q_f16(result, accX8);
    for (unsigned int i = 0; i < 8; i++)
      ret += result[i];
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);

    // x*x + accX4 -> accX4
    accX4 = vfma_f16(accX4, x, x);
  }

  // check at least one batch of 4 is processed
  if (N % 8 >= 4) {
    __fp16 result[4];
    vst1_f16(result, accX4);
    ret += result[0] + result[1] + result[2] + result[3];
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    ret += X[idx] * X[idx];

  return ret;
}

void sscal_neon_fp16(const unsigned int N, __fp16 *X, const float alpha) {
  const float16x8_t v_alphaX8 = vmovq_n_f16(alpha);
  const float16x4_t v_alphaX4 = vmov_n_f16(alpha);

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t x = vld1q_f16(&X[idx]);

    // alpha*X -> X
    float16x8_t mulacc = vmulq_f16(v_alphaX8, x);
    vst1q_f16(&X[idx], mulacc);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t x = vld1_f16(&X[idx]);

    // alpha*X -> X
    float16x4_t mulacc = vmul_f16(v_alphaX4, x);
    vst1_f16(&X[idx], mulacc);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    X[idx] = alpha * X[idx];
}

void scopy_neon_fp16(const unsigned int N, const __fp16 *X, __fp16 *Y) {

  unsigned int idx = 0;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t batch = vld1q_f16(&X[idx]);
    vst1q_f16(&Y[idx], batch);
  }

  // processing remaining batch of 4
  for (; (N - idx) >= 4; idx += 4) {
    float16x4_t batch = vld1_f16(&X[idx]);
    vst1_f16(&Y[idx], batch);
  }

  // pocessing remaining values
  for (; idx < N; idx++)
    Y[idx] = X[idx];
}

void scopy_neon_int4(const unsigned int N, const uint8_t *X, __fp16 *Y) {

  unsigned int idx = 0;

  // keep in mind that : len(X) = N, and len(Y) = 2*N

  // processing batch of 16

  float16x8_t y0, y1, y2, y3;
  float16x4_t yh0, yh1;

  uint8_t low0, low1, high0, high1;

  for (; (N - idx) >= 16; idx += 16) {
    uint8x16_t batch = vld1q_u8(&X[idx]);

    uint8x8_t low = vget_low_u8(batch);
    uint8x8_t high = vget_high_u8(batch);

    for (int i = 0; i < 8; ++i) {
      low0 = low[i] >> 4;
      low1 = low[i] & 0x0f;

      high0 = high[i] >> 4;
      high1 = high[i] & 0x0f;

      if (i < 4) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else {
        y1[2 * (i - 4)] = low0;
        y1[2 * (i - 4) + 1] = low1;
      }

      if (i < 4) {
        y2[2 * i] = high0;
        y2[2 * i + 1] = high1;
      } else {
        y3[2 * (i - 4)] = high0;
        y3[2 * (i - 4) + 1] = high1;
      }
    }

    vst1q_f16(&Y[2 * idx], y0);
    vst1q_f16(&Y[2 * idx + 8], y1);
    vst1q_f16(&Y[2 * idx + 16], y2);
    vst1q_f16(&Y[2 * idx + 24], y3);
  }

  // processing remaining batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    int8x8_t batch = vld1_u8(&X[idx]);

    for (int i = 0; i < 8; ++i) {
      low0 = batch[i] >> 4;
      low1 = batch[i] & 0x0f;

      if (i < 4) {
        y0[2 * i] = low0;
        y0[2 * i + 1] = low1;
      } else {
        y1[2 * (i - 4)] = low0;
        y1[2 * (i - 4) + 1] = low1;
      }
    }

    vst1q_f16(&Y[2 * idx], y0);
    vst1q_f16(&Y[2 * idx + 8], y1);
  }

  // pocessing remaining values
  for (; idx < N; idx++) {
    Y[2 * idx] = X[idx] >> 4;
    Y[2 * idx + 1] = X[idx] & 0x0f;
  }
}

unsigned int isamax_neon_fp16(const unsigned int N, const __fp16 *X) {

  unsigned int retIdx;
  __fp16 maxNum;

  uint16_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint16x8_t stride = vmovq_n_u16(8);
  float16x8_t batch = vld1q_f16(&X[0]);
  uint16x8_t curr_index = vld1q_u16(indices);
  uint16x8_t max_index = curr_index;

  unsigned int idx = 8;

  // processing batch of 8
  for (; (N - idx) >= 8; idx += 8) {
    float16x8_t values = vld1q_f16(&X[idx]);
    curr_index = vaddq_u16(curr_index, stride);

    // comparison
    uint16x8_t mask = vcgtq_f16(batch, values);

    // blend values and indices based on the mask
    batch = vbslq_f16(mask, batch, values);
    max_index = vbslq_u16(mask, max_index, curr_index);
  }

  // storing indices and max values
  __fp16 maxVal[8];
  vst1q_f16(maxVal, batch);
  vst1q_u16(indices, max_index);

  // getting the index of the maxima
  maxNum = maxVal[0];
  retIdx = max_index[0];
  for (int i = 1; i < 8; i++) {
    if (maxVal[i] > maxNum) {
      maxNum = maxVal[i];
      retIdx = max_index[i];
    }
  }

  // processing remaining values
  for (; idx < N; idx++) {
    if (X[idx] > maxNum) {
      maxNum = X[idx];
      retIdx = idx;
    }
  }

  return retIdx;
}

void sgemm_neon_fp16(const __fp16 *A, const __fp16 *B, __fp16 *C, uint32_t M,
                     uint32_t N, uint32_t K, float alpha, float beta,
                     bool TransA, bool TransB) {

  float16x8_t v_beta = vmovq_n_f16(beta);

  // performing beta*C
  unsigned int idx = 0;
  unsigned int size = M * N;
  for (; idx < (size - idx) >= 8; idx += 8) {
    float16x8_t c = vld1q_f16(&C[idx]);
    c = vmulq_f16(v_beta, c);
    vst1q_f16(&C[idx], c);
  }

  // remaining values if dimensions not a multiple of 8
  for (; idx < size; idx++) {
    C[idx] *= beta;
  }

  if (!TransA && TransB) {
    sgemm_neon_fp16_transB(A, B, C, M, N, K, alpha, beta);
  } else if (TransA && !TransB) {
    sgemm_neon_fp16_transA(A, B, C, M, N, K, alpha, beta);
  } else if (!TransA && !TransB) {
    sgemm_neon_fp16_noTrans(A, B, C, M, N, K, alpha, beta);
  } else { // TransA && TransB
    sgemm_neon_fp16_transAB(A, B, C, M, N, K, alpha, beta, idx);
  }
}


void sgemm_neon_fp16_noTrans(const __fp16 *A, const __fp16 *B, __fp16 *C,
                             uint32_t M, uint32_t N, uint32_t K, float alpha,
                             float beta) {

  unsigned int k = 0, n = 0;

  for (; (K - k) >= 8; k += 8) {
    for (unsigned int m = 0; m < M; m++) {
      float a0 = alpha * A[m * K + k];
      float a1 = alpha * A[m * K + k + 1];
      float a2 = alpha * A[m * K + k + 2];
      float a3 = alpha * A[m * K + k + 3];
      float a4 = alpha * A[m * K + k + 4];
      float a5 = alpha * A[m * K + k + 5];
      float a6 = alpha * A[m * K + k + 6];
      float a7 = alpha * A[m * K + k + 7];

      for (n = 0; (N - n) >= 8; n += 8) {
        float16x8_t b0_7_0 = vld1q_f16(&B[k * N + n]);
        float16x8_t b0_7_1 = vld1q_f16(&B[(k + 1) * N + n]);
        float16x8_t b0_7_2 = vld1q_f16(&B[(k + 2) * N + n]);
        float16x8_t b0_7_3 = vld1q_f16(&B[(k + 3) * N + n]);
        float16x8_t b0_7_4 = vld1q_f16(&B[(k + 4) * N + n]);
        float16x8_t b0_7_5 = vld1q_f16(&B[(k + 5) * N + n]);
        float16x8_t b0_7_6 = vld1q_f16(&B[(k + 6) * N + n]);
        float16x8_t b0_7_7 = vld1q_f16(&B[(k + 7) * N + n]);

        float32x4_t b0_7_0_low = vcvt_f32_f16(vget_low_f16(b0_7_0));
        float32x4_t b0_7_0_high = vcvt_f32_f16(vget_high_f16(b0_7_0));

        float32x4_t b0_7_1_low = vcvt_f32_f16(vget_low_f16(b0_7_1));
        float32x4_t b0_7_1_high = vcvt_f32_f16(vget_high_f16(b0_7_1));

        float32x4_t b0_7_2_low = vcvt_f32_f16(vget_low_f16(b0_7_2));
        float32x4_t b0_7_2_high = vcvt_f32_f16(vget_high_f16(b0_7_2));

        float32x4_t b0_7_3_low = vcvt_f32_f16(vget_low_f16(b0_7_3));
        float32x4_t b0_7_3_high = vcvt_f32_f16(vget_high_f16(b0_7_3));

        float32x4_t b0_7_4_low = vcvt_f32_f16(vget_low_f16(b0_7_4));
        float32x4_t b0_7_4_high = vcvt_f32_f16(vget_high_f16(b0_7_4));

        float32x4_t b0_7_5_low = vcvt_f32_f16(vget_low_f16(b0_7_5));
        float32x4_t b0_7_5_high = vcvt_f32_f16(vget_high_f16(b0_7_5));

        float32x4_t b0_7_6_low = vcvt_f32_f16(vget_low_f16(b0_7_6));
        float32x4_t b0_7_6_high = vcvt_f32_f16(vget_high_f16(b0_7_6));

        float32x4_t b0_7_7_low = vcvt_f32_f16(vget_low_f16(b0_7_7));
        float32x4_t b0_7_7_high = vcvt_f32_f16(vget_high_f16(b0_7_7));

        float16x8_t c0_7 = vld1q_f16(&C[m * N + n]);
        float32x4_t c0_7_low_32 = vcvt_f32_f16(vget_low_f16(c0_7));
        float32x4_t c0_7_high_32 = vcvt_f32_f16(vget_high_f16(c0_7));

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_0_low, a0);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_0_high, a0);

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_1_low, a1);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_1_high, a1);

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_2_low, a2);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_2_high, a2);

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_3_low, a3);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_3_high, a3);

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_4_low, a4);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_4_high, a4);

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_5_low, a5);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_5_high, a5);

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_6_low, a6);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_6_high, a6);

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_7_low, a7);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_7_high, a7);

        float16x4_t c0_7_low_16 = vcvt_f16_f32(c0_7_low_32);
        float16x4_t c0_7_high_16 = vcvt_f16_f32(c0_7_high_32);

        c0_7 = vcombine_f16(c0_7_low_16, c0_7_high_16);

        vst1q_f16(&C[m * N + n], c0_7);
      }
    }
  }

  for (; (K - k) >= 4; k += 4) {
    for (unsigned int m = 0; m < M; m++) {
      float a0 = alpha * A[m * K + k];
      float a1 = alpha * A[m * K + k + 1];
      float a2 = alpha * A[m * K + k + 2];
      float a3 = alpha * A[m * K + k + 3];

      for (n = 0; (N - n) >= 8; n += 8) {
        float16x8_t b0_7_0 = vld1q_f16(&B[k * N + n]);
        float16x8_t b0_7_1 = vld1q_f16(&B[(k + 1) * N + n]);
        float16x8_t b0_7_2 = vld1q_f16(&B[(k + 2) * N + n]);
        float16x8_t b0_7_3 = vld1q_f16(&B[(k + 3) * N + n]);

        float32x4_t b0_7_0_low = vcvt_f32_f16(vget_low_f16(b0_7_0));
        float32x4_t b0_7_0_high = vcvt_f32_f16(vget_high_f16(b0_7_0));

        float32x4_t b0_7_1_low = vcvt_f32_f16(vget_low_f16(b0_7_1));
        float32x4_t b0_7_1_high = vcvt_f32_f16(vget_high_f16(b0_7_1));

        float32x4_t b0_7_2_low = vcvt_f32_f16(vget_low_f16(b0_7_2));
        float32x4_t b0_7_2_high = vcvt_f32_f16(vget_high_f16(b0_7_2));

        float32x4_t b0_7_3_low = vcvt_f32_f16(vget_low_f16(b0_7_3));
        float32x4_t b0_7_3_high = vcvt_f32_f16(vget_high_f16(b0_7_3));

        float16x8_t c0_7 = vld1q_f16(&C[m * N + n]);
        float32x4_t c0_7_low_32 = vcvt_f32_f16(vget_low_f16(c0_7));
        float32x4_t c0_7_high_32 = vcvt_f32_f16(vget_high_f16(c0_7));

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_0_low, a0);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_0_high, a0);

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_1_low, a1);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_1_high, a1);

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_2_low, a2);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_2_high, a2);

        c0_7_low_32 = vfmaq_n_f32(c0_7_low_32, b0_7_3_low, a3);
        c0_7_high_32 = vfmaq_n_f32(c0_7_high_32, b0_7_3_high, a3);

        float16x4_t c0_7_low_16 = vcvt_f16_f32(c0_7_low_32);
        float16x4_t c0_7_high_16 = vcvt_f16_f32(c0_7_high_32);

        c0_7 = vcombine_f16(c0_7_low_16, c0_7_high_16);

        vst1q_f16(&C[m * N + n], c0_7);
      }
    }
  }

  // remaining K values
  for (; k < K; k++) {
    for (unsigned int m = 0; m < M; m++) {
      __fp16 a0 = alpha * A[m * K + k];

      for (n = 0; (N - n) >= 8; n += 8) {
        float16x8_t b0_7 = vld1q_f16(&B[k * N + n]);

        float16x8_t c0_7 = vld1q_f16(&C[m * N + n]);

        c0_7 = vfmaq_n_f16(c0_7, b0_7, a0);

        vst1q_f16(&C[m * N + n], c0_7);
      }
    }
  }

  // remaining N values (can be optimized by putting inside previous loops)
  if (n < N) {
    __fp16 valsB[8];
    __fp16 valsC[8];
    for (k = 0; k < K; k++) {
      for (unsigned int m = 0; m < M; m++) {
        __fp16 a = alpha * A[m * K + k];
        for (unsigned int idx = n; idx < N; idx++) {
          valsB[idx - n] = B[k * N + idx];

          // load previously calculated C
          valsC[idx - n] = C[m * N + idx];
        }
        float16x8_t b = vld1q_f16(valsB);
        float16x8_t c = vld1q_f16(valsC);
        c = vfmaq_n_f16(c, b, a);
        vst1q_f16(valsC, c);

        for (unsigned int idx = n; idx < N; idx++) {
          C[m * N + idx] = valsC[idx - n];
        }
      }
    }
  }
}

void sgemm_neon_fp16_transA(const __fp16 *A, const __fp16 *B, __fp16 *C,
                            uint32_t M, uint32_t N, uint32_t K, float alpha,
                            float beta) {
  __fp16 valsB[8];
  __fp16 valsC[8];
  for (unsigned int k = 0; k < K; k++) {
    for (unsigned int m = 0; m < M; m++) {
      __fp16 a = alpha * A[k * M + m];
      unsigned int n = 0;
      for (; (N - n) >= 8; n += 8) {
        float16x8_t b = vld1q_f16(&B[k * N + n]);

        // load previously calculated C
        float16x8_t c = vld1q_f16(&C[m * N + n]);
        c = vfmaq_n_f16(c, b, a);
        vst1q_f16(&C[m * N + n], c);
      }

      // remaining N values
      if (n < N) {
        for (unsigned int idx = n; idx < N; idx++) {
          valsB[idx - n] = B[k * N + idx];

          // load previously calculated C
          valsC[idx - n] = C[m * N + idx];
        }
        float16x8_t b = vld1q_f16(valsB);
        float16x8_t c = vld1q_f16(valsC);
        c = vfmaq_n_f16(c, b, a);
        vst1q_f16(valsC, c);

        for (unsigned int idx = n; idx < N; idx++) {
          C[m * N + idx] = valsC[idx - n];
        }
      }
    }
  }
}


void sgemm_neon_fp16_transB(const __fp16 *A, const __fp16 *B, __fp16 *C,
                            uint32_t M, uint32_t N, uint32_t K, float alpha,
                            float beta) {
  __fp16 r[4];
  float16x8_t v_alpha = vmovq_n_f16(alpha);
  if (K % 16 == 0) {
    for (unsigned int m = 0; m < M; m++) {
      for (unsigned int n = 0; n < N; n++) {
        float16x8_t sum = vmovq_n_f16(0);
        unsigned int k = 0;
        for (; (K - k) >= 16; k += 16) {
          float16x8_t a = vld1q_f16(&A[m * K + k]);
          float16x8_t a8_15 = vld1q_f16(&A[m * K + k + 8]);
          float16x8_t b = vld1q_f16(&B[n * K + k]);
          float16x8_t b8_15 = vld1q_f16(&B[n * K + k + 8]);

          sum = vfmaq_f16(sum, a, b);
          sum = vfmaq_f16(sum, a8_15, b8_15);
        }

        sum = vmulq_f16(v_alpha, sum);

        float16x4_t sum_high = vget_high_f16(sum);
        float16x4_t sum_low = vget_low_f16(sum);

        float32x4_t sum_high_32 = vcvt_f32_f16(sum_high);
        float32x4_t sum_low_32 = vcvt_f32_f16(sum_low);

        float32x2_t sum_high_32_two = vpadd_f32(vget_low_f32(sum_high_32), vget_high_f32(sum_high_32));
        float32x2_t sum_low_32_two = vpadd_f32(vget_low_f32(sum_low_32), vget_low_f32(sum_low_32));

        float32x2_t sum_high_32_one = vpadd_f32(sum_high_32_two, sum_high_32_two);
        float32x2_t sum_low_32_one = vpadd_f32(sum_low_32_two, sum_low_32_two);

        float result_high, result_low;
        vst1_lane_f32(&result_high, sum_high_32_one, 0);
        vst1_lane_f32(&result_low, sum_low_32_one, 0);

        C[m * N + n] += result_high + result_low;
      }
    }
  } else {
    __fp16 valsB[8];
    __fp16 valsA[8];
    for (unsigned int m = 0; m < M; m++) {
      for (unsigned int n = 0; n < N; n++) {
        float16x8_t sum = vmovq_n_f16(0);
        unsigned int k = 0;
        for (; (K - k) >= 8; k += 8) {
          float16x8_t a = vld1q_f16(&A[m * K + k]);
          float16x8_t b = vld1q_f16(&B[n * K + k]);
          sum = vfmaq_f16(sum, a, b);
        }

        // remaining K values
        if (k < K) {
          unsigned int idx;
          for (idx = k; idx < K; idx++) {
            valsA[idx - k] = A[m * K + idx];
            valsB[idx - k] = B[n * K + idx];
          }
          // to cover entire 128 bits (reset unused bits)
          while (idx < (k + 8)) {
            valsA[idx - k] = 0;
            valsB[idx - k] = 0;
            idx++;
          }
          // updating sum
          float16x8_t a = vld1q_f16(valsA);
          float16x8_t b = vld1q_f16(valsB);
          sum = vfmaq_f16(sum, a, b);
        }

        sum = vmulq_f16(v_alpha, sum);

        float16x4_t sum_high = vget_high_f16(sum);
        float16x4_t sum_low = vget_low_f16(sum);

        float32x4_t sum_high_32 = vcvt_f32_f16(sum_high);
        float32x4_t sum_low_32 = vcvt_f32_f16(sum_low);

        float32x2_t sum_high_32_two = vpadd_f32(vget_low_f32(sum_high_32), vget_high_f32(sum_high_32));
        float32x2_t sum_low_32_two = vpadd_f32(vget_low_f32(sum_low_32), vget_low_f32(sum_low_32));

        float32x2_t sum_high_32_one = vpadd_f32(sum_high_32_two, sum_high_32_two);
        float32x2_t sum_low_32_one = vpadd_f32(sum_low_32_two, sum_low_32_two);

        float result_high, result_low;
        vst1_lane_f32(&result_high, sum_high_32_one, 0);
        vst1_lane_f32(&result_low, sum_low_32_one, 0);

        C[m * N + n] += result_high + result_low;
      }
    }
  }
}

void sgemm_neon_fp16_transAB(const __fp16 *A, const __fp16 *B, __fp16 *C,
                             uint32_t M, uint32_t N, uint32_t K, float alpha,
                             float beta, uint32_t idx) {
  __fp16 vals[8];
  for (unsigned int n = 0; n < N; n++) {
    for (unsigned int k = 0; k < K; k++) {

      __fp16 b = alpha * B[n * K + k];
      unsigned int m = 0;
      for (; (M - m) >= 8; m += 8) {
        float16x8_t a = vld1q_f16(&A[k * M + m]);
        a = vmulq_n_f16(a, b);
        vst1q_f16(vals, a);

        // calculations for all M values
        for (unsigned int idx = m; idx < m + 8; idx++)
          C[idx * N + n] += vals[idx - m];
      }

      // remaining when M is not a multiple of 8
      if (m < M) {
        for (idx = m; idx < M; idx++) {
          vals[idx - m] = A[k * M + idx];
        }

        float16x8_t a = vld1q_f16(vals);
        a = vmulq_n_f16(a, b);
        vst1q_f16(vals, a);

        // calculations for all remaining M values
        for (idx = m; idx < M; idx++)
          C[idx * N + n] += vals[idx - m];
      }
    }
  }
}

void elementwise_vector_multiplication_neon_fp16(const unsigned int N, const __fp16 *X,
                                                 const __fp16 *Y, __fp16 *Z) {
  int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    float16x8_t z0_7 = vmulq_f16(x0_7, y0_7);

    vst1q_f16(&Z[i], z0_7);
  }
  while (i < N){
    Z[i] = X[i] * Y[i];
    ++i;
  }
}

void elementwise_vector_addition_neon_fp16(const unsigned int N, const __fp16 *X,
                                                 const __fp16 *Y, __fp16 *Z) {
  int i = 0;
  for (; N - i >= 8; i += 8) {
    float16x8_t x0_7 = vld1q_f16(&X[i]);
    float16x8_t y0_7 = vld1q_f16(&Y[i]);
    float16x8_t z0_7 = vaddq_f16(x0_7, y0_7);

    vst1q_f16(&Z[i], z0_7);
  }
  while (i < N){
    Z[i] = X[i] * Y[i];
    ++i;
  }
}
  

#endif
} // namespace nntrainer::neon

// // SPDX-License-Identifier: Apache-2.0
// /**
//  * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
//  *
//  * @file   blas_neon.cpp
//  * @date   4 Aug 2022
//  * @see    https://github.com/nnstreamer/nntrainer
//  * @author Jijoong Moon <jijoong.moon@samsung.com>
//  * @bug    No known bugs except for NYI items
//  * @brief  This is Source for blas neon implementation
//  *
//  */

// #include <blas_neon.h>
// #include <nntrainer_error.h>

// namespace nntrainer::neon {

// void sgemv_neon(const float *A, const float *X, float *Y, uint32_t rows,
//                 uint32_t cols, float alpha, float beta) {
//   const float *__restrict x;

//   for (unsigned int i = 0; i < rows; ++i) {
//     Y[i] = Y[i] * beta;
//   }

//   float32x4_t v_alpha = vmovq_n_f32(alpha);

//   if (cols % 16 == 0) {
//     for (unsigned i = 0; i < cols; i += 16) {
//       float32x4_t x0_3 = vld1q_f32(&X[i]);
//       float32x4_t x4_7 = vld1q_f32(&X[i + 4]);
//       float32x4_t x8_11 = vld1q_f32(&X[i + 8]);
//       float32x4_t x12_15 = vld1q_f32(&X[i + 12]);

//       if (alpha != 1.0) {
//         x0_3 = vmulq_f32(x0_3, v_alpha);
//         x4_7 = vmulq_f32(x4_7, v_alpha);
//         x8_11 = vmulq_f32(x8_11, v_alpha);
//         x12_15 = vmulq_f32(x12_15, v_alpha);
//       }

//       float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;

//       const float *__restrict w;

//       float32x4_t y0;

//       for (unsigned int j = 0; j < rows; ++j) {
//         w = &A[j * cols + i];
//         y0 = vmovq_n_f32(0);

//         float r[4];
//         wvec0_3 = vld1q_f32(&w[0]);
//         wvec4_7 = vld1q_f32(&w[4]);
//         wvec8_11 = vld1q_f32(&w[8]);
//         wvec12_15 = vld1q_f32(&w[12]);

//         y0 = vmlaq_f32(y0, wvec0_3, x0_3);
//         y0 = vmlaq_f32(y0, wvec4_7, x4_7);
//         y0 = vmlaq_f32(y0, wvec8_11, x8_11);
//         y0 = vmlaq_f32(y0, wvec12_15, x12_15);

//         vst1q_f32(r, y0);
//         for (unsigned int k = 0; k < 4; ++k) {
//           Y[j] = Y[j] + r[k];
//         }
//       }
//     }

//   } else if (cols % 8 == 0) {
//     for (unsigned i = 0; i < cols; i += 8) {
//       float32x4_t x0_3 = vld1q_f32(&X[i]);
//       float32x4_t x4_7 = vld1q_f32(&X[i + 4]);

//       if (alpha != 1.0) {
//         x0_3 = vmulq_f32(x0_3, v_alpha);
//         x4_7 = vmulq_f32(x4_7, v_alpha);
//       }

//       float32x4_t wvec0_3, wvec4_7;

//       const float *__restrict w;

//       float32x4_t y0;

//       for (unsigned int j = 0; j < rows; ++j) {
//         w = &A[j * cols + i];
//         y0 = vmovq_n_f32(0);

//         float r[4];
//         wvec0_3 = vld1q_f32(&w[0]);
//         wvec4_7 = vld1q_f32(&w[4]);

//         y0 = vmlaq_f32(y0, wvec0_3, x0_3);
//         y0 = vmlaq_f32(y0, wvec4_7, x4_7);

//         vst1q_f32(r, y0);
//         for (unsigned int k = 0; k < 4; ++k) {
//           Y[j] = Y[j] + r[k];
//         }
//       }
//     }
//   } else if (cols % 4 == 0) {
//     for (unsigned i = 0; i < cols; i += 4) {
//       float32x4_t x0_3 = vld1q_f32(&X[i]);

//       if (alpha != 1.0) {
//         x0_3 = vmulq_f32(x0_3, v_alpha);
//       }

//       float32x4_t wvec0_3, wvec4_7;

//       const float *__restrict w;

//       float32x4_t y0;

//       for (unsigned int j = 0; j < rows; ++j) {
//         w = &A[j * cols + i];
//         y0 = vmovq_n_f32(0);

//         float r[4];
//         wvec0_3 = vld1q_f32(&w[0]);

//         y0 = vmlaq_f32(y0, wvec0_3, x0_3);

//         vst1q_f32(r, y0);
//         for (unsigned int k = 0; k < 4; ++k) {
//           Y[j] = Y[j] + r[k];
//         }
//       }
//     }
//   }
// }

// void sgemv_transpose_neon(const float *A, const float *X, float *Y,
//                           uint32_t rows, uint32_t cols, float alpha,
//                           float beta) {
//   const float *__restrict x;

//   const float32x4_t v_beta = vdupq_n_f32(beta);
//   const float32x4_t v_alpha = vdupq_n_f32(alpha);

//   if (cols % 16 == 0) {
//     unsigned int n = cols / 16;
//     bool *initialized = (bool *)malloc(sizeof(bool) * n);
//     unsigned int step;
//     for (unsigned int i = 0; i < cols / 16; ++i) {
//       initialized[i] = false;
//     }

//     for (unsigned int i = 0; i < rows; ++i) {
//       float32x4_t x = vld1q_dup_f32(&X[i]);
//       x = vmulq_f32(x, v_alpha);

//       for (unsigned int j = 0; j < cols; j += 16) {
//         float *__restrict y = &Y[j];

//         float32x4_t y0_3 = vld1q_f32(&y[0]);
//         float32x4_t y4_7 = vld1q_f32(&y[4]);
//         float32x4_t y8_11 = vld1q_f32(&y[8]);
//         float32x4_t y12_15 = vld1q_f32(&y[12]);
//         step = j / 16;
//         if (!initialized[step]) {
//           y0_3 = vmulq_f32(y0_3, v_beta);
//           y4_7 = vmulq_f32(y4_7, v_beta);
//           y8_11 = vmulq_f32(y8_11, v_beta);
//           y12_15 = vmulq_f32(y12_15, v_beta);
//           initialized[step] = true;
//         }

//         float32x4_t wvec0_3, wvec4_7, wvec8_11, wvec12_15;
//         const float *__restrict w;

//         w = &A[i * cols + j];

//         wvec0_3 = vld1q_f32(&w[0]);
//         wvec4_7 = vld1q_f32(&w[4]);
//         wvec8_11 = vld1q_f32(&w[8]);
//         wvec12_15 = vld1q_f32(&w[12]);

//         y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
//         y4_7 = vmlaq_f32(y4_7, wvec4_7, x);
//         y8_11 = vmlaq_f32(y8_11, wvec8_11, x);
//         y12_15 = vmlaq_f32(y12_15, wvec12_15, x);

//         vst1q_f32(&y[0], y0_3);
//         vst1q_f32(&y[4], y4_7);
//         vst1q_f32(&y[8], y8_11);
//         vst1q_f32(&y[12], y12_15);
//       }
//     }
//     free(initialized);
//     return;
//   } else if (cols % 8 == 0) {
//     unsigned int n = cols / 8;
//     bool *initialized = (bool *)malloc(sizeof(bool) * n);
//     unsigned int step;
//     for (unsigned int i = 0; i < cols / 8; ++i) {
//       initialized[i] = false;
//     }

//     for (unsigned int i = 0; i < rows; ++i) {
//       float32x4_t x = vld1q_dup_f32(&X[i]);
//       x = vmulq_f32(x, v_alpha);

//       for (unsigned int j = 0; j < cols; j += 8) {
//         float *__restrict y = &Y[j];

//         float32x4_t y0_3 = vld1q_f32(&y[0]);
//         float32x4_t y4_7 = vld1q_f32(&y[4]);

//         step = j / 8;
//         if (!initialized[step]) {
//           y0_3 = vmulq_f32(y0_3, v_beta);
//           y4_7 = vmulq_f32(y4_7, v_beta);
//           initialized[step] = true;
//         }

//         float32x4_t wvec0_3, wvec4_7;
//         const float *__restrict w;

//         w = &A[i * cols + j];

//         wvec0_3 = vld1q_f32(&w[0]);
//         wvec4_7 = vld1q_f32(&w[4]);

//         y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
//         y4_7 = vmlaq_f32(y4_7, wvec4_7, x);
//         vst1q_f32(&y[0], y0_3);
//         vst1q_f32(&y[4], y4_7);
//       }
//     }
//     free(initialized);
//     return;
//   } else if (cols % 4 == 0) {
//     unsigned int n = cols / 4;
//     bool *initialized = (bool *)malloc(sizeof(bool) * n);

//     unsigned int step;
//     for (unsigned int i = 0; i < cols / 4; ++i) {
//       initialized[i] = false;
//     }
//     for (unsigned int i = 0; i < rows; ++i) {
//       float32x4_t x = vld1q_dup_f32(&X[i]);
//       x = vmulq_f32(x, v_alpha);

//       for (unsigned int j = 0; j < cols; j += 4) {
//         float *__restrict y = &Y[j];

//         float32x4_t y0_3 = vld1q_f32(&y[0]);
//         step = j / 4;
//         if (!initialized[step]) {
//           y0_3 = vmulq_f32(y0_3, v_beta);
//           initialized[step] = true;
//         }

//         float32x4_t wvec0_3;
//         const float *__restrict w;

//         w = &A[i * cols + j];

//         wvec0_3 = vld1q_f32(&w[0]);

//         y0_3 = vmlaq_f32(y0_3, wvec0_3, x);
//         vst1q_f32(&y[0], y0_3);
//       }
//     }
//     free(initialized);
//   }

//   return;
// }

// #ifdef ENABLE_FP16
// void sgemv_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y, uint32_t
// rows,
//                      uint32_t cols, float alpha, float beta) {
//   const __fp16 *__restrict x;

//   float16x8_t v_beta = vmovq_n_f16(beta);

//   for (unsigned int i = 0; i < rows; i += 8) {
//     float16x8_t y = vld1q_f16(&Y[i]);
//     y = vmulq_f16(v_beta, y);
//     vst1q_f16(&Y[i], y);
//   }

//   float16x8_t v_alpha = vmovq_n_f16(alpha);

//   if (cols % 32 == 0) {
//     for (unsigned i = 0; i < cols; i += 32) {
//       float16x8_t x0_7 = vld1q_f16(&X[i]);
//       float16x8_t x8_15 = vld1q_f16(&X[i + 8]);
//       float16x8_t x16_23 = vld1q_f16(&X[i + 16]);
//       float16x8_t x24_31 = vld1q_f16(&X[i + 24]);

//       if (alpha != 1.0) {
//         x0_7 = vmulq_f16(x0_7, v_alpha);
//         x8_15 = vmulq_f16(x8_15, v_alpha);
//         x16_23 = vmulq_f16(x16_23, v_alpha);
//         x24_31 = vmulq_f16(x24_31, v_alpha);
//       }

//       float16x8_t wvec0_7, wvec8_15, wvec16_23, wvec24_31;

//       const __fp16 *__restrict w;

//       float16x8_t y0;
//       __fp16 r[4];

//       float16x4_t y0_high;
//       float16x4_t y0_low;
//       for (unsigned int j = 0; j < rows; ++j) {
//         w = &A[j * cols + i];
//         y0 = vmovq_n_f16(0);

//         wvec0_7 = vld1q_f16(&w[0]);
//         wvec8_15 = vld1q_f16(&w[8]);
//         wvec16_23 = vld1q_f16(&w[16]);
//         wvec24_31 = vld1q_f16(&w[24]);

//         y0 = vfmaq_f16(y0, wvec0_7, x0_7);
//         y0 = vfmaq_f16(y0, wvec8_15, x8_15);
//         y0 = vfmaq_f16(y0, wvec16_23, x16_23);
//         y0 = vfmaq_f16(y0, wvec24_31, x24_31);

//         y0_high = vget_high_f16(y0);
//         y0_low = vget_low_f16(y0);

//         y0_low = vadd_f16(y0_high, y0_low);
//         vst1_f16(r, y0_low);

//         Y[j] += r[0] + r[1] + r[2] + r[3];
//       }
//     }

//   } else if (cols % 16 == 0) {

//     for (unsigned i = 0; i < cols; i += 16) {
//       float16x8_t x0_7 = vld1q_f16(&X[i]);
//       float16x8_t x8_15 = vld1q_f16(&X[i + 8]);

//       if (alpha != 1.0) {
//         x0_7 = vmulq_f16(x0_7, v_alpha);
//         x8_15 = vmulq_f16(x8_15, v_alpha);
//       }

//       float16x8_t wvec0_7, wvec8_15;

//       const __fp16 *__restrict w;

//       float16x8_t y0;
//       __fp16 r[4];

//       float16x4_t y0_high;
//       float16x4_t y0_low;
//       for (unsigned int j = 0; j < rows; ++j) {
//         w = &A[j * cols + i];
//         y0 = vmovq_n_f16(0);

//         wvec0_7 = vld1q_f16(&w[0]);
//         wvec8_15 = vld1q_f16(&w[8]);

//         y0 = vfmaq_f16(y0, wvec0_7, x0_7);
//         y0 = vfmaq_f16(y0, wvec8_15, x8_15);

//         y0_high = vget_high_f16(y0);
//         y0_low = vget_low_f16(y0);

//         y0_low = vadd_f16(y0_high, y0_low);
//         vst1_f16(r, y0_low);

//         Y[j] += r[0] + r[1] + r[2] + r[3];
//       }
//     }
//   } else if (cols % 8 == 0) {
//     for (unsigned i = 0; i < cols; i += 8) {
//       float16x8_t x0_7 = vld1q_f16(&X[i]);

//       if (alpha != 1.0) {
//         x0_7 = vmulq_f16(x0_7, v_alpha);
//       }

//       float16x8_t wvec0_7;

//       float16x8_t y0;
//       __fp16 r[4];

//       float16x4_t y0_high;
//       float16x4_t y0_low;
//       for (unsigned int j = 0; j < rows; ++j) {

//         wvec0_7 = vld1q_f16(&A[j * cols + i]);

//         y0 = vmulq_f16(wvec0_7, x0_7);

//         y0_high = vget_high_f16(y0);
//         y0_low = vget_low_f16(y0);

//         y0_low = vadd_f16(y0_high, y0_low);
//         vst1_f16(r, y0_low);

//         Y[j] += r[0] + r[1] + r[2] + r[3];
//       }
//     }
//   }
// }

// void sgemv_transpose_neon_fp16(const __fp16 *A, const __fp16 *X, __fp16 *Y,
//                                uint32_t rows, uint32_t cols, float alpha,
//                                float beta) {

//   const float16x8_t v_beta = vmovq_n_f16(beta);
//   const float16x8_t v_alpha = vmovq_n_f16(alpha);

//   if (cols % 32 == 0) {

//     for (unsigned int j = 0; j < cols; j += 8) {
//       float16x8_t y0_7 = vld1q_f16(&Y[j]);
//       y0_7 = vmulq_f16(y0_7, v_beta);
//       vst1q_f16(&Y[j], y0_7);
//     }

//     for (unsigned int i = 0; i < rows; ++i) {
//       __fp16 x = alpha * X[i];

//       for (unsigned int j = 0; j < cols; j += 32) {
//         __fp16 *__restrict y = &Y[j];

//         float16x8_t y0_7 = vld1q_f16(&y[0]);
//         float16x8_t y8_15 = vld1q_f16(&y[8]);
//         float16x8_t y16_23 = vld1q_f16(&y[16]);
//         float16x8_t y24_31 = vld1q_f16(&y[24]);

//         float16x8_t wvec0_7, wvec8_15, wvec16_23, wvec24_31;
//         const __fp16 *__restrict w;

//         w = &A[i * cols + j];

//         wvec0_7 = vld1q_f16(&w[0]);
//         wvec8_15 = vld1q_f16(&w[8]);
//         wvec16_23 = vld1q_f16(&w[16]);
//         wvec24_31 = vld1q_f16(&w[24]);

//         y0_7 = vfmaq_n_f16(y0_7, wvec0_7, x);
//         y8_15 = vfmaq_n_f16(y8_15, wvec8_15, x);
//         y16_23 = vfmaq_n_f16(y16_23, wvec16_23, x);
//         y24_31 = vfmaq_n_f16(y24_31, wvec24_31, x);

//         vst1q_f16(&y[0], y0_7);
//         vst1q_f16(&y[8], y8_15);
//         vst1q_f16(&y[16], y16_23);
//         vst1q_f16(&y[24], y24_31);
//       }
//     }
//     return;
//   } else if (cols % 16 == 0) {

//     for (unsigned int j = 0; j < cols; j += 8) {
//       float16x8_t y0_7 = vld1q_f16(&Y[j]);
//       y0_7 = vmulq_f16(y0_7, v_beta);
//       vst1q_f16(&Y[j], y0_7);
//     }

//     for (unsigned int i = 0; i < rows; ++i) {
//       __fp16 x = alpha * X[i];

//       for (unsigned int j = 0; j < cols; j += 16) {
//         __fp16 *__restrict y = &Y[j];

//         float16x8_t y0_7 = vld1q_f16(&y[0]);
//         float16x8_t y8_15 = vld1q_f16(&y[8]);

//         float16x8_t wvec0_7, wvec8_15;
//         const __fp16 *__restrict w;

//         w = &A[i * cols + j];

//         wvec0_7 = vld1q_f16(&w[0]);
//         wvec8_15 = vld1q_f16(&w[8]);

//         y0_7 = vfmaq_n_f16(y0_7, wvec0_7, x);
//         y8_15 = vfmaq_n_f16(y8_15, wvec8_15, x);

//         vst1q_f16(&y[0], y0_7);
//         vst1q_f16(&y[8], y8_15);
//       }
//     }
//     return;
//   } else if (cols % 8 == 0) {

//     for (unsigned int j = 0; j < cols; j += 8) {
//       float16x8_t y0_7 = vld1q_f16(&Y[j]);
//       y0_7 = vmulq_f16(y0_7, v_beta);
//       vst1q_f16(&Y[j], y0_7);
//     }

//     for (unsigned int i = 0; i < rows; ++i) {

//       __fp16 x = alpha * X[i];

//       for (unsigned int j = 0; j < cols; j += 8) {

//         float16x8_t y0_7 = vld1q_f16(&Y[j]);
//         float16x8_t wvec0_7 = vld1q_f16(&A[i * cols + j]);

//         y0_7 = vfmaq_n_f16(y0_7, wvec0_7, x);

//         vst1q_f16(&Y[j], y0_7);
//       }
//     }
//     return;
//   }
// }

// void saxpy_neon_fp16(const unsigned int N, const float alpha, const __fp16
// *X,
//                      __fp16 *Y) {

//   const float16x8_t v_alphaX8 = vmovq_n_f16(alpha);
//   const float16x4_t v_alphaX4 = vmov_n_f16(alpha);

//   unsigned int idx = 0;

//   // processing batch of 8
//   for (; (N - idx) >= 8; idx += 8) {
//     float16x8_t x = vld1q_f16(&X[idx]);
//     float16x8_t y = vld1q_f16(&Y[idx]);

//     // alpha*X + Y -> mulacc
//     float16x8_t mulacc = vfmaq_f16(y, v_alphaX8, x);
//     vst1q_f16(&Y[idx], mulacc);
//   }

//   // processing remaining batch of 4
//   for (; (N - idx) >= 4; idx += 4) {
//     float16x4_t x = vld1_f16(&X[idx]);
//     float16x4_t y = vld1_f16(&Y[idx]);

//     // alpha*X + Y -> mulacc
//     float16x4_t mulacc = vfma_f16(y, v_alphaX4, x);
//     vst1_f16(&Y[idx], mulacc);
//   }

//   // pocessing remaining values
//   for (; idx < N; idx++)
//     Y[idx] = Y[idx] + alpha * X[idx];
// }

// __fp16 sdot_neon_fp16(const unsigned int N, const __fp16 *X, const __fp16 *Y)
// {

//   float16x8_t accX8 = vmovq_n_f16(0);
//   float16x4_t accX4 = vmov_n_f16(0);

//   unsigned int idx = 0;
//   __fp16 ret = 0;

//   // processing batch of 8
//   for (; (N - idx) >= 8; idx += 8) {
//     float16x8_t x = vld1q_f16(&X[idx]);
//     float16x8_t y = vld1q_f16(&Y[idx]);

//     // x*y + accX8 -> accX8
//     accX8 = vfmaq_f16(accX8, x, y);
//   }

//   // check at least one batch of 8 is processed
//   if (N - 8 >= 0) {
//     __fp16 result[8];
//     vst1q_f16(result, accX8);
//     for (unsigned int i = 0; i < 8; i++)
//       ret += result[i];
//   }

//   // processing remaining batch of 4
//   for (; (N - idx) >= 4; idx += 4) {
//     float16x4_t x = vld1_f16(&X[idx]);
//     float16x4_t y = vld1_f16(&Y[idx]);

//     // x*y + accX4 -> accX4
//     accX4 = vfma_f16(accX4, x, y);
//   }

//   // check at least one batch of 4 is processed
//   if (N % 8 >= 4) {
//     __fp16 result[4];
//     vst1_f16(result, accX4);
//     ret += result[0] + result[1] + result[2] + result[3];
//   }

//   // pocessing remaining values
//   for (; idx < N; idx++)
//     ret += X[idx] * Y[idx];

//   return ret;
// }

// __fp16 snrm2_neon_fp16(const unsigned int N, const __fp16 *X) {

//   float16x8_t accX8 = vmovq_n_f16(0);
//   float16x4_t accX4 = vmov_n_f16(0);

//   unsigned int idx = 0;
//   __fp16 ret = 0;

//   // processing batch of 8
//   for (; (N - idx) >= 8; idx += 8) {
//     float16x8_t x = vld1q_f16(&X[idx]);

//     // x*x + accX8 -> accX8
//     accX8 = vfmaq_f16(accX8, x, x);
//   }

//   // check at least one batch of 8 is processed
//   if (N - 8 >= 0) {
//     __fp16 result[8];
//     vst1q_f16(result, accX8);
//     for (unsigned int i = 0; i < 8; i++)
//       ret += result[i];
//   }

//   // processing remaining batch of 4
//   for (; (N - idx) >= 4; idx += 4) {
//     float16x4_t x = vld1_f16(&X[idx]);

//     // x*x + accX4 -> accX4
//     accX4 = vfma_f16(accX4, x, x);
//   }

//   // check at least one batch of 4 is processed
//   if (N % 8 >= 4) {
//     __fp16 result[4];
//     vst1_f16(result, accX4);
//     ret += result[0] + result[1] + result[2] + result[3];
//   }

//   // pocessing remaining values
//   for (; idx < N; idx++)
//     ret += X[idx] * X[idx];

//   return ret;
// }

// void sscal_neon_fp16(const unsigned int N, __fp16 *X, const float alpha) {
//   const float16x8_t v_alphaX8 = vmovq_n_f16(alpha);
//   const float16x4_t v_alphaX4 = vmov_n_f16(alpha);

//   unsigned int idx = 0;

//   // processing batch of 8
//   for (; (N - idx) >= 8; idx += 8) {
//     float16x8_t x = vld1q_f16(&X[idx]);

//     // alpha*X -> X
//     float16x8_t mulacc = vmulq_f16(v_alphaX8, x);
//     vst1q_f16(&X[idx], mulacc);
//   }

//   // processing remaining batch of 4
//   for (; (N - idx) >= 4; idx += 4) {
//     float16x4_t x = vld1_f16(&X[idx]);

//     // alpha*X -> X
//     float16x4_t mulacc = vmul_f16(v_alphaX4, x);
//     vst1_f16(&X[idx], mulacc);
//   }

//   // pocessing remaining values
//   for (; idx < N; idx++)
//     X[idx] = alpha * X[idx];
// }

// void scopy_neon_fp16(const unsigned int N, const __fp16 *X, __fp16 *Y) {

//   unsigned int idx = 0;

//   // processing batch of 8
//   for (; (N - idx) >= 8; idx += 8) {
//     float16x8_t batch = vld1q_f16(&X[idx]);
//     vst1q_f16(&Y[idx], batch);
//   }

//   // processing remaining batch of 4
//   for (; (N - idx) >= 4; idx += 4) {
//     float16x4_t batch = vld1_f16(&X[idx]);
//     vst1_f16(&Y[idx], batch);
//   }

//   // pocessing remaining values
//   for (; idx < N; idx++)
//     Y[idx] = X[idx];
// }

// unsigned int isamax_neon_fp16(const unsigned int N, const __fp16 *X) {

//   unsigned int retIdx;
//   __fp16 maxNum;

//   uint16_t indices[] = {0, 1, 2, 3, 4, 5, 6, 7};
//   uint16x8_t stride = vmovq_n_u16(8);
//   float16x8_t batch = vld1q_f16(&X[0]);
//   uint16x8_t curr_index = vld1q_u16(indices);
//   uint16x8_t max_index = curr_index;

//   unsigned int idx = 8;

//   // processing batch of 8
//   for (; (N - idx) >= 8; idx += 8) {
//     float16x8_t values = vld1q_f16(&X[idx]);
//     curr_index = vaddq_u16(curr_index, stride);

//     // comparison
//     uint16x8_t mask = vcgtq_f16(batch, values);

//     // blend values and indices based on the mask
//     batch = vbslq_f16(mask, batch, values);
//     max_index = vbslq_u16(mask, max_index, curr_index);
//   }

//   // storing indices and max values
//   __fp16 maxVal[8];
//   vst1q_f16(maxVal, batch);
//   vst1q_u16(indices, max_index);

//   // getting the index of the maxima
//   maxNum = maxVal[0];
//   retIdx = max_index[0];
//   for (int i = 1; i < 8; i++) {
//     if (maxVal[i] > maxNum) {
//       maxNum = maxVal[i];
//       retIdx = max_index[i];
//     }
//   }

//   // processing remaining values
//   for (; idx < N; idx++) {
//     if (X[idx] > maxNum) {
//       maxNum = X[idx];
//       retIdx = idx;
//     }
//   }

//   return retIdx;
// }

// void sgemm_neon_fp16(const __fp16 *A, const __fp16 *B, __fp16 *C, uint32_t M,
//                      uint32_t N, uint32_t K, float alpha, float beta,
//                      bool TransA, bool TransB) {

//   float16x8_t v_alpha = vmovq_n_f16(alpha);
//   float16x8_t v_beta = vmovq_n_f16(beta);

//   // performing beta*C
//   unsigned int idx = 0;
//   unsigned int size = M * N;
//   for (; idx < (size - idx) >= 8; idx += 8) {
//     float16x8_t c = vld1q_f16(&C[idx]);
//     c = vmulq_f16(v_beta, c);
//     vst1q_f16(&C[idx], c);
//   }

//   // remaining values if dimensions not a multiple of 8
//   for (; idx < size; idx++) {
//     C[idx] *= beta;
//   }

//   __fp16 r[4];

//   if (!TransA && TransB) {
//     __fp16 valsB[8];
//     __fp16 valsA[8];
//     for (unsigned int m = 0; m < M; m++) {
//       for (unsigned int n = 0; n < N; n++) {
//         float16x8_t sum = vmovq_n_f16(0);
//         unsigned int k = 0;
//         for (; (K - k) >= 8; k += 8) {
//           float16x8_t a = vld1q_f16(&A[m * K + k]);
//           float16x8_t b = vld1q_f16(&B[n * K + k]);
//           sum = vfmaq_f16(sum, a, b);
//         }

//         // remaining K values
//         if (k < K) {
//           unsigned int idx;
//           for (idx = k; idx < K; idx++) {
//             valsA[idx - k] = A[m * K + idx];
//             valsB[idx - k] = B[n * K + idx];
//           }
//           // to cover entire 128 bits (reset unused bits)
//           while (idx < (k + 8)) {
//             valsA[idx - k] = 0;
//             valsB[idx - k] = 0;
//             idx++;
//           }
//           // updating sum
//           float16x8_t a = vld1q_f16(valsA);
//           float16x8_t b = vld1q_f16(valsB);
//           sum = vfmaq_f16(sum, a, b);
//         }

//         sum = vmulq_f16(v_alpha, sum);

//         float16x4_t sum_high = vget_high_f16(sum);
//         float16x4_t sum_low = vget_low_f16(sum);

//         sum_low = vadd_f16(sum_high, sum_low);
//         vst1_f16(r, sum_low);

//         C[m * N + n] += r[0] + r[1] + r[2] + r[3];
//       }
//     }
//   } else if (TransA && !TransB) {
//     __fp16 valsB[8];
//     __fp16 valsC[8];
//     for (unsigned int k = 0; k < K; k++) {
//       for (unsigned int m = 0; m < M; m++) {
//         __fp16 a = alpha * A[k * M + m];
//         unsigned int n = 0;
//         for (; (N - n) >= 8; n += 8) {
//           float16x8_t b = vld1q_f16(&B[k * N + n]);

//           // load previously calculated C
//           float16x8_t c = vld1q_f16(&C[m * N + n]);
//           c = vfmaq_n_f16(c, b, a);
//           vst1q_f16(&C[m * N + n], c);
//         }

//         // remaining N values
//         if (n < N) {
//           for (unsigned int idx = n; idx < N; idx++) {
//             valsB[idx - n] = B[k * N + idx];

//             // load previously calculated C
//             valsC[idx - n] = C[m * N + idx];
//           }
//           float16x8_t b = vld1q_f16(valsB);
//           float16x8_t c = vld1q_f16(valsC);
//           c = vfmaq_n_f16(c, b, a);
//           vst1q_f16(valsC, c);

//           for (unsigned int idx = n; idx < N; idx++) {
//             C[m * N + idx] = valsC[idx - n];
//           }
//         }
//       }
//     }
//   } else if (!TransA && !TransB) {

//     __fp16 valsB[8];
//     __fp16 valsC[8];
//     for (unsigned int k = 0; k < K; k++) {
//       for (unsigned int m = 0; m < M; m++) {
//         __fp16 a = alpha * A[m * K + k];
//         unsigned int n = 0;
//         for (; (N - n) >= 8; n += 8) {
//           float16x8_t b = vld1q_f16(&B[k * N + n]);

//           // load previously calculated C
//           float16x8_t c = vld1q_f16(&C[m * N + n]);
//           c = vfmaq_n_f16(c, b, a);
//           vst1q_f16(&C[m * N + n], c);
//         }

//         // remaining N values
//         if (n < N) {
//           for (unsigned int idx = n; idx < N; idx++) {
//             valsB[idx - n] = B[k * N + idx];

//             // load previously calculated C
//             valsC[idx - n] = C[m * N + idx];
//           }
//           float16x8_t b = vld1q_f16(valsB);
//           float16x8_t c = vld1q_f16(valsC);
//           c = vfmaq_n_f16(c, b, a);
//           vst1q_f16(valsC, c);

//           for (unsigned int idx = n; idx < N; idx++) {
//             C[m * N + idx] = valsC[idx - n];
//           }
//         }
//       }
//     }
//   } else { // TransA && TransB
//     __fp16 vals[8];
//     for (unsigned int n = 0; n < N; n++) {
//       for (unsigned int k = 0; k < K; k++) {

//         __fp16 b = alpha * B[n * K + k];
//         unsigned int m = 0;
//         for (; (M - m) >= 8; m += 8) {
//           float16x8_t a = vld1q_f16(&A[k * M + m]);
//           a = vmulq_n_f16(a, b);
//           vst1q_f16(vals, a);

//           // calculations for all M values
//           for (unsigned int idx = m; idx < m + 8; idx++)
//             C[idx * N + n] += vals[idx - m];
//         }

//         // remaining when M is not a multiple of 8
//         if (m < M) {
//           for (idx = m; idx < M; idx++) {
//             vals[idx - m] = A[k * M + idx];
//           }

//           float16x8_t a = vld1q_f16(vals);
//           a = vmulq_n_f16(a, b);
//           vst1q_f16(vals, a);

//           // calculations for all remaining M values
//           for (idx = m; idx < M; idx++)
//             C[idx * N + n] += vals[idx - m];
//         }
//       }
//     }
//   }
// }
// #endif

// } // namespace nntrainer::neon

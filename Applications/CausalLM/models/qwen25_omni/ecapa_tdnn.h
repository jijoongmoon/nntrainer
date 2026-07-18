// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   ecapa_tdnn.h
 * @date   18 July 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  ECAPA-TDNN speaker encoder for the Qwen2.5-Omni Token2Wav DiT
 *         (mel [T,80] -> 128-d speaker embedding), host-side FP32.
 *
 * Port of HF ECAPA_TimeDelayNet (transformers 4.57.6), validated against
 * /tmp/omni_dit_dump/ecapa_out.npy via the numpy reference (6.6e-7 pos /
 * 8.9e-8 neg). Structure: TDNN(K5) -> 3 x SE-Res2Net(scale 2, dilations
 * 2/3/4) -> MFA concat(3x256)+1x1 -> attentive-stats pooling (eps 1e-12,
 * global-context concat, per-channel softmax over time) -> fc 1536->128.
 * All convs stride 1, padding "same" REFLECT; conv+ReLU, no norm layers.
 * The CFG null branch is forward(zeros) — NOT a zero vector.
 *
 * Weights: ecapa.bin (40 tensors, raw [C_out, C_in, K] + bias, in the fixed
 * order emitted by token2wav_dit_converter.py).
 */

#ifndef __ECAPA_TDNN_H__
#define __ECAPA_TDNN_H__

#include <string>
#include <vector>

namespace causallm {

/**
 * @brief ECAPA-TDNN speaker encoder (host-side, no nntrainer graph)
 */
class EcapaTdnn {
public:
  /** @brief Load ecapa.bin; throws on open/short-read. */
  void load(const std::string &path);

  /** @brief true once load() succeeded. */
  bool loaded() const { return loaded_; }

  /**
   * @brief mel [T, 80] row-major (time-major, natural HF ref_mel layout)
   * @return speaker embedding [128]
   */
  std::vector<float> forward(const float *mel, unsigned int T) const;

private:
  struct Conv {
    std::vector<float> w; /**< [cout * cin * k] */
    std::vector<float> b; /**< [cout] */
    unsigned int cout, cin, k, dil;
  };

  /** same-padding reflect conv1d + optional ReLU; in [cin*T] -> out [cout*T] */
  static void conv1d(const Conv &c, const float *in, unsigned int T,
                     float *out, bool relu);

  Conv blk0;                    /**< TDNN K5, 80 -> 256 */
  Conv tdnn1[3], res2[3], tdnn2[3], se1[3], se2[3]; /**< SE-Res2Net x3 */
  Conv mfa;                     /**< 1x1, 768 -> 768 */
  Conv asp_tdnn;                /**< 1x1, 2304 -> 64 (+ReLU, then tanh) */
  Conv asp_conv;                /**< 1x1, 64 -> 768 (no act) */
  Conv fc;                      /**< 1x1, 1536 -> 128 (no act) */
  bool loaded_ = false;
};

} // namespace causallm

#endif /* __ECAPA_TDNN_H__ */

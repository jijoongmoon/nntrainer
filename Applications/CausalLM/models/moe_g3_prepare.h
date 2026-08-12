// SPDX-License-Identifier: Apache-2.0
/**
 * @file moe_g3_prepare.h
 * @brief Load-time hook interface for the NNTR_MOE_G3 table build + payload
 *        repack. Transformer::repack_weight() dynamic_casts each layer to
 *        this, so the one-time fragment repack runs during model load (where
 *        vLLM runs its Marlin repack) instead of on the first prefill
 *        chunk's timer. Pure interface: no link dependency between the
 *        generic transformer and the MoE module.
 */
#ifndef __MOE_G3_PREPARE_H__
#define __MOE_G3_PREPARE_H__

namespace nntrainer {
class RunLayerContext;
}

namespace causallm {

struct MoeG3Prepare {
  virtual ~MoeG3Prepare() = default;
  /** @brief Build MoE pointer tables and (under NNTR_MOE_G3) repack payloads. */
  virtual void prepareMoeG3(nntrainer::RunLayerContext &context) = 0;
};

} // namespace causallm

#endif // __MOE_G3_PREPARE_H__

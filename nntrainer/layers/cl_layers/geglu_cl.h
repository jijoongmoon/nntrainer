// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   geglu_cl.h
 * @date   08 Jun 2026
 * @brief  GPU GeGLU activation: gelu_tanh(in1) * in2. Mirrors SwiGLULayerCl but
 *         uses the tanh-approx GELU gate (Gemma2 / gelu_pytorch_tanh) instead
 *         of SiLU. {gate, up} -> gelu_tanh(gate) * up.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __GEGLU_LAYER_CL_H__
#define __GEGLU_LAYER_CL_H__

#include <cl_context.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <layer_impl_cl.h>
#include <node_exporter.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>
#include <utility>

namespace nntrainer {

/**
 * @brief A GeGLU layer (gelu_tanh(gate) * up)
 */
class GeGLULayerCl final : public LayerImplCl {

public:
  /**
   * @brief Construct a new GeGLU layer object
   */
  GeGLULayerCl() :
    LayerImplCl(), geglu_props(props::Print(), props::SkipPrefill()) {}

  /**
   * @brief Destroy the GeGLU layer object
   */
  ~GeGLULayerCl() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return GeGLULayerCl::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "geglu";

  /**
   * @brief common process for forwarding / incremental_forwarding
   */
  void gegluProcess(Tensor const &in1, Tensor const &in2, Tensor &result,
                    unsigned int active_rows, unsigned int row_offset);

  /**
   * @brief geglu computation (fp32)
   */
  void geglu_cl(float *matAdata, float *vecXdata, float *vecYdata,
                unsigned int dim1, unsigned int dim2, bool svm = true,
                void *resident_out = nullptr);

#ifdef ENABLE_FP16
  /**
   * @brief geglu computation (fp16)
   * @details resident_out / in1_clmem / in2_clmem bind that argument as a
   *          device cl_mem (the tensor's planner sub-buffer, static GPU_CLMEM
   *          residency) instead of the SVM pointer; mixing cl_mem and SVM args
   *          in one kernel is valid. A cl_mem-bound input skips its SVM unmap
   *          (its SVM shadow was never written by the cl_mem producer).
   */
  void geglu_cl_fp16(_FP16 *matAdata, _FP16 *vecXdata, _FP16 *vecYdata,
                     unsigned int dim1, unsigned int dim2, bool svm = true,
                     void *resident_out = nullptr, bool skip_out_map = false,
                     void *in1_clmem = nullptr, void *in2_clmem = nullptr,
                     unsigned int row_off = 0);
#endif

  /**
   * @brief Register OpenCL kernels for GeGLU layer.
   */
  static bool registerClKernels(ClContext &cl_context);

private:
  bool skip_prefill = false; /**< skip compute during prefill (Gemma4 KV-share) */

  std::tuple<props::Print, props::SkipPrefill>
    geglu_props; /**< geglu layer properties */

  static std::vector<ClContext::SharedPtrClKernel> &getLayerKernelPtrs();

  enum Kernels { GEGLU_CL, GEGLU_CL_FP16 }; /** kernels enum */
};

} // namespace nntrainer

#endif /* __GEGLU_LAYER_CL_H__ */

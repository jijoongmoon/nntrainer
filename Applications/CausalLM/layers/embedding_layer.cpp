// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding.cpp
 * @date   04 March 2021
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This embedding layer supports FP32/FP16/Q6_K data type only.
 */

#include <layer_prof.h>
#if defined(ENABLE_OPENCL)
// OpenCL GPU residency handoff (clmem_raise_cl / cl_svm_unmap_force). Guarded so
// the FP32 CPU build (enable-opencl=false) compiles as a host-only embedding. [T12]
#include <blas_kernel_interface.h>
#include <blas_kernels.h>
#endif
#include <embedding_layer.h>
#include <layer_context.h>
#include <memory_data.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <thread_manager.h>
#include <util_func.h>

#include <vector>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum EmbeddingParams { weight };

EmbeddingLayer::EmbeddingLayer() :
  LayerImpl(),
  embedding_props(nntrainer::props::InDim(), nntrainer::props::OutDim(),
                  nntrainer::props::Scale()),
  weight_idx(std::numeric_limits<unsigned>::max()) {}

void EmbeddingLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Embedding layer takes only one input";

  // Token IDs are integers — embedding caller is expected to provide FP32
  // input (e.g., via an explicit input layer with input_dtype=FP32). The
  // historical "must be FP32" check is removed so FP16-activation models
  // still construct, but the actual lookup expects integer-valued data.

  const nntrainer::TensorDim &input_dim =
    context.getInputDimensions()[SINGLE_INOUT_IDX];
  NNTR_THROW_IF(input_dim.channel() != 1, std::invalid_argument)
    << "Embedding layer takes only one for channel size";

  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::NONE;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);

  size_t in_dim =
    static_cast<size_t>(std::get<nntrainer::props::InDim>(embedding_props));
  size_t out_dim =
    static_cast<size_t>(std::get<nntrainer::props::OutDim>(embedding_props));

  nntrainer::TensorDim output_dim = input_dim;

  // output_dim expected as hidden x num input (batch size)
  output_dim.height(input_dim.width());
  output_dim.width(out_dim);
  output_dim.setTensorType(
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({output_dim});

  nntrainer::TensorDim dim = output_dim;

  dim.setTensorType({context.getFormat(), context.getWeightDataType()});

  dim.height(in_dim);
  dim.width(out_dim);
  dim.batch(1);

  weight_idx = context.requestWeight(
    dim, weight_initializer, weight_regularizer, weight_regularizer_constant,
    weight_decay, "Embedding", true);
}

void EmbeddingLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, embedding_props);
  LayerImpl::setProperty(remain_props);
}

void EmbeddingLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  nntrainer::LayerProfScope _prof("embedding_fwd", false);
}

void EmbeddingLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  nntrainer::LayerProfScope _prof("embedding", (to - from) == 1);

  /// @todo get input and output dimension from input_ and hidden itself
  unsigned int in_dim = std::get<nntrainer::props::InDim>(embedding_props);
  unsigned int out_dim = std::get<nntrainer::props::OutDim>(embedding_props);
  float scale = std::get<nntrainer::props::Scale>(embedding_props).empty()
                  ? 1.0f
                  : std::get<nntrainer::props::Scale>(embedding_props).get();
  unsigned int _from = from;

  nntrainer::Tensor &weight = context.getWeight(weight_idx);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  nntrainer::TensorDim out_tensor_dim =
    nntrainer::TensorDim({1, 1, 1, out_dim}, hidden_.getTensorType());

  unsigned int b_size = input_.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    float *in_data =
      input_.getAddress<float>(b * input_.getDim().getFeatureLen());
    nntrainer::Tensor batchsliced_hidden = hidden_.getBatchSlice(b, 1);

    int iter = to - from;

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // Device-only activation pool (NNTR_CUDA_DEV_ACT): the PLE output is real
    // device memory (not host-addressable). Dequant into a host staging buffer
    // and push it H2D on the backend stream -- the CUDA mirror of the
    // clmem_raise_cl device-upload OpenCL already does below.
    // Persistent + PINNED host staging (was a local std::vector). Under
    // CUDA-graph stream capture a local vector fails twice: (a) a pageable
    // cudaMemcpyAsync is NOT capturable, and (b) the vector is freed when this
    // function returns, but the captured graph REPLAYS afterwards -- it would
    // copy from freed memory => garbage. A process-lifetime pinned (cudaHostAlloc)
    // buffer is capturable and survives the replay. Grows monotonically (decode
    // iter==1; prefill iter<=max_seq_len); single sequence (b_size==1) so one
    // shared buffer per layer is sufficient.
    static _FP16 *emb_stage = nullptr;
    static size_t emb_stage_cap = 0; // capacity in _FP16 elements
    bool emb_dev_only = false;
    if (hidden_.getDataType() == nntrainer::TensorDim::DataType::FP16) {
      cudaPointerAttributes pa{};
      emb_dev_only =
        cudaPointerGetAttributes(&pa, batchsliced_hidden.getData<_FP16>()) ==
          cudaSuccess &&
        pa.type == cudaMemoryTypeDevice;
      cudaGetLastError();
      if (emb_dev_only) {
        size_t need = (size_t)iter * out_dim;
        if (need > emb_stage_cap) {
          if (emb_stage)
            cudaFreeHost(emb_stage);
          cudaHostAlloc((void **)&emb_stage, need * sizeof(_FP16),
                        cudaHostAllocDefault);
          emb_stage_cap = need;
        }
      }
    }
#endif

    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, static_cast<size_t>(iter), [&](size_t i) {
      size_t embed_idx = static_cast<size_t>(in_data[i]);
      if (embed_idx >= in_dim) {
        throw std::invalid_argument("input word index is greater than in_dim");
      }

      nntrainer::Tensor cur_weight =
        weight.getSharedDataTensor(out_tensor_dim, out_dim * embed_idx);
      nntrainer::Tensor out_tensor =
        batchsliced_hidden.getSharedDataTensor(out_tensor_dim, out_dim * (i));

      const auto wt = weight.getDataType();
      if (wt == nntrainer::TensorDim::DataType::Q6_K ||
          wt == nntrainer::TensorDim::DataType::Q4_0) {
        // dequantize_row_q{6_K,4_0} ALWAYS writes out_dim FP32 values. In an
        // FP16-activation run out_tensor is FP16, so writing FP32 straight in
        // (the old `out_tensor.getData()` == float*) overruns the buffer 2x and
        // corrupts every value => garbage PLE row added to every layer =>
        // prompt-independent garbage output. Mirror TieWordEmbedding: dequant
        // into an FP32 scratch, then cast into the output's real dtype, folding
        // the embed scale.
        std::vector<float> tmp(out_dim);
        if (wt == nntrainer::TensorDim::DataType::Q6_K) {
          int num_blocks_per_row = (weight.width() + 256 - 1) / 256;
          nntrainer::dequantize_row_q6_K(
            (void *)((char *)weight.getData<uint8_t>() +
                     (210 * num_blocks_per_row) * embed_idx),
            tmp.data(), out_dim);
        } else {
          int num_blocks_per_row = (weight.width() + 32 - 1) / 32;
          nntrainer::dequantize_row_q4_0(
            (void *)((char *)weight.getData<uint8_t>() +
                     (18 * num_blocks_per_row) * embed_idx),
            tmp.data(), out_dim);
        }
        if (out_tensor.getDataType() == nntrainer::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
          _FP16 *o =
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
            emb_dev_only ? (emb_stage + (size_t)i * out_dim) :
#endif
                         out_tensor.getData<_FP16>();
          for (unsigned int k = 0; k < (unsigned int)out_dim; ++k)
            o[k] = static_cast<_FP16>(tmp[k] * scale);
#else
          throw std::invalid_argument("FP16 out_tensor requires ENABLE_FP16");
#endif
        } else {
          float *o = out_tensor.getData<float>();
          for (unsigned int k = 0; k < (unsigned int)out_dim; ++k)
            o[k] = tmp[k] * scale;
        }
      } else if (wt == nntrainer::TensorDim::DataType::FP32 &&
                 out_tensor.getDataType() ==
                   nntrainer::TensorDim::DataType::FP16) {
        // FP32 weight row -> FP16 activation needs an explicit narrowing cast;
        // copyData would byte-copy out_dim*4 bytes into an out_dim*2 buffer.
#ifdef ENABLE_FP16
        const float *src = cur_weight.getData<float>();
        _FP16 *o =
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
          emb_dev_only ? (emb_stage + (size_t)i * out_dim) :
#endif
                       out_tensor.getData<_FP16>();
        for (unsigned int k = 0; k < (unsigned int)out_dim; ++k)
          o[k] = static_cast<_FP16>(src[k] * scale);
#else
        throw std::invalid_argument("FP16 out_tensor requires ENABLE_FP16");
#endif
      } else {
        out_tensor.copyData(cur_weight);
        if (scale != 1.0f) {
          out_tensor.multiply_i(scale);
        }
      }
    });

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // CUDA mirror of clmem_raise_cl: push the host-dequantized PLE rows into the
    // device-only output on the backend stream (ordered before the GPU consumer).
    if (emb_dev_only)
      cudaMemcpyAsync(batchsliced_hidden.getData<_FP16>(), emb_stage,
                      (size_t)iter * out_dim * sizeof(_FP16),
                      cudaMemcpyHostToDevice,
                      nntrainer::cuda::StreamManager::Global().GetStream());
#endif

#ifdef DEBUG
    std::cout << context.getName() << " : "
              << "\n input:" << input_ << "\n weight: " << weight
              << "\n hidden: " << hidden_ << std::endl;
#endif
  }

  // The embedding row is written on the HOST (dequant loop above) into the
  // output activation. When the output is SVM-resident, hand the buffer back
  // to the device so the GPU consumer (e.g. gemma4 per_layer_input_sum
  // addition) reads fresh data instead of a stale host-mapped shadow. This is
  // looked up EVERY token (prefill + each decode step), so a missing handoff
  // corrupts every step. No-op when not SVM-resident. Mirror TieWordEmbedding.
#if defined(ENABLE_OPENCL) && defined(ENABLE_FP16)
  // SVM residency hand-back is FP16-OpenCL-only; guard so the FP32/no-OpenCL
  // build compiles as host-only. [T12]
  {
    const auto h_md = hidden_.getMemoryData();
    if (h_md && h_md->isSVM())
      nntrainer::cl_svm_unmap_force(hidden_.getData<uint8_t>());
  }
#endif
  // GPU_CLMEM residency: upload host-written rows into the planner cl_mem
  // sub-buffer so GPU consumers read fresh device memory. No-op when the class
  // is SVM (handled above).
#if defined(ENABLE_OPENCL)
  // GPU cl_mem device-upload; no-op on the no-OpenCL host build. [T12]
  nntrainer::clmem_raise_cl(
    hidden_, (unsigned int)((size_t)(to - from) * out_dim *
                            hidden_.getDim().getDataTypeSize()));
#endif
}

void EmbeddingLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for Embedding layer is not supported");
}

void EmbeddingLayer::calcGradient(nntrainer::RunLayerContext &context) {}

void EmbeddingLayer::exportTo(nntrainer::Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(embedding_props, method, this);
}

void EmbeddingLayer::save(std::ofstream &file,
                          nntrainer::RunLayerContext &run_context, bool opt_var,
                          ml::train::ExecutionMode mode, bool trainable,
                          nntrainer::TensorDim::DataType dtype,
                          ml::train::ISA target_isa) const {
  // @note shared weights are only be saved at the first access
  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    if (run_context.isGradientFirstAccess(i)) {
      auto &weight = run_context.getWeight(i);
      if (dtype == nntrainer::TensorDim::DataType::NONE ||
          weight.getDataType() == dtype)
        weight.save(file);
      else {
        NNTR_THROW_IF(weight.getDataType() !=
                        nntrainer::TensorDim::DataType::FP32,
                      std::runtime_error)
          << "Save with quantization only supports for FP32 weight.";
        ///@note The codelines below can be replaced with quantizer's
        /// quantize()
        nntrainer::TensorDim dim = weight.getDim();
        unsigned int K = dim.height();
        unsigned int N = dim.width();

        if (dtype == nntrainer::TensorDim::DataType::Q4_0) {

          // Skip quantization for bias-like tensors (1D with height == 1)
          // as they are not suitable for Q4_0 block quantization
          if (K == 1) {
            weight.save(file);
          } else {
            NNTR_THROW_IF(N % 32 != 0, std::invalid_argument)
              << "Q4_0 embedding quantization requires width to be "
                 "divisible by 32, but got width="
              << N;
            //////////////////////////////////////////////////////////////////
            ///@note Please note that Embedding layer doesn't need to be
            /// transposed!
            //////////////////////////////////////////////////////////////////
            nntrainer::Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                           {nntrainer::Tformat::NCHW, dtype});
            nntrainer::quantize_q4_0(weight.getData<float>(),
                                     quant_weight.getData<uint8_t>(), K, N,
                                     nullptr);
            quant_weight.save(file);
          }
        } else if (dtype == nntrainer::TensorDim::DataType::Q6_K) {
          //////////////////////////////////////////////////////////////////
          ///@note Please note that Embedding layer doesn't need to be
          /// transposed!
          //////////////////////////////////////////////////////////////////
          nntrainer::Tensor quant_weight(dim.batch(), dim.channel(), K, N,
                                         {nntrainer::Tformat::NCHW, dtype});
          nntrainer::quantize_q6_K(weight.getData<float>(),
                                   quant_weight.getData<uint8_t>(), K, N,
                                   nullptr);
          quant_weight.save(file);
        } else {
          NNTR_THROW_IF(true, std::runtime_error)
            << "This dtype is not supported in save with quantization";
        }
      }
    }
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_embedding_layer() {
  auto layer = new EmbeddingLayer();
  std::cout << "embedding layer created\n";
  return layer;
}

void destroy_embedding_layer(nntrainer::Layer *layer) {
  std::cout << "embeddinglayer is deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_embedding_layer,
                                                   destroy_embedding_layer};
}

#endif

} // namespace causallm

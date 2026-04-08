// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   litert_graph.cpp
 * @date   08 Apr 2026
 * @brief  LiteRT-LM Graph Layer implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author NNTrainer Contributors
 * @bug    No known bugs except for NYI items
 */

#include "litert_graph.h"
#include <chrono>
#include <iostream>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#ifdef ENABLE_LITERT_LM
#include "litert_context.h"
#include "runtime/engine/io_types.h"
#endif

namespace nntrainer {

LiteRTGraph::LiteRTGraph() :
  LayerImpl(),
  graph_props({}, {}, props::FilePath()),
  is_session_created(false) {}

LiteRTGraph::~LiteRTGraph() {
#ifdef ENABLE_LITERT_LM
  session_.reset();
#endif
}

void LiteRTGraph::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, graph_props);
  LayerImpl::setProperty(remain_props);
}

void LiteRTGraph::finalize(InitLayerContext &context) {
  model_path = std::get<props::FilePath>(graph_props).get();

  auto &dims = std::get<std::vector<props::TensorDimension>>(graph_props);
  t_dims.assign(dims.begin(), dims.end());

  auto &t_dtype = std::get<std::vector<props::TensorDataType>>(graph_props);

  auto engine = context.getComputeEngineType();

  if (t_dtype.empty()) {
    t_dtype.resize(t_dims.size());
    for (auto &t : t_dims) {
      t_dtype.push_back(context.getActivationDataType());
    }
  }

  std::vector<TensorDim> out_dim;

  for (unsigned int i = 0; i < t_dims.size(); ++i) {
    t_dims[i].setFormat(context.getFormat());
    t_dims[i].setDataType(t_dtype[i]);
    out_dim.push_back(t_dims[i]);
  }

  context.setOutputDimensions(out_dim);
}

void LiteRTGraph::forwarding(RunLayerContext &context, bool training) {
  auto start = std::chrono::high_resolution_clock::now();

#ifdef ENABLE_LITERT_LM
  // Get or create session from LiteRTContext
  if (!is_session_created || !session_) {
    // Access the LiteRT context to create a session
    auto &litert_ctx = LiteRTContext::Global();
    session_ = litert_ctx.createSession();
    if (!session_) {
      ml_loge("Failed to create LiteRT-LM session");
      return;
    }
    is_session_created = true;
    ml_logi("LiteRT-LM session created successfully");
  }

  // Execute inference
  if (!input_prompt_.empty()) {
    // Synchronous generation
    using LiteRTInput =
        std::variant<litert::lm::InputText, litert::lm::InputImage,
                     litert::lm::InputAudio, litert::lm::InputImageEnd,
                     litert::lm::InputAudioEnd>;
    std::vector<LiteRTInput> inputs;
    inputs.emplace_back(litert::lm::InputText(std::string(input_prompt_)));
    auto responses = session_->GenerateContent(std::move(inputs));

    if (responses.ok()) {
      const auto &texts = responses->GetTexts();
      if (!texts.empty()) {
        last_output_ = texts[0];
        ml_logi("LiteRT-LM generated %zu chars", last_output_.size());
      }

      // Extract benchmark info if available
      auto bench = session_->GetBenchmarkInfo();
      if (bench.ok()) {
        if (bench->GetTotalPrefillTurns() > 0) {
          auto prefill_turn = bench->GetPrefillTurn(0);
          if (prefill_turn.ok()) {
            last_prefill_tokens_ = prefill_turn->num_tokens;
            last_prefill_ms_ =
                absl::ToDoubleMilliseconds(prefill_turn->duration);
          }
        }
        if (bench->GetTotalDecodeTurns() > 0) {
          auto decode_turn = bench->GetDecodeTurn(0);
          if (decode_turn.ok()) {
            last_decode_tokens_ = decode_turn->num_tokens;
            last_decode_ms_ =
                absl::ToDoubleMilliseconds(decode_turn->duration);
          }
        }
      }
    } else {
      ml_loge("LiteRT-LM GenerateContent failed: %s",
              responses.status().message().data());
    }
  }

#else
  // Fallback when ENABLE_LITERT_LM is not defined: pass-through
  for (unsigned int i = 0; i < context.getNumOutputs(); ++i) {
    Tensor &input_ = context.getInput(i);
    Tensor &hidden_ = context.getOutput(i);
    if (!context.getInPlace()) {
      hidden_.copyData(input_);
    }
  }
  ml_logw("LiteRT-LM not enabled, forwarding is pass-through only");
#endif

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  ml_logd("LiteRTGraph::forwarding elapsed: %.2f ms", elapsed.count());
}

void LiteRTGraph::read(std::ifstream &file, RunLayerContext &run_context,
                       bool opt_var, ml::train::ExecutionMode mode,
                       bool trainable, TensorDim::DataType defineWeightDataType,
                       bool fsu, size_t start_offset, bool read_from_offset,
                       int file_fd) {
  // No-op: LiteRT-LM model is self-contained in .litertlm file
  // All weights and configurations are loaded via LiteRTContext::load()
}

void LiteRTGraph::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  exporter.saveResult(graph_props, method, this);
}

} // namespace nntrainer

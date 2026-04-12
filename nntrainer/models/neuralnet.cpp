/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	neuralnet.cpp
 * @date	04 December 2019
 * @brief	This is Neural Network Class
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "layer_context.h"
#include <compute_ops.h>
#include "model.h"
#include "model_common_properties.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <sstream>

#include <activation_realizer.h>
#include <adamw.h>
#include <common_properties.h>
#include <databuffer.h>
#include <flatten_realizer.h>
#include <ini_interpreter.h>
#include <ini_wrapper.h>
#include <input_realizer.h>
#include <model_loader.h>
#include <multiout_realizer.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <optimizer_context.h>
#include <optional>
#include <previous_input_realizer.h>
#include <profiler.h>
#include <recurrent_realizer.h>
#include <remap_realizer.h>
#include <slice_realizer.h>
#include <util_func.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <tflite_interpreter.h>
#endif

/**
 * @brief Internal enum values for nntrainer to summarize model accuracy & loss
 */
#define ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS 101
#define ML_TRAIN_SUMMARY_MODEL_VALID_LOSS 102
#define ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY 103

namespace nntrainer {

NeuralNetwork::NeuralNetwork() :
  model_props(props::LossType(), {}, {}, props::ClipGradByGlobalNorm(),
              props::LossScale()),
  model_flex_props(props::Epochs(), props::TrainingBatchSize(),
                   props::SavePath(), props::ContinueTrain(),
                   props::SaveBestPath(), props::MemoryOptimization(),
                   props::Fsu(), props::FsuPath(), props::FsuLookahead(),
                   props::TensorFormat(), props::ModelTensorDataType()),
  load_path(std::string()),
  epoch_idx(0),
  iter(0),
  loss(0.0f),
  data_buffers({nullptr, nullptr, nullptr}),
  initialized(false),
  compiled(false),
  loadedFromConfig(false),
  exec_mode(ExecutionMode::TRAIN),
  ct_engine(&Engine::Global()) {}

NeuralNetwork::NeuralNetwork(const Engine *ct_engine_) :
  model_props(props::LossType(), {}, {}, props::ClipGradByGlobalNorm(),
              props::LossScale()),
  model_flex_props(props::Epochs(), props::TrainingBatchSize(),
                   props::SavePath(), props::ContinueTrain(),
                   props::SaveBestPath(), props::MemoryOptimization(),
                   props::Fsu(), props::FsuPath(), props::FsuLookahead(),
                   props::TensorFormat(), props::ModelTensorDataType()),
  load_path(std::string()),
  epoch_idx(0),
  iter(0),
  loss(0.0f),
  data_buffers({nullptr, nullptr, nullptr}),
  initialized(false),
  compiled(false),
  loadedFromConfig(false),
  exec_mode(ExecutionMode::TRAIN),
  ct_engine(ct_engine_) {}

int NeuralNetwork::loadFromConfig(const std::string &config) {
  if (loadedFromConfig == true) {
    ml_loge("can not do loadFromConfig twice");
    return ML_ERROR_INVALID_PARAMETER;
  }

  ModelLoader loader(ct_engine);
  NeuralNetwork tempNet(*this);

  int status = loader.loadFromContext(tempNet);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  status = loader.loadFromConfig(config, tempNet);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  tempNet.loadedFromConfig = true;
  swap(tempNet, *this);

  return ML_ERROR_NONE;
}

unsigned int NeuralNetwork::getCurrentEpoch() {
#ifdef DEBUG
  ml_logd("[NNTrainer] Current epoch: %d", epoch_idx);
#endif
  return epoch_idx;
};

void NeuralNetwork::setProperty(const std::vector<std::string> &values) {
  auto left_props = loadProperties(values, model_props);
  setTrainConfig(left_props);
}

void NeuralNetwork::setTrainConfig(const std::vector<std::string> &values) {
  auto left_props = loadProperties(values, model_flex_props);
  NNTR_THROW_IF(left_props.size(), std::invalid_argument)
    << "Model has unparsed properties, size: " << left_props.size()
    << " of first element: " << left_props.front();
}

int NeuralNetwork::compile(ExecutionMode mode) {

  exec_mode = mode;

  std::string loss_type = std::get<props::LossType>(model_props).empty()
                            ? std::string()
                            : std::get<props::LossType>(model_props);

  auto &input_conn = std::get<std::vector<props::InputConnection>>(model_props);
  /// @note label layer might need to be treated in the similar way as well

  /// @todo make NetworkGraph compiled at the construction instead of having
  /// graph.compile(), neuralnetwork have ownership of list of layer nodes,
  /// which will be passed at compile time.

  std::vector<std::unique_ptr<GraphRealizer>> realizers;

  realizers.emplace_back(new PreviousInputRealizer(
    std::vector<Connection>(input_conn.begin(), input_conn.end())));
  realizers.emplace_back(new MultioutRealizer());
  realizers.emplace_back(new FlattenRealizer());
  realizers.emplace_back(new ActivationRealizer());

  for (auto &realizer : realizers) {
    graph_representation = realizer->realize(graph_representation);
  }

  bool fsu = std::get<props::Fsu>(model_flex_props);
  const std::string fsu_path = std::get<props::FsuPath>(model_flex_props);
  unsigned int lookahead = std::get<props::FsuLookahead>(model_flex_props);

  const std::string tensor_format =
    to_string(std::get<props::TensorFormat>(model_flex_props));

  const std::string tensor_type =
    to_string(std::get<props::ModelTensorDataType>(model_flex_props));

  model_graph =
    NetworkGraph(fsu, mode, fsu_path, lookahead, tensor_format, tensor_type);

  model_graph.setMemoryOptimizations(
    std::get<props::MemoryOptimization>(model_flex_props));
  for (auto &node : graph_representation) {
    if (auto &prop = std::get<props::ClipGradByGlobalNorm>(model_props);
        !prop.empty()) {
      node->setProperty({"clip_grad_by_norm=" + to_string(prop)});
    }
    if (auto &prop = std::get<props::LossScale>(model_props); !prop.empty()) {
      node->setProperty({"loss_scale=" + to_string(prop)});
    }
    model_graph.addLayer(node);
  }

  int status = model_graph.compile(loss_type);
  NN_RETURN_STATUS();

  compiled = true;

  return status;
}

int NeuralNetwork::initialize(ExecutionMode mode) {
  int status = ML_ERROR_NONE;

  if (mode != exec_mode) {
    if (mode == ExecutionMode::INFERENCE) {
      ml_logd("Execution mode mismatch : train mode @compile & inference mode "
              "@ initialize");
      exec_mode = mode;
    } else {
      NNTR_THROW_IF(exec_mode == ExecutionMode::TRAIN, std::invalid_argument)
        << "Execution mode mismatch : trying to train with compiled for "
           "inference";
    }
  }

  if (initialized) {
    ml_loge("Error: Initializing the model again");
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (!compiled) {
    ml_loge("Error: Need to compile first");
    return ML_ERROR_NOT_SUPPORTED;
  }

  unsigned int n_layers = (unsigned int)model_graph.size();

  ml_logd("initializing neural network, layer size: %d", n_layers);
  PROFILE_MEM_ANNOTATE("Initialize");

  auto &input_conn_prop =
    std::get<std::vector<props::InputConnection>>(model_props);
  auto &label_layer_prop =
    std::get<std::vector<props::LabelLayer>>(model_props);

  std::vector<Connection> input_conn(input_conn_prop.begin(),
                                     input_conn_prop.end());
  std::vector<std::string> label_layers;

  if (!label_layer_prop.empty()) {
    label_layers = std::vector<std::string>(label_layer_prop.begin(),
                                            label_layer_prop.end());
  }

  status = model_graph.initialize(
    exec_mode, input_conn,
    std::vector<Connection>(label_layers.begin(), label_layers.end()));
  NN_RETURN_STATUS();

  model_graph.setBatchSize(
    std::get<props::TrainingBatchSize>(model_flex_props));

  // If the execution mode is `train`, the optimizer and its relevant variables
  // are initialized. Throws an error if the optimizer is not set for training;
  // otherwise, it initializes
  if (exec_mode == ExecutionMode::TRAIN) {

    if (!opt) {
      ml_loge("Optimizer should be set before initialization for training.");
      return ML_ERROR_INVALID_PARAMETER;
    }
    /** TODO: update request of optimizer to be of same format as
     * Layer::requestTensor */
    opt->finalize();
    std::function<std::vector<TensorDim>(const TensorDim &)> cb =
      [this](const TensorDim &dim) {
        return opt->getOptimizerVariableDim(dim);
      };
    model_graph.requestOptimizerVariable(cb, true);
  }

  // Allocate weights
  model_graph.allocateWeights(exec_mode != ExecutionMode::INFERENCE);
  // enable this to save initialized weights for INFERENCE
  // model_graph.allocateWeights(true);

  initialized = true;

  if (!load_path.empty()) {
    load(load_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  }

  return status;
}

int NeuralNetwork::reinitialize() {
  int status = ML_ERROR_NONE;

  if (!initialized) {
    ml_loge("Error: Need to initialize first");
    return ML_ERROR_NOT_SUPPORTED;
  }

  unsigned int n_layers = (unsigned int)model_graph.size();

  ml_logd("reinitializing neural network, layer size: %d", n_layers);
  PROFILE_MEM_ANNOTATE("Reinitialize");

  auto &input_conn_prop =
    std::get<std::vector<props::InputConnection>>(model_props);
  auto &label_layer_prop =
    std::get<std::vector<props::LabelLayer>>(model_props);

  std::vector<Connection> input_conn(input_conn_prop.begin(),
                                     input_conn_prop.end());
  std::vector<std::string> label_layers;

  if (!label_layer_prop.empty()) {
    label_layers = std::vector<std::string>(label_layer_prop.begin(),
                                            label_layer_prop.end());
  }

  status = model_graph.reinitialize(
    input_conn,
    std::vector<Connection>(label_layers.begin(), label_layers.end()));
  NN_RETURN_STATUS();

  return status;
}

/**
 * @brief     free layers
 */
NeuralNetwork::~NeuralNetwork() {
  try {
    deallocate();
  } catch (const std::runtime_error &e) {
    std::cerr << "Error occurred during destroying NeuralNetwork: " << e.what()
              << std::endl;
  }

  /** if neuralnet open fd */
  if (model_file_fd != -1)
    close(model_file_fd);
}

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensors NeuralNetwork::forwarding(
  bool training, std::function<bool(void *userdata)> stop_cb, void *userdata) {

  unsigned int lookahead = std::get<props::FsuLookahead>(model_flex_props);
  bool fsu_mode = std::get<props::Fsu>(model_flex_props);
  if (fsu_mode) {
    for (unsigned int i = 0; i < lookahead; ++i) {
      model_graph.LoadTensors(i);
    }
  }
  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op =
    [this, stop_cb, lookahead, fsu_mode](std::shared_ptr<LayerNode> node,
                                         bool training) -> void {
    (void)this;
    PROFILE_MEM_ANNOTATE("Forwarding for layer: " + node->getName());

    auto f = std::get<0>(node->getExecutionOrder());
    if (exec_mode == ExecutionMode::TRAIN or
        (exec_mode == ExecutionMode::INFERENCE and !fsu_mode)) {
      model_graph.flushCacheExcept(f);
      node->forwarding(training);
    } else {
      /**
         currently, it supports FSU asynch mode for inference. The prcedure of
         FSU is below,

         Prerequests : This function is called node by node at the forwarding
         function in network graph.

         Step 1. If the execution order is the first (f==0) then, it will try
       to load tensors which used at layer 0.

         Step 2. It check whether these tensors from Step 1, then do the
                 forwarding of the first node.

         Step 3. Then check the look a head which says how many layer weights
       need to be loaded before running to hide overehad due to FSU,

         Step 4. Try to get the tesors by asking tensors for layers which is
       done by thread pool

         Step 5. Try to release the weights which has execution order less
       then f.

         Step n. repeat next layer starting with checking the tenosrs are
       loaded, and if it is loaded, then run forwarding. Every time it
       finishes the forwarding, ask load tensors for next n layers.

      **/
      model_graph.checkLoadComplete(f);
      node->forwarding(training);
      model_graph.inActive(f);
      model_graph.LoadTensors(f + lookahead);
    }
  };

  return model_graph.forwarding(training, forwarding_op, stop_cb, userdata);
}

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensors NeuralNetwork::forwarding(sharedConstTensors input,
                                             sharedConstTensors label,
                                             bool training) {
  auto current_batch = model_graph.getBatchSize();
  if (current_batch != input[0]->batch()) {
    model_graph.setBatchSize(input[0]->batch());
    current_batch = model_graph.getBatchSize();
  }

  NNTR_THROW_IF(input[0]->batch() != current_batch ||
                  (!label.empty() && label[0]->batch() != current_batch),
                std::logic_error)
    << "Error: mismatch in batchsize for data and model."
    << " input_batch: " << input[0]->batch()
    << " label_batch: " << label[0]->batch()
    << " target_batch: " << current_batch;

  model_graph.setInputsLabels(input, label);

  return forwarding(training);
}

sharedConstTensors NeuralNetwork::incremental_forwarding(
  unsigned int from, unsigned int to, bool training,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {

  unsigned int lookahead = std::get<props::FsuLookahead>(model_flex_props);
  bool fsu_mode = std::get<props::Fsu>(model_flex_props);

  if (fsu_mode) {
    for (unsigned int i = 0; i < lookahead; ++i) {
      model_graph.LoadTensors(i);
    }
  }

  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op =
    [this, from, to, stop_cb, fsu_mode,
     lookahead](std::shared_ptr<LayerNode> node, bool training) -> void {
    PROFILE_MEM_ANNOTATE("Forwarding for layer: " + node->getName());

    auto f = std::get<0>(node->getExecutionOrder());
    if (exec_mode == ExecutionMode::TRAIN or
        (exec_mode == ExecutionMode::INFERENCE and !fsu_mode)) {
      // auto start_layer =
      //      std::chrono::high_resolution_clock::now(); // log the
      //      start_prefill time
      model_graph.flushCacheExcept(f);
      node->incremental_forwarding(from, to, training);
      // auto end_layer =
      //  std::chrono::high_resolution_clock::now(); // log th
      //   auto duration_ =
      //   std::chrono::duration_cast<std::chrono::nanoseconds>(end_layer-start_layer);
      // std::cout << node->getName() <<" : "<< duration_.count()<<"
      // ns"<<std::endl;
    } else {
      model_graph.checkLoadComplete(f);
      node->incremental_forwarding(from, to, training);
      model_graph.inActive(f);
      model_graph.LoadTensors(f + lookahead);
    }
  };

  return model_graph.incremental_forwarding(from, to, training, forwarding_op,
                                            stop_cb, userdata);
}

sharedConstTensors
NeuralNetwork::incremental_forwarding(unsigned int from, unsigned int to,
                                      sharedConstTensors input,
                                      sharedConstTensors label, bool training) {
  auto current_batch = model_graph.getBatchSize();
  NNTR_THROW_IF(input[0]->batch() != current_batch ||
                  (!label.empty() && label[0]->batch() != current_batch),
                std::logic_error)
    << "Error: mismatch in batchsize for data and model."
    << " input_batch: " << input[0]->batch()
    << " label_batch: " << label[0]->batch()
    << " target_batch: " << current_batch;

  model_graph.setInputsLabels(input, label);

  return incremental_forwarding(from, to, training);
}

/**
 * @brief     back propagation
 *            Call backwarding function of layer in reverse order
 *            No need to call at first Input Layer (No data to be updated)
 */
void NeuralNetwork::backwarding(int iteration,
                                std::function<bool(void *userdata)> stop_cb,
                                void *userdata) {

#ifdef DEBUG
  NNTR_THROW_IF(!opt, std::invalid_argument) << "optimizer is null!";
#endif

  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op =
    [this, stop_cb](std::shared_ptr<LayerNode> node, bool training) -> void {
    (void)this;
    PROFILE_MEM_ANNOTATE("Forwarding for layer: " + node->getName());

    auto f = std::get<0>(node->getExecutionOrder());
    model_graph.flushCacheExcept(f);

    node->forwarding(training);
  };

  std::function<bool(std::shared_ptr<LayerNode>, int)> backwarding_op =
    [this, stop_cb, userdata](std::shared_ptr<LayerNode> node,
                              int iteration) -> bool {
    /**
     * Do not change this order:
     * 1. calcGradient
     * 2. calcDerivative
     * 3. applyGradient
     * 4. gradientClippingOnLastAccess
     */

    model_graph.flushCacheExcept(std::get<1>(node->getExecutionOrder()));
    PROFILE_MEM_ANNOTATE("CalcGradient: " + node->getName());

    bool apply_gradient = true;
    if (node->getTrainable()) {
      /** If gradient optimization mode, then calculate gradient first */
      if (dynamic_training_opt.isGradientMode())
        node->calcGradient();

      /**
       * If optimization off, or gradient must be applied, then this will be
       * true
       * @todo This apply gradient should be passed to the each weight and later
       * be queried when updating gradient at once. (after moving apply_gradient
       * out of this function)
       *
       */
      // auto &layer = node->getObject();
      // apply_gradient = dynamic_training_opt.checkIfApply(
      //   layer->getWeightsRef(), layer->net_input[0], layer->net_hidden[0],
      //   opt, iteration);

      /** If gradient must be applied and its not gradient mode, calculate
       * gradient
       */
      if (!dynamic_training_opt.isGradientMode() && apply_gradient) {
        node->calcGradient();

        RunLayerContext &rc = node->getRunContext();
        if (model_graph.isMixedPrecision()) {
          for (auto w : rc.getWeights()) {
            if (w->hasGradient())
              if (!w->getGradientRef().isValid())
                return false;
          }
        }
      }
    }

    model_graph.flushCacheExcept(std::get<2>(node->getExecutionOrder()));
    PROFILE_MEM_ANNOTATE("CalcDerivative: " + node->getName());

    if (stop_cb(userdata)) {
      return true;
    }

    if (node->needsCalcDerivative()) {
      node->calcDerivative();
    }

    model_graph.flushCacheExcept(std::get<3>(node->getExecutionOrder()));
    PROFILE_MEM_ANNOTATE("ApplyGradient: " + node->getName());

    if (apply_gradient) {
      /// Apply gradient only at the end of the last shared weight access
      model_graph.applyGradients(
        node.get(), [iteration, opt_ = opt.get()](Weight &w) {
          w.calcRegularizationGradient();
          if (opt_->getType() != AdamW::type) {
            w.calcWeightDecayGradient();
          }
          RunOptimizerContext opt_context(&w, iteration,
                                          opt_->getLearningRate(iteration));
          opt_->applyGradient(opt_context);
        });
    }
    return true;
  };

  std::function<void(Weight &, int)> lazy_apply_grad_op =
    [opt_ = opt.get()](Weight &w, int iteration) -> void {
    w.calcRegularizationGradient();
    w.calcWeightDecayGradient();
    RunOptimizerContext opt_context(&w, iteration,
                                    opt_->getLearningRate(iteration));
    opt_->applyGradient(opt_context);
  };

  // return false if the gradient is not valid
  bool ret = false;

  while (!ret) {
    ret = model_graph.backwarding(iteration, forwarding_op, backwarding_op,
                                  lazy_apply_grad_op, stop_cb, userdata);
  }
}

void NeuralNetwork::save(
  const std::string &file_path, ml::train::ModelFormat format,
  TensorDim::DataType dtype,
  const std::map<std::string, TensorDim::DataType> &layer_dtype_map) {
  NNTR_THROW_IF(!initialized, std::runtime_error)
    << "Cannot save model if not initialized yet, path: " << file_path
    << " format: " << static_cast<unsigned>(format);

  NNTR_THROW_IF(format != ml::train::ModelFormat::MODEL_FORMAT_BIN &&
                  dtype != TensorDim::DataType::NONE,
                std::runtime_error)
    << "Cannot save the model with a specific data type unless the model "
       "format is `MODEL_FORMAT_BIN`.";

  /// @todo this switch case should be delegating the function call only. It's
  /// not delegating for now as required logics are manageable for now.
  switch (format) {
  case ml::train::ModelFormat::MODEL_FORMAT_BIN: {
    auto model_file = checkedOpenStream<std::ofstream>(
      file_path, std::ios::out | std::ios::binary | std::ios::trunc);

    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      const auto &layer_node = *iter;
      auto it = layer_dtype_map.find(layer_node->getName());
      auto target_dtype = (it != layer_dtype_map.end()) ? it->second : dtype;
      layer_node->save(model_file, false, exec_mode, target_dtype);
    }

    if (opt && istrequal(opt->getType(), "adam")) {
      std::string adam = "adam";
      model_file.write(adam.c_str(), 4);
      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           iter++) {
        (*iter)->save(model_file, true);
      }
    }

    if (exec_mode == ml::train::ExecutionMode::TRAIN) {
      model_file.write((char *)&epoch_idx, sizeof(epoch_idx));
      model_file.write((char *)&iter, sizeof(iter));
    }

    model_file.close();
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_INI:
    saveModelIni(file_path);
    break;
  case ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN: {
    auto old_save_path = std::get<props::SavePath>(model_flex_props);
    auto bin_file_name =
      file_path.substr(0, file_path.find_last_of('.')) + ".bin";

    std::get<props::SavePath>(model_flex_props).set(bin_file_name);
    save(file_path, ml::train::ModelFormat::MODEL_FORMAT_INI);
    save(bin_file_name, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    std::get<props::SavePath>(model_flex_props) = old_save_path;
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS: {
    /**
     * Safetensors format (schema_version 2):
     *   [8B]  header_size (little-endian uint64)
     *   [header_size B] JSON header with tensor metadata
     *   [data section]  raw tensor bytes
     *
     * JSON header format:
     *   {
     *     "__metadata__": {
     *       "format": "nntrainer",
     *       "schema_version": "2"
     *     },
     *     "weight_name": {
     *       "dtype": "F32"|"F16"|"I4"|...,
     *       "shape": [d1, d2, ...],
     *       "data_offsets": [start, end],
     *       "quant": {                       // optional, only for quantized weights
     *         "encoding": "axis_scale_offset" | "per_tensor_affine" |
     *                     "q4_0" | "q6_k" | ...,
     *         "axis": 0,                     // per-axis scale/offset axis
     *         "bitwidth": 4,                 // effective bit width
     *         "group_size": 0,               // 0 = pure per-channel, >0 = grouped
     *         "has_zero_point": false        // true for asymmetric (UINT*)
     *       }
     *     },
     *     ...
     *   }
     *
     * Notes on the embedded-scales layout (Int4QTensor / Uint4QTensor):
     *   `data_offsets` covers the full packed region, i.e.
     *     [packed_4bit_values | scales | (zero_points)?]
     *   exactly as Tensor::save writes it. Readers that want raw weight
     *   bytes only should use `shape` + `bitwidth` to compute the weight
     *   sub-region; the remainder is the scale/zp section.
     */
    auto model_file = checkedOpenStream<std::ofstream>(
      file_path, std::ios::out | std::ios::binary | std::ios::trunc);

    // -- 1. Collect weight metadata --
    struct QuantInfo {
      bool present = false;
      std::string encoding;
      int axis = 0;
      int bitwidth = 0;
      int group_size = 0;
      bool has_zero_point = false;
    };

    struct SafetensorEntry {
      std::string name;
      size_t data_size;
      std::string dtype_str;
      std::vector<size_t> shape;
      QuantInfo quant;
    };
    std::vector<SafetensorEntry> entries;
    size_t total_data_size = 0;

    // Helper: derive quant metadata from a weight tensor. Returns
    // {present=false} for non-quantized (FP32/FP16) tensors; otherwise
    // fills encoding/bitwidth/axis/group_size from the tensor's dtype
    // and QScheme, plus scale_size() when available.
    auto deriveQuantInfo = [](Weight &w) -> QuantInfo {
      QuantInfo qi;
      auto dtype = w.getDim().getDataType();
      // getVariable() returns by value; we need a non-const reference
      // to call q_scheme()/scale_size() on the Tensor's itensor_ view.
      auto &var = w.getVariableRef();

      // Map dtype to bitwidth + signed/unsigned
      auto affineBits = [&](int bits, bool unsigned_) {
        qi.present = true;
        qi.bitwidth = bits;
        qi.has_zero_point = unsigned_;
        try {
          auto scheme = var.q_scheme();
          if (scheme == QScheme::PER_CHANNEL_AFFINE) {
            qi.encoding = "axis_scale_offset";
            qi.axis = 0;
            size_t ss = var.scale_size();
            size_t total = w.getDim().getDataLen();
            if (ss > 0 && ss <= total) {
              // group_size = total_elements / scale_count.
              // If it equals one output channel's width (one scale
              // per output row), normalize to 0 == "pure per-channel".
              size_t gs = total / ss;
              size_t row_width = w.getDim().width();
              qi.group_size = (gs == row_width) ? 0 : static_cast<int>(gs);
            }
          } else {
            qi.encoding = "per_tensor_affine";
          }
        } catch (...) {
          qi.encoding = "per_tensor_affine";
        }
      };

      switch (dtype) {
      case TensorDim::DataType::QINT4:
        affineBits(4, false);
        break;
      case TensorDim::DataType::QINT8:
        affineBits(8, false);
        break;
      case TensorDim::DataType::QINT16:
        affineBits(16, false);
        break;
      case TensorDim::DataType::UINT4:
        affineBits(4, true);
        break;
      case TensorDim::DataType::UINT8:
        affineBits(8, true);
        break;
      case TensorDim::DataType::UINT16:
        affineBits(16, true);
        break;
      case TensorDim::DataType::Q4_0:
        qi.present = true;
        qi.encoding = "q4_0";
        qi.bitwidth = 4;
        qi.axis = 0;
        qi.group_size = 32; // GGML Q4_0 block size
        break;
      case TensorDim::DataType::Q4_K:
        qi.present = true;
        qi.encoding = "q4_k";
        qi.bitwidth = 4;
        qi.axis = 0;
        qi.group_size = 256; // GGML Q4_K super-block
        break;
      case TensorDim::DataType::Q6_K:
        qi.present = true;
        qi.encoding = "q6_k";
        qi.bitwidth = 6;
        qi.axis = 0;
        qi.group_size = 256; // GGML super-block
        break;
      case TensorDim::DataType::Q1_0:
        qi.present = true;
        qi.encoding = "q1_0";
        qi.bitwidth = 1;
        qi.axis = 0;
        qi.group_size = 128; // nntr_ggml_impl Q1_0 group size (QK1_0_TENSOR)
        break;
      default:
        // FP32/FP16/UINT32/etc. — not a quantized weight
        break;
      }
      return qi;
    };

    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      auto weights = (*iter)->getRunContext().getWeights();
      for (unsigned int i = 0; i < weights.size(); ++i) {
        if (!(*iter)->getRunContext().isGradientFirstAccess(i))
          continue;
        auto &w = *weights[i];
        size_t sz = w.getVariable().getMemoryBytes();
        auto dtype = w.getDim().getDataType();

        std::string dtype_str;
        switch (dtype) {
        case TensorDim::DataType::FP32:
          dtype_str = "F32";
          break;
        case TensorDim::DataType::FP16:
          dtype_str = "F16";
          break;
        case TensorDim::DataType::QINT4:
          dtype_str = "I4";
          break;
        case TensorDim::DataType::QINT8:
          dtype_str = "I8";
          break;
        case TensorDim::DataType::QINT16:
          dtype_str = "I16";
          break;
        case TensorDim::DataType::UINT4:
          dtype_str = "U4";
          break;
        case TensorDim::DataType::UINT8:
          dtype_str = "U8";
          break;
        case TensorDim::DataType::UINT16:
          dtype_str = "U16";
          break;
        case TensorDim::DataType::UINT32:
          dtype_str = "U32";
          break;
        // Lane B: block-quantized tensors (GGML-style layout). The dtype
        // string labels the block format; the actual block size and
        // scale layout are implicit in the tensor class.
        case TensorDim::DataType::Q4_0:
          dtype_str = "Q4_0";
          break;
        case TensorDim::DataType::Q4_K:
          dtype_str = "Q4_K";
          break;
        case TensorDim::DataType::Q6_K:
          dtype_str = "Q6_K";
          break;
        case TensorDim::DataType::Q1_0:
          dtype_str = "Q1_0";
          break;
        default:
          dtype_str = "F32";
          break;
        }

        auto dim = w.getDim();
        std::vector<size_t> shape;
        const size_t *dims = dim.getDim();
        bool found_nonone = false;
        for (size_t d = 0; d < TensorDim::MAXDIM; ++d) {
          if (dims[d] != 1)
            found_nonone = true;
          if (found_nonone)
            shape.push_back(dims[d]);
        }
        if (shape.empty())
          shape.push_back(1);

        QuantInfo qi = deriveQuantInfo(w);
        entries.push_back({w.getName(), sz, dtype_str, shape, qi});
        total_data_size += sz;
      }
    }

    // -- 2. Build JSON header string --
    std::ostringstream json_ss;
    json_ss << "{\"__metadata__\":{\"format\":\"nntrainer\","
            << "\"schema_version\":\"2\"}";

    size_t data_offset = 0;
    for (auto &e : entries) {
      json_ss << ",\"" << e.name << "\":{\"dtype\":\"" << e.dtype_str
              << "\",\"shape\":[";
      for (size_t si = 0; si < e.shape.size(); ++si) {
        if (si > 0)
          json_ss << ",";
        json_ss << e.shape[si];
      }
      json_ss << "],\"data_offsets\":[" << data_offset << ","
              << (data_offset + e.data_size) << "]";

      if (e.quant.present) {
        json_ss << ",\"quant\":{\"encoding\":\"" << e.quant.encoding
                << "\",\"axis\":" << e.quant.axis
                << ",\"bitwidth\":" << e.quant.bitwidth
                << ",\"group_size\":" << e.quant.group_size
                << ",\"has_zero_point\":"
                << (e.quant.has_zero_point ? "true" : "false") << "}";
      }

      json_ss << "}";
      data_offset += e.data_size;
    }
    json_ss << "}";
    std::string header_json = json_ss.str();

    // Pad header to 8-byte alignment
    size_t header_len = header_json.size();
    size_t header_padded = (header_len + 7) & ~static_cast<size_t>(7);
    header_json.resize(header_padded, ' ');

    // -- 3. Write header_size + header + data --
    uint64_t hdr_size_val = static_cast<uint64_t>(header_padded);
    model_file.write(reinterpret_cast<const char *>(&hdr_size_val),
                     sizeof(hdr_size_val));
    model_file.write(header_json.data(), header_padded);

    // Write data section (weight bytes in graph order)
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      (*iter)->save(model_file, false, exec_mode);
    }

    model_file.close();
    ml_logi("Saved model in safetensors format: %s (%zu weights, %zu bytes)",
            file_path.c_str(), entries.size(), total_data_size);
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_ONNX: {
    throw nntrainer::exception::not_supported(
      "saving with ONNX format is not supported yet.");
    break;
  }
  default:
    throw nntrainer::exception::not_supported(
      "saving with given format is not supported yet");
  }
}

void NeuralNetwork::convertBinToSafetensors(const std::string &bin_path,
                                            const std::string &st_path) {
  NNTR_THROW_IF(!initialized, std::runtime_error)
    << "Model must be initialized before converting weight format";

  load(bin_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  save(st_path, ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);

  ml_logi("Converted weight file from BIN to safetensors: %s -> %s",
          bin_path.c_str(), st_path.c_str());
}

void NeuralNetwork::load(const std::string &file_path,
                         ml::train::ModelFormat format) {
  /// @todo this switch case should be delegating the function call only. It's
  /// not delegating for now as required logics are manageable for now.

  bool fsu_mode = std::get<props::Fsu>(model_flex_props);

  const std::regex reg_("\\s*\\;\\s*");
  auto v = split(file_path, reg_);

  auto f_path = (v.size() == 2) ? v[1] : v[0];

  /**
   * For safetensors format, parse JSON header to get name-based offsets.
   * For BIN format, use sequential offset assignment (legacy).
   */
  size_t data_section_start = 0;

  /**
   * @brief Parsed per-tensor metadata from the safetensors JSON header.
   *
   * Schema version 1 (legacy) populates only {offset,size,shape?}; schema
   * version 2 additionally fills {dtype, quant_*} when the writer emits
   * them. quant_present indicates whether the "quant" object was found on
   * this entry; absent + a quantized model-side dtype is accepted with a
   * warning (legacy files) while a mismatching "quant" object triggers a
   * strict error.
   */
  struct SafetensorEntry {
    size_t offset = 0;
    size_t size = 0;
    std::string dtype; ///< empty if not present in the header
    bool quant_present = false;
    std::string quant_encoding;
    int quant_axis = 0;
    int quant_bitwidth = 0;
    int quant_group_size = 0;
    bool quant_has_zero_point = false;
  };

  std::unordered_map<std::string, SafetensorEntry> name_offset_map;
  std::unordered_map<std::string, std::pair<size_t, size_t>> prefix_offset_map;
  std::unordered_map<std::string, int> prefix_count;
  std::string safetensors_schema_version = "1";
  std::string safetensors_format;
  bool is_safetensors = (format == ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);

  if (is_safetensors) {
    auto probe = checkedOpenStream<std::ifstream>(
      f_path, std::ios::in | std::ios::binary);
    uint64_t header_size = 0;
    probe.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
    NNTR_THROW_IF(!probe.good(), std::runtime_error)
      << "Failed to read safetensors header size from: " << f_path;

    data_section_start = sizeof(uint64_t) + static_cast<size_t>(header_size);

    std::string header_json(static_cast<size_t>(header_size), '\0');
    probe.read(&header_json[0], header_size);
    NNTR_THROW_IF(!probe.good(), std::runtime_error)
      << "Failed to read safetensors JSON header from: " << f_path;
    probe.close();

    // Minimal JSON parser for safetensors header. Extracts:
    //   - __metadata__.schema_version, __metadata__.format
    //   - per-entry: data_offsets, dtype, quant object
    auto parse_safetensors_header =
      [](const std::string &json,
         std::unordered_map<std::string, SafetensorEntry> &out_map,
         std::string &out_schema_version, std::string &out_format) {
        size_t pos = 0;
        auto skip_ws = [&]() {
          while (pos < json.size() &&
                 (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' ||
                  json[pos] == '\t'))
            ++pos;
        };
        auto parse_string = [&]() -> std::string {
          skip_ws();
          if (pos >= json.size() || json[pos] != '"')
            return "";
          ++pos;
          std::string result;
          while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size())
              ++pos;
            result += json[pos++];
          }
          if (pos < json.size())
            ++pos;
          return result;
        };
        auto parse_number = [&]() -> size_t {
          skip_ws();
          size_t val = 0;
          while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
            val = val * 10 + (json[pos] - '0');
            ++pos;
          }
          return val;
        };
        auto parse_bool = [&]() -> bool {
          skip_ws();
          if (pos + 4 <= json.size() && json.compare(pos, 4, "true") == 0) {
            pos += 4;
            return true;
          }
          if (pos + 5 <= json.size() && json.compare(pos, 5, "false") == 0) {
            pos += 5;
            return false;
          }
          return false;
        };
        auto skip_value = [&]() {
          skip_ws();
          if (pos >= json.size())
            return;
          if (json[pos] == '"') {
            parse_string();
          } else if (json[pos] == '{') {
            int depth = 1;
            ++pos;
            while (pos < json.size() && depth > 0) {
              if (json[pos] == '{')
                ++depth;
              else if (json[pos] == '}')
                --depth;
              ++pos;
            }
          } else if (json[pos] == '[') {
            int depth = 1;
            ++pos;
            while (pos < json.size() && depth > 0) {
              if (json[pos] == '[')
                ++depth;
              else if (json[pos] == ']')
                --depth;
              ++pos;
            }
          } else {
            while (pos < json.size() && json[pos] != ',' && json[pos] != '}' &&
                   json[pos] != ']')
              ++pos;
          }
        };

        // Parse the __metadata__ nested object and extract recognized
        // fields. Unknown fields are skipped for forward compatibility.
        auto parse_metadata_object = [&]() {
          skip_ws();
          if (pos >= json.size() || json[pos] != '{') {
            skip_value();
            return;
          }
          ++pos;
          while (pos < json.size() && json[pos] != '}') {
            skip_ws();
            if (pos < json.size() && json[pos] == ',')
              ++pos;
            std::string mk = parse_string();
            skip_ws();
            if (pos < json.size() && json[pos] == ':')
              ++pos;
            skip_ws();
            if (mk == "schema_version") {
              out_schema_version = parse_string();
            } else if (mk == "format") {
              out_format = parse_string();
            } else {
              skip_value();
            }
          }
          if (pos < json.size() && json[pos] == '}')
            ++pos;
        };

        // Parse the "quant" nested object inside a per-tensor entry.
        auto parse_quant_object = [&](SafetensorEntry &entry) {
          skip_ws();
          if (pos >= json.size() || json[pos] != '{') {
            skip_value();
            return;
          }
          ++pos;
          entry.quant_present = true;
          while (pos < json.size() && json[pos] != '}') {
            skip_ws();
            if (pos < json.size() && json[pos] == ',')
              ++pos;
            std::string qf = parse_string();
            skip_ws();
            if (pos < json.size() && json[pos] == ':')
              ++pos;
            skip_ws();
            if (qf == "encoding") {
              entry.quant_encoding = parse_string();
            } else if (qf == "axis") {
              entry.quant_axis = static_cast<int>(parse_number());
            } else if (qf == "bitwidth") {
              entry.quant_bitwidth = static_cast<int>(parse_number());
            } else if (qf == "group_size") {
              entry.quant_group_size = static_cast<int>(parse_number());
            } else if (qf == "has_zero_point") {
              entry.quant_has_zero_point = parse_bool();
            } else {
              skip_value();
            }
          }
          if (pos < json.size() && json[pos] == '}')
            ++pos;
        };

        skip_ws();
        if (pos >= json.size() || json[pos] != '{')
          return;
        ++pos;

        while (pos < json.size()) {
          skip_ws();
          if (json[pos] == '}')
            break;
          if (json[pos] == ',')
            ++pos;

          std::string key = parse_string();
          skip_ws();
          if (pos < json.size() && json[pos] == ':')
            ++pos;
          skip_ws();

          if (key == "__metadata__") {
            parse_metadata_object();
            continue;
          }

          if (pos < json.size() && json[pos] == '{') {
            ++pos;
            SafetensorEntry entry;
            size_t offset_start = 0, offset_end = 0;
            bool found_offsets = false;
            while (pos < json.size() && json[pos] != '}') {
              skip_ws();
              if (json[pos] == ',')
                ++pos;
              std::string field = parse_string();
              skip_ws();
              if (pos < json.size() && json[pos] == ':')
                ++pos;

              if (field == "data_offsets") {
                skip_ws();
                if (pos < json.size() && json[pos] == '[') {
                  ++pos;
                  offset_start = parse_number();
                  skip_ws();
                  if (pos < json.size() && json[pos] == ',')
                    ++pos;
                  offset_end = parse_number();
                  skip_ws();
                  if (pos < json.size() && json[pos] == ']')
                    ++pos;
                  found_offsets = true;
                }
              } else if (field == "dtype") {
                entry.dtype = parse_string();
              } else if (field == "quant") {
                parse_quant_object(entry);
              } else {
                skip_value();
              }
            }
            if (pos < json.size() && json[pos] == '}')
              ++pos;

            if (found_offsets) {
              entry.offset = offset_start;
              entry.size = offset_end - offset_start;
              out_map[key] = entry;
            }
          } else {
            skip_value();
          }
        }
      };

    parse_safetensors_header(header_json, name_offset_map,
                             safetensors_schema_version, safetensors_format);

    // Version management: reject unknown schemas. schema_version "1"
    // is the legacy untagged format (files that predate P2 and have
    // no schema_version field at all); "2" is the current format
    // with optional per-entry "quant" metadata.
    if (safetensors_schema_version != "1" && safetensors_schema_version != "2") {
      std::ostringstream oss;
      oss << "[safetensors] Unsupported schema_version '"
          << safetensors_schema_version
          << "' in file " << f_path
          << ". This nntrainer build understands schema_version 1 and 2. "
          << "Please regenerate the file with a compatible converter or "
          << "update nntrainer.";
      throw std::runtime_error(oss.str());
    }

    std::cout << "[safetensors] Loaded " << name_offset_map.size()
              << " tensor entries (schema_version="
              << safetensors_schema_version
              << (safetensors_format.empty()
                    ? ""
                    : ", format=" + safetensors_format)
              << ")" << std::endl;

    // Build prefix-based fallback map: "layer_name" -> (offset, size)
    // This allows matching when the weight suffix differs between
    // the safetensors file and the C++ model (e.g., ":weight" vs ":gamma").
    //
    // Safety: only add to prefix_offset_map when the prefix is UNAMBIGUOUS
    // (exactly one safetensors entry shares that prefix). When multiple
    // entries share a prefix (e.g. both :weight and :bias), prefix fallback
    // is refused — otherwise unordered_map iteration order would make the
    // result non-deterministic and the wrong tensor could be loaded
    // silently.
    for (auto &kv : name_offset_map) {
      auto colon_pos = kv.first.find(':');
      if (colon_pos != std::string::npos) {
        std::string prefix = kv.first.substr(0, colon_pos);
        prefix_count[prefix]++;
      }
    }
    for (auto &kv : name_offset_map) {
      auto colon_pos = kv.first.find(':');
      if (colon_pos != std::string::npos) {
        std::string prefix = kv.first.substr(0, colon_pos);
        if (prefix_count[prefix] == 1) {
          prefix_offset_map[prefix] = {kv.second.offset, kv.second.size};
        }
      }
    }

    // Diagnostic dump: list all safetensors entries in sorted order so the
    // user can visually diff against the C++ model weight list (printed
    // below, right before the matching loop).
    {
      std::vector<std::string> sorted_names;
      sorted_names.reserve(name_offset_map.size());
      for (auto &kv : name_offset_map)
        sorted_names.push_back(kv.first);
      std::sort(sorted_names.begin(), sorted_names.end());
      std::cout << "[safetensors] File tensor list (sorted):" << std::endl;
      for (auto &n : sorted_names) {
        auto &e = name_offset_map[n];
        std::cout << "  " << n << "  (offset=" << e.offset
                  << ", size=" << e.size;
        if (!e.dtype.empty())
          std::cout << ", dtype=" << e.dtype;
        if (e.quant_present) {
          std::cout << ", quant={enc=" << e.quant_encoding
                    << ",axis=" << e.quant_axis
                    << ",bw=" << e.quant_bitwidth
                    << ",gs=" << e.quant_group_size
                    << (e.quant_has_zero_point ? ",zp" : "") << "}";
        }
        std::cout << ")" << std::endl;
      }
    }
  }

  // Diagnostic dump: list all C++ model weight names in graph order so the
  // user can visually diff against the safetensors file list (printed
  // above). This is the authoritative "what C++ expects" reference.
  if (is_safetensors) {
    std::cout << "[nntrainer] C++ model weight list (graph order):"
              << std::endl;
    std::unordered_set<const Tensor *> dumped;
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      auto weights = (*iter)->getRunContext().getWeights();
      for (auto weight : weights) {
        if (!dumped.insert(&weight->getVariableRef()).second)
          continue;
        std::cout << "  " << weight->getName()
                  << "  (bytes=" << weight->getVariable().getMemoryBytes()
                  << ", dtype="
                  << static_cast<int>(weight->getDim().getDataType()) << ")"
                  << std::endl;
      }
    }
  }

  /**
   * Assign file offsets to each weight tensor.
   *   Safetensors: look up by weight name in the JSON header (order-independent).
   *   BIN: sequential accumulation in topological order (legacy).
   */
  size_t start_from = 0;
  std::vector<std::pair<size_t, size_t>> file_offset;
  std::unordered_set<const Tensor *> visited_weights;
  unsigned int weight_match_count = 0;
  unsigned int weight_miss_count = 0;
  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
    auto weights = (*iter)->getRunContext().getWeights();
    for (auto weight : weights) {
      // Shared weights (e.g., TieWordEmbedding) reference the same Tensor
      // object via requestOrExtend. Calling setFileOffset on the second
      // occurrence overwrites the correct offset by the first.
      // Skip duplicates so that:
      // 1. file_offset is only set once (at the position where save writes)
      // 2. start_from is only advanced once (matching actual file layout)
      if (!visited_weights.insert(&weight->getVariableRef()).second) {
        continue;
      }
      size_t size = weight->getVariable().getMemoryBytes();
      auto tensor_data_type = weight->getDim().getDataType();

      if (is_safetensors) {
        const std::string &wname = weight->getName();
        auto it = name_offset_map.find(wname);
        if (it != name_offset_map.end()) {
          // Exact match: safetensors name == C++ weight name
          const SafetensorEntry &entry = it->second;
          weight->getVariableRef().setFileOffset(data_section_start +
                                                 entry.offset);
          std::cout << "  [MATCH]  '" << wname << "' -> offset="
                    << (data_section_start + entry.offset) << std::endl;
          weight_match_count++;

          // ---- Validation (schema_version 2+) ----
          // Map C++ model dtype to the short string the writer emits.
          auto dtype_to_str =
            [](TensorDim::DataType t) -> const char * {
            switch (t) {
            case TensorDim::DataType::FP32:
              return "F32";
            case TensorDim::DataType::FP16:
              return "F16";
            case TensorDim::DataType::QINT4:
              return "I4";
            case TensorDim::DataType::QINT8:
              return "I8";
            case TensorDim::DataType::QINT16:
              return "I16";
            case TensorDim::DataType::UINT4:
              return "U4";
            case TensorDim::DataType::UINT8:
              return "U8";
            case TensorDim::DataType::UINT16:
              return "U16";
            case TensorDim::DataType::UINT32:
              return "U32";
            case TensorDim::DataType::Q4_0:
              return "Q4_0";
            case TensorDim::DataType::Q4_K:
              return "Q4_K";
            case TensorDim::DataType::Q6_K:
              return "Q6_K";
            case TensorDim::DataType::Q1_0:
              return "Q1_0";
            default:
              return "";
            }
          };
          const char *c_dtype = dtype_to_str(tensor_data_type);

          // dtype mismatch is a hard error if the file declared a dtype.
          // Legacy schema_version 1 files have no dtype so we skip.
          if (!entry.dtype.empty() && *c_dtype != '\0' &&
              entry.dtype != std::string(c_dtype)) {
            std::ostringstream oss;
            oss << "[safetensors] dtype mismatch for weight '" << wname
                << "': file has '" << entry.dtype
                << "' but C++ model expects '" << c_dtype
                << "'. The safetensors file was likely generated for a "
                   "different quantization config.";
            throw std::runtime_error(oss.str());
          }

          // Quant metadata validation. We only validate in one direction
          // for now: if BOTH sides declare quant info, they must agree.
          // If the file has no quant info (v1 or omitted), we accept it
          // and trust the C++ model's dtype — this preserves backward
          // compatibility with existing files.
          bool c_is_quant = (tensor_data_type == TensorDim::DataType::QINT4 ||
                             tensor_data_type == TensorDim::DataType::QINT8 ||
                             tensor_data_type == TensorDim::DataType::QINT16 ||
                             tensor_data_type == TensorDim::DataType::UINT4 ||
                             tensor_data_type == TensorDim::DataType::UINT8 ||
                             tensor_data_type == TensorDim::DataType::UINT16 ||
                             tensor_data_type == TensorDim::DataType::Q4_0 ||
                             tensor_data_type == TensorDim::DataType::Q4_K ||
                             tensor_data_type == TensorDim::DataType::Q6_K ||
                             tensor_data_type == TensorDim::DataType::Q1_0);
          if (entry.quant_present && !c_is_quant) {
            std::ostringstream oss;
            oss << "[safetensors] weight '" << wname
                << "' has a quant object in the file (encoding="
                << entry.quant_encoding
                << ") but the C++ model's dtype (" << c_dtype
                << ") is not a quantized type.";
            throw std::runtime_error(oss.str());
          }
          if (c_is_quant && !entry.quant_present &&
              safetensors_schema_version == "2") {
            // Schema v2 should include quant info for quantized
            // weights; absence is suspicious but not fatal.
            std::cout << "  [WARN]   '" << wname
                      << "' C++ expects a quantized dtype but the "
                         "schema_version 2 file has no quant object"
                      << std::endl;
          }
        } else {
          // Prefix fallback: match by layer name prefix (before ':')
          // This handles suffix mismatches like ":weight" vs ":gamma"
          auto colon_pos = wname.find(':');
          std::string prefix =
            (colon_pos != std::string::npos) ? wname.substr(0, colon_pos) : wname;
          auto pit = prefix_offset_map.find(prefix);
          if (pit != prefix_offset_map.end()) {
            weight->getVariableRef().setFileOffset(data_section_start +
                                                   pit->second.first);
            std::cout << "  [PREFIX] '" << wname << "' matched by prefix '"
                      << prefix << "' -> offset="
                      << (data_section_start + pit->second.first) << std::endl;
            weight_match_count++;
          } else {
            // Strict: a MISS used to set file_offset=0, which silently read
            // garbage from the start of the file (the safetensors header).
            // Fail loudly with a diagnostic message so name-mismatch bugs
            // surface immediately instead of producing wrong outputs.
            weight_miss_count++;
            std::cout << "  [MISS]   '" << wname
                      << "' NOT FOUND in safetensors" << std::endl;
            std::ostringstream oss;
            oss << "[safetensors] weight '" << wname
                << "' NOT FOUND: no exact match, ";
            auto pc_it = prefix_count.find(prefix);
            if (pc_it != prefix_count.end() && pc_it->second > 1) {
              oss << "prefix '" << prefix << "' is ambiguous ("
                  << pc_it->second
                  << " safetensors entries share it; prefix fallback refused)";
            } else if (pc_it == prefix_count.end()) {
              oss << "prefix '" << prefix
                  << "' not present in safetensors either";
            } else {
              oss << "prefix '" << prefix << "' present but lookup failed";
            }
            oss << ". See the '[safetensors] File tensor list' and "
                   "'[nntrainer] C++ model weight list' above to diff names.";
            throw std::runtime_error(oss.str());
          }
        }
      } else {
        weight->getVariableRef().setFileOffset(start_from);
      }

      if (tensor_data_type != TensorDim::DataType::FP32 &&
          tensor_data_type != TensorDim::DataType::FP16 &&
          tensor_data_type != TensorDim::DataType::Q6_K &&
          tensor_data_type != TensorDim::DataType::Q4_0) {
        size += sizeof(uint16_t);
      }
      file_offset.emplace_back(
        std::make_pair(weight->getVariable().getFileOffset(), size));
      start_from += size;
    }
  }

  if (is_safetensors) {
    std::cout << "=== Safetensors load summary: " << weight_match_count
              << " matched, " << weight_miss_count << " missed ===" << std::endl;
  }

  if (exec_mode == ExecutionMode::INFERENCE && fsu_mode) {
    model_graph.setFsuWeightPath((v.size() == 2) ? v[1] : v[0]);
    model_graph.setWeightOffset(file_offset);
  }

  switch (format) {
  case ml::train::ModelFormat::MODEL_FORMAT_BIN: {
    NNTR_THROW_IF(!initialized, std::runtime_error)
      << "Cannot load if not initialized yet, path: " << file_path
      << " format: " << static_cast<unsigned>(format);

    auto model_file =
      checkedOpenStream<std::ifstream>(f_path, std::ios::in | std::ios::binary);

#if defined(_WIN32)
    HANDLE hFile, hMap;
#endif

    if (exec_mode == ml::train::ExecutionMode::INFERENCE) {
      if (!MMAP_READ) {
        ///@note for slim-tensor. This should be removed.
        model_file_fd = open(f_path.c_str(), O_RDONLY);
        NNTR_THROW_IF((model_file_fd == -1), std::invalid_argument)
          << "Cannot open file : " << f_path;
      }
      // std::vector<std::future<void>> futures;
      std::vector<std::thread> threads;
      threads.reserve(model_graph.size());
      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           ++iter) {
        auto node = *iter;
        auto exec_order = std::get<0>((*iter)->getExecutionOrder());

        threads.emplace_back([&, node]() {
          if (!MMAP_READ) {
            auto local_model_file = checkedOpenStream<std::ifstream>(
              (v.size() == 2) ? v[1] : v[0], std::ios::in | std::ios::binary);
            node->read(local_model_file, false, exec_mode, fsu_mode,
                       std::numeric_limits<size_t>::max(), true, model_file_fd);
          } else {
#if defined(_WIN32)
            // Map per-ask, then unmap immediately after: enables early release
            // of pages
            HANDLE hFile =
              CreateFileA(f_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            NNTR_THROW_IF((hFile == INVALID_HANDLE_VALUE), std::runtime_error)
              << "CreateFileA failed";

            HANDLE hMap =
              CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
            NNTR_THROW_IF((hMap == NULL), std::runtime_error)
              << "CreateFileMapping failed";

            char *view =
              static_cast<char *>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
            NNTR_THROW_IF((view == nullptr), std::runtime_error)
              << "MapViewOfFile failed";

            node->read(view, false, exec_mode, fsu_mode,
                       std::numeric_limits<size_t>::max(), true);

            // Early unmap: let the OS reclaim the working set ASAP
            UnmapViewOfFile(view);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            // POSIX: map per-task, advise kernel, drop pages, unmap
            int fd = ::open(f_path.c_str(), O_RDONLY);
            NNTR_THROW_IF((fd == -1), std::invalid_argument)
              << "Cannot open file : " << f_path;

            struct stat st {};
            NNTR_THROW_IF((::fstat(fd, &st) == -1), std::invalid_argument)
              << "Cannot get file info (fstat): " << f_path;

            size_t f_size = static_cast<size_t>(st.st_size);
            void *mmap_ptr =
              ::mmap(nullptr, f_size, PROT_READ, MAP_PRIVATE, fd, 0);
            ::close(fd); // fd not needed after mmap
            NNTR_THROW_IF((mmap_ptr == MAP_FAILED), std::runtime_error)
              << "mmap failed";

            // Hint: many model loads touch scattered regions -> RANDOM helps
            // reduce readahead
            (void)::posix_madvise(mmap_ptr, f_size, POSIX_MADV_RANDOM);

            char *view = static_cast<char *>(mmap_ptr);
            node->read(view, false, exec_mode, fsu_mode,
                       std::numeric_limits<size_t>::max(), true);

            // Early drop: pages no longer needed; helps lower peak RSS during
            // overlap
            (void)::posix_madvise(mmap_ptr, f_size, POSIX_MADV_DONTNEED);

            ::munmap(mmap_ptr, f_size);
#endif
          }
        });
      }
      for (auto &t : threads) {
        if (t.joinable())
          t.join();
      }
    } else {
      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           ++iter) {
        (*iter)->read(model_file, false, exec_mode, fsu_mode);
      }

      try {
        /// this is assuming that the failure is allowed at the end of the file
        /// read. so, after this line, additional read shouldn't be called
        if (opt && istrequal(opt->getType(), "adam")) {
          std::string opt_type;
          opt_type.resize(4);
          model_file.read((char *)&opt_type[0], 4);

          if (istrequal(opt_type, "adam")) {
            for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
                 iter++) {
              (*iter)->read(model_file, true, exec_mode);
            }
          }
        }

        if (!fsu_mode && exec_mode == ml::train::ExecutionMode::TRAIN) {

          checkedRead(model_file, (char *)&epoch_idx, sizeof(epoch_idx),
                      "[NeuralNetwork::readModel] failed to read epoch_idx");
          checkedRead(model_file, (char *)&iter, sizeof(iter),
                      "[NeuralNetwork::readModel] failed to read iteration");
        }
      } catch (...) {
        std::cerr << "failed to read additional data like optimizer variable, "
                     "iteration, proceeding with default\n";
      }
    }

    ml_logi("read modelfile: %s",
            (v.size() == 2) ? v[1].c_str() : v[0].c_str());
    break;
  }

  case ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN: {
    int ret = loadFromConfig((v.size() == 2) ? v[1] : v[0]);
    throw_status(ret);
    auto &save_path = std::get<props::SavePath>(model_flex_props);
    if (!save_path.empty()) {
      checkedOpenStream<std::ifstream>(save_path,
                                       std::ios::in | std::ios::binary);
      load_path = save_path;
    }
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_INI: {
    int ret = loadFromConfig((v.size() == 2) ? v[1] : v[0]);
    throw_status(ret);
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_FLATBUFFER: {
    break;
  }

  case ml::train::ModelFormat::MODEL_FORMAT_ONNX: {
    int ret = loadFromConfig((v.size() == 2) ? v[1] : v[0]);
    throw_status(ret);
    break;
  }

  case ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS: {
    NNTR_THROW_IF(!initialized, std::runtime_error)
      << "Cannot load if not initialized yet, path: " << file_path
      << " format: safetensors";

    // Safetensors: parallel mmap loading using name-based offsets
    if (exec_mode == ml::train::ExecutionMode::INFERENCE) {
      std::vector<std::thread> threads;
      threads.reserve(model_graph.size());
      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           ++iter) {
        auto node = *iter;
        threads.emplace_back([&, node]() {
#if defined(_WIN32)
          HANDLE hFile =
            CreateFileA(f_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
          NNTR_THROW_IF((hFile == INVALID_HANDLE_VALUE), std::runtime_error)
            << "CreateFileA failed";
          HANDLE hMap =
            CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
          NNTR_THROW_IF((hMap == NULL), std::runtime_error)
            << "CreateFileMapping failed";
          char *view =
            static_cast<char *>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
          NNTR_THROW_IF((view == nullptr), std::runtime_error)
            << "MapViewOfFile failed";
          node->read(view, false, exec_mode, fsu_mode,
                     std::numeric_limits<size_t>::max(), true);
          UnmapViewOfFile(view);
          CloseHandle(hMap);
          CloseHandle(hFile);
#else
          int fd = ::open(f_path.c_str(), O_RDONLY);
          NNTR_THROW_IF((fd == -1), std::invalid_argument)
            << "Cannot open file : " << f_path;
          struct stat st {};
          NNTR_THROW_IF((::fstat(fd, &st) == -1), std::invalid_argument)
            << "Cannot get file info (fstat): " << f_path;
          size_t f_size = static_cast<size_t>(st.st_size);
          void *mmap_ptr =
            ::mmap(nullptr, f_size, PROT_READ, MAP_PRIVATE, fd, 0);
          ::close(fd);
          NNTR_THROW_IF((mmap_ptr == MAP_FAILED), std::runtime_error)
            << "mmap failed";
          (void)::posix_madvise(mmap_ptr, f_size, POSIX_MADV_RANDOM);
          char *view = static_cast<char *>(mmap_ptr);
          node->read(view, false, exec_mode, fsu_mode,
                     std::numeric_limits<size_t>::max(), true);
          (void)::posix_madvise(mmap_ptr, f_size, POSIX_MADV_DONTNEED);
          ::munmap(mmap_ptr, f_size);
#endif
        });
      }
      for (auto &t : threads) {
        if (t.joinable())
          t.join();
      }
    } else {
      // Training mode: sequential loading (no optimizer state in safetensors)
      // Use read_from_offset=true so each weight seeks to its file_offset
      auto model_file = checkedOpenStream<std::ifstream>(
        f_path, std::ios::in | std::ios::binary);
      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           ++iter) {
        (*iter)->read(model_file, false, exec_mode, fsu_mode,
                      std::numeric_limits<size_t>::max(), true);
      }
    }

    // Print first 4 values of each weight for verification
    std::cout << "\n=== Loaded weight values (first 4) ===" << std::endl;
    std::unordered_set<const Tensor *> printed_weights;
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      auto weights = (*iter)->getRunContext().getWeights();
      for (auto weight : weights) {
        if (!printed_weights.insert(&weight->getVariableRef()).second)
          continue;
        const float *data = weight->getVariable().getData<float>();
        if (!data)
          continue;
        size_t total = weight->getVariable().size();
        size_t show = std::min<size_t>(4, total);
        std::cout << "  " << weight->getName() << " first4=[";
        for (size_t j = 0; j < show; ++j) {
          if (j)
            std::cout << ", ";
          printf("%.8f", data[j]);
        }
        std::cout << "]" << std::endl;
      }
    }
    std::cout << "=======================================" << std::endl;

    ml_logi("read safetensors modelfile: %s", f_path.c_str());
    break;
  }

  case ml::train::ModelFormat::MODEL_FORMAT_QNN: {
    // for now, we only support to QNN binary format for Inference mode.
    // expect to have the file path for qnn bin and nntrainer bin seperated by
    // ":" QNN bin ( graph ) : NNTrainer bin (weight)
    NNTR_THROW_IF(exec_mode != ExecutionMode::INFERENCE, std::invalid_argument)
      << "Only support QNN biarny for Infernece";
    NNTR_THROW_IF(!isFileExist(props::FilePath(v[0])), std::invalid_argument)
      << "Cannot open QNN context bin file";

    std::thread qnn_load([this, &v]() {
      int ret =
        ct_engine->getRegisteredContext("qnn")->load(props::FilePath(v[0]));
      throw_status(ret);
    });

    if (!fsu_mode && v.size() > 1) {
      NNTR_THROW_IF(!isFileExist(props::FilePath(v[1])), std::invalid_argument)
        << "Cannot open weight bin file";
      load(props::FilePath(v[1]), ml::train::ModelFormat::MODEL_FORMAT_BIN);
    } else if (fsu_mode) {
      NNTR_THROW_IF(v.size() <= 1, std::invalid_argument)
        << "Swap mode should run with loading a weight-bin file";
      NNTR_THROW_IF(!isFileExist(props::FilePath(v[1])), std::invalid_argument)
        << "Cannot open weight bin file";
      // model_graph.setFsuWeightPath(v[1]);
    }

    qnn_load.join();
    break;
  }
  default:
    throw nntrainer::exception::not_supported(
      "loading with given format is not supported yet");
  }
}

float NeuralNetwork::getLoss() {
  loss = 0.0f;

  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
    loss += (*iter)->getLoss();
  }
  return loss;
}

void NeuralNetwork::setLoss(float l) { loss = l; }

NeuralNetwork &NeuralNetwork::copy(NeuralNetwork &from) {
  if (this != &from) {
    model_props = from.model_props;
    model_flex_props = from.model_flex_props;
    loss = from.loss;
    opt = from.opt;

    model_graph.copy(from.model_graph);
  }
  return *this;
}

void NeuralNetwork::saveModelIni(const std::string &file_path) {
  NNTR_THROW_IF(isFileExist(file_path), std::invalid_argument)
    << "There is already a file, overriding to the existing file is not "
       "permitted, path: "
    << file_path;

  std::vector<IniSection> sections;

  IniSection model_section = IniSection::FromExportable("model", *this);
  model_section.setEntry("type", "NeuralNetwork");
  sections.push_back(model_section);

  auto add_section_if_any = [&sections](const std::string &section_name,
                                        auto obj_ptr, auto pred) {
    if (pred(obj_ptr)) {
      IniSection s = IniSection::FromExportable(section_name, *obj_ptr);
      s.setEntry("type", obj_ptr->getType());
      sections.push_back(s);
    }
  };

  add_section_if_any("optimizer", opt,
                     [](const auto &obj) { return static_cast<bool>(obj); });

  auto &[train_buffer, valid_buffer, test_buffer] = data_buffers;
  auto data_buffer_valid = [](const auto &buffer) {
    return buffer && buffer->isSerializable(
                       ml::train::ExportMethods::METHOD_STRINGVECTOR);
  };

  add_section_if_any("train_set", train_buffer, data_buffer_valid);
  add_section_if_any("valid_set", valid_buffer, data_buffer_valid);
  add_section_if_any("test_set", test_buffer, data_buffer_valid);

  IniWrapper wrapper("model_saver", sections);
  wrapper.save_ini(file_path);

  IniGraphInterpreter interpreter;
  interpreter.serialize(graph_representation, file_path);
}

bool NeuralNetwork::validateInput(sharedConstTensors X) {
  auto input_dim = getInputDimension();
  if (X.size() != input_dim.size()) {
    ml_loge("Error: provided number of inputs %d, required %d", (int)X.size(),
            (int)input_dim.size());
    return false;
  }

  for (unsigned int dim = 0; dim < input_dim.size(); dim++) {
    if (input_dim[dim] != X[dim]->getDim()) {
      ml_loge("Error: provided input shape does not match required shape");
      std::stringstream ss;
      ss << X[dim]->getDim();
      ml_loge("Provided tensor summary : %s", ss.str().c_str());

      ss.str(std::string());
      ss << input_dim[dim];
      ml_loge("Required tensor summary : %s", ss.str().c_str());
      return false;
    }
  }

  return true;
}

sharedConstTensors NeuralNetwork::inference(sharedConstTensors X,
                                            bool free_mem) {
  return inference(X, {}, free_mem);
}

sharedConstTensors NeuralNetwork::inference(sharedConstTensors X,
                                            sharedConstTensors label,
                                            bool free_mem) {
  if (model_graph.getBatchSize() != X[0]->batch()) {
    model_graph.setBatchSize(X[0]->batch());
  }

  sharedConstTensors out;
  if (!validateInput(X))
    throw std::invalid_argument("Input validation failed.");

  allocate(ExecutionMode::INFERENCE);

  int nn_foward;
  PROFILE_TIME_REGISTER_EVENT(nn_foward, "nn_forward");
  PROFILE_TIME_START(nn_foward);
  out = forwarding(X, label, false);
  PROFILE_TIME_END(nn_foward);

  if (free_mem)
    /**
     * Free the memory needed for training before exiting.
     * Note that this does not free the weights for the model.
     * Weights of the model will be freed when the model is destroyed.
     */
    model_graph.deallocateTensors(false);

  /** Clear the set inputs and labels */
  model_graph.setInputsLabels({}, {});

  return out;
}

std::vector<float *>
NeuralNetwork::inference(unsigned int batch_size,
                         const std::vector<float *> &input,
                         const std::vector<float *> &label) {
  sharedConstTensors input_tensors, output_tensors;
  auto in_dim = getInputDimension();

  input_tensors.reserve(input.size());
  for (unsigned int idx = 0; idx < in_dim.size(); idx++) {
    in_dim[idx].batch(batch_size);
    input_tensors.emplace_back(MAKE_SHARED_TENSOR(Tensor::Map(
      input[idx], in_dim[idx].getDataLen() * sizeof(float), in_dim[idx], 0)));
  }

  if (!label.empty()) {
    sharedConstTensors label_tensors;
    auto label_dim = getOutputDimension();
    label_tensors.reserve(label.size());
    for (unsigned int idx = 0; idx < label_dim.size(); idx++) {
      label_dim[idx].batch(batch_size);
      label_tensors.emplace_back(MAKE_SHARED_TENSOR(
        Tensor::Map(label[idx], label_dim[idx].getDataLen() * sizeof(float),
                    label_dim[idx], 0)));
    }
    output_tensors = inference(input_tensors, label_tensors, false);
  } else {
    output_tensors = inference(input_tensors, false);
  }

  std::vector<float *> output;
  output.reserve(output_tensors.size());

  for (auto &out : output_tensors) {
    auto out_t = *out.get();
    output.push_back(out_t.getData());
  }

  return output;
}

sharedConstTensors
NeuralNetwork::incremental_inference(sharedConstTensors X,
                                     unsigned int init_seq_len,
                                     unsigned int from, unsigned int to) {
  return incremental_inference(X, {}, init_seq_len, from, to);
}

sharedConstTensors NeuralNetwork::incremental_inference(
  sharedConstTensors X, sharedConstTensors label, unsigned int init_seq_len,
  unsigned int from, unsigned int to) {
  if (model_graph.getBatchSize() != X[0]->batch()) {
    model_graph.setBatchSize(X[0]->batch());
  }

  sharedConstTensors out;
  if (!validateInput(X))
    throw std::invalid_argument("Input validation failed.");

  if (!from) {
    model_graph.allocateTensors(ExecutionMode::INFERENCE);
  }

  int nn_foward;
  PROFILE_TIME_REGISTER_EVENT(nn_foward, "nn_forward");
  PROFILE_TIME_START(nn_foward);

  out = incremental_forwarding(from, to, X, label, false);

  PROFILE_TIME_END(nn_foward);

  /** @todo: deallocate tensor after incremental inference **/
  /** Clear the set inputs and labels */
  model_graph.setInputsLabels({}, {});

  return out;
}

std::vector<float *> NeuralNetwork::incremental_inference(
  unsigned int batch_size, const std::vector<float *> &input,
  const std::vector<float *> &label, unsigned int init_seq_len,
  unsigned int from, unsigned int to, bool output_hidden_state) {

  // auto start_in_neuralnet = std::chrono::high_resolution_clock::now();

  sharedConstTensors input_tensors, output_tensors;
  auto in_dim = getInputDimension();

  input_tensors.reserve(input.size());
  for (unsigned int idx = 0; idx < in_dim.size(); idx++) {
    in_dim[idx].batch(batch_size);
    input_tensors.emplace_back(MAKE_SHARED_TENSOR(Tensor::Map(
      input[idx], in_dim[idx].getDataLen() * sizeof(float), in_dim[idx], 0)));
  }

  // auto start_increment = std::chrono::high_resolution_clock::now();
  if (!label.empty()) {
    sharedConstTensors label_tensors;
    auto label_dim = getOutputDimension();
    label_tensors.reserve(label.size());
    for (unsigned int idx = 0; idx < label_dim.size(); idx++) {
      label_dim[idx].batch(batch_size);
      label_tensors.emplace_back(MAKE_SHARED_TENSOR(
        Tensor::Map(label[idx], label_dim[idx].getDataLen() * sizeof(float),
                    label_dim[idx], 0)));
    }
    output_tensors = incremental_inference(input_tensors, label_tensors,
                                           init_seq_len, from, to);
  } else {
    output_tensors =
      incremental_inference(input_tensors, init_seq_len, from, to);
  }
  // auto end_increment = std::chrono::high_resolution_clock::now();
  std::vector<float *> output;

  for (auto &out : output_tensors) {
    auto out_t = *out.get();
    float *last_out_buf_data;

    if (output_hidden_state) {
      std::cout << "Warning: output_hidden_state is not supported yet.\n"
                << "Returning last hidden state only...\n"
                << "Please free output memory after use!";
    }
    const size_t buf_size = batch_size * out_t.getDim().getFeatureLen();
    last_out_buf_data = new float[buf_size];

    if (out->getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16

      nntrainer::getComputeOps()->scopy_fp16_to_fp32(
        buf_size, out_t.getData<_FP16>(), 1, last_out_buf_data, 1);
#else
      throw std::invalid_argument("Error: enable-fp16 is not set");
#endif
    } else if (out->getDataType() == ml::train::TensorDim::DataType::FP32) {

      std::memcpy(last_out_buf_data, out_t.getData(), sizeof(float) * buf_size);
    }

    output.push_back(last_out_buf_data);
  }
  // auto end_net_inference = std::chrono::high_resolution_clock::now();
  // auto prepare =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(start_increment-start_in_neuralnet);
  // auto run_inf =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(end_increment-start_increment);;
  // auto out_gen =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(end_net_inference-end_increment);;
  // auto net_gen =
  // std::chrono::duration_cast<std::chrono::nanoseconds>(end_net_inference-start_in_neuralnet);

  // std::cout <<"prepare : "<< prepare.count() << " run_inf : "<<
  // run_inf.count() << " out_gen : "<< out_gen.count()<<std::endl; std::cout <<
  // "-------- net_inference: "<< net_gen.count() << std::endl;

  return output;
}

void NeuralNetwork::resetInputDimension(std::vector<TensorDim> dims) {
  model_graph.resetInputDimension(dims);
}

int NeuralNetwork::setDataset(const DatasetModeType &mode,
                              std::shared_ptr<ml::train::Dataset> dataset) {
  return setDataBuffer(mode, std::static_pointer_cast<DataBuffer>(dataset));
}

int NeuralNetwork::allocate(ExecutionMode mode) {
  model_graph.deallocateTensors();
  model_graph.allocateTensors(mode);

  return ML_ERROR_NONE;
}

int NeuralNetwork::deallocate() {
  try {
    model_graph.deallocateTensors(true);
    return ML_ERROR_NONE;
  } catch (const std::exception &e) {
    std::cerr << "Error occurred during deallocation of NeuralNetwork: "
              << e.what() << std::endl;
    return ML_ERROR_UNKNOWN;
  }
}

int NeuralNetwork::train(const std::vector<std::string> &values,
                         std::function<bool(void *)> stop_cb,
                         void *stop_user_data,
                         std::function<void(void *)> epoch_complete_cb,
                         void *epoch_user_data) {
  int status = ML_ERROR_NONE;

  if (data_buffers[static_cast<int>(DatasetModeType::MODE_TRAIN)] == nullptr) {
    ml_loge("Cannot initialize the model without the train data buffer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!opt) {
    ml_loge("Cannot train network without optimizer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  setTrainConfig(values);

  /** set batch size just before training */
  model_graph.setBatchSize(
    std::get<props::TrainingBatchSize>(model_flex_props));

  status = allocate(ExecutionMode::TRAIN);
  NN_RETURN_STATUS();

  status =
    train_run(stop_cb, stop_user_data, epoch_complete_cb, epoch_user_data);
  NN_RETURN_STATUS();

  /**
   * Free the memory needed for training before exiting.
   * Note that this does not free the weights for the model.
   * Weights of the model will be freed when the model is destroyed.
   */
  model_graph.deallocateTensors(false);
  return status;
}

/**
 * @brief     Run NeuralNetwork train with callback function by user
 */
int NeuralNetwork::train_run(
  std::function<bool(void *userdata)> stop_cb, void *stop_user_data,
  std::function<void(void *userdata)> epoch_complete_cb,
  void *epoch_user_data) {
  int status = ML_ERROR_NONE;

  if (!std::get<props::ContinueTrain>(model_flex_props)) {
    epoch_idx = 0;
    iter = 0;
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      (*iter)->clearOptVar();
    }
  }

  auto batch_size = std::get<props::TrainingBatchSize>(model_flex_props);

  auto const &outputs = model_graph.getOutputTensors();
  auto in_dims = model_graph.getInputDimension();
  auto label_dims = model_graph.getOutputDimension();

  auto &[train_buffer, valid_buffer, test_buffer] = data_buffers;

  if (train_buffer == nullptr) {
    ml_loge("[NeuralNetworks] there is no train dataset!");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /**
   * @brief run a single epoch with given callback, @a auto is used instead of
   * std::function for performance measure
   * @param buffer buffer to run
   * @param shuffle whether to shuffle or not
   * @param on_iteration_fetch function that will receive reference to stat,
   * buffer which will be called every time data is fetched and set
   * @param on_epoch_end function that will receive reference to stat,
   * buffer which will be called on the epoch end
   */
  auto run_epoch = [this, &in_dims, &label_dims, &outputs, batch_size](
                     DataBuffer *buffer, bool shuffle,
                     auto &&on_iteration_fetch, auto &&on_iteration_update_stat,
                     auto &&on_epoch_end, RunStats &stat) {
    /// @todo managing metrics must be handled here as well!! for now it is
    /// handled in individual callbacks
    // RunStats stat;

    stat.accuracy = 0.0;
    stat.loss = 0.0;
    stat.num_iterations = 0;
    stat.num_correct_predictions = 0;
    stat.max_epoch = getEpochs();
    stat.epoch_idx = epoch_idx;

    std::future<std::shared_ptr<IterationQueue>> future_iq =
      buffer->startFetchWorker(in_dims, label_dims, shuffle);
    while (true) {
      ScopedView<Iteration> iter_view = buffer->fetch();
      if (iter_view.isEmpty()) {
        break;
      }
      auto &iteration = iter_view.get();
      if (iteration.batch() != static_cast<unsigned int>(batch_size)) {
        /// @todo support partial batch
        continue;
      }

      auto const &labels = iteration.getLabelsRef();
      auto const &inputs = iteration.getInputsRef();
      model_graph.setInputsLabels(inputs, labels);

      on_iteration_fetch(stat, *buffer);
      on_iteration_update_stat(stat, outputs, labels);
    }
    future_iq.get();
    on_epoch_end(stat, *buffer);

    if (stat.num_iterations == 0) {
      throw std::runtime_error("No data came while buffer ran");
    }

    return stat;
  };

  auto train_for_iteration =
    [this, stop_cb, stop_user_data](RunStats &stat, DataBuffer &buffer) {
      ml_logi("train for iteration");
      forwarding(true, stop_cb, stop_user_data);
      backwarding(iter++, stop_cb, stop_user_data);

      // To avoid unconsidered memory leak, we need to clear the cache
      model_graph.flushCache();

      if (!stop_cb(stop_user_data)) {
        std::cout << "#" << epoch_idx << "/" << getEpochs();
        ml_logi("# %d / %d", epoch_idx, getEpochs());
        auto loss = getLoss();
        buffer.displayProgress(stat.num_iterations, loss);
      }
    };

  auto update_train_stat = [this](RunStats &stat,
                                  const std::vector<Tensor> &outputs,
                                  const std::vector<Tensor> &labels) {
    stat.loss += getLoss();
    stat.num_iterations++;
  };

  auto train_epoch_end = [this, stop_cb, stop_user_data](RunStats &stat,
                                                         DataBuffer &buffer) {
    if (stat.num_iterations != 0) {
      stat.loss /= static_cast<float>(stat.num_iterations);
    } else {
      std::cerr << "stat.num_iterations is 0" << std::endl;
      return;
    }
    auto &save_path = std::get<props::SavePath>(model_flex_props);
    if (!stop_cb(stop_user_data)) {
      if (!save_path.empty()) {
        save(save_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
      }

      std::cout << "#" << epoch_idx << "/" << getEpochs()
                << " - Training Loss: " << stat.loss;
      ml_logi("# %d / %d - Training Loss: %f", epoch_idx, getEpochs(),
              stat.loss);
      ml_logd("[NNTrainer] Training epoch %d / %d finished successfully.",
              epoch_idx, getEpochs());
    } else {
      ml_logd("[NNTrainer] Training stopped by stop callback function during "
              "epoch %d.",
              epoch_idx);
    }
  };

  auto eval_for_iteration = [this, batch_size, stop_cb, stop_user_data](
                              RunStats &stat, DataBuffer &buffer) {
    forwarding(false, stop_cb, stop_user_data);
  };

  auto update_eval_stat = [batch_size, &update_train_stat](
                            RunStats &stat, const std::vector<Tensor> &outputs,
                            const std::vector<Tensor> &labels) {
    auto model_out = outputs[0].argmax();
    auto label_out = labels[0].argmax();

    for (unsigned int b = 0; b < batch_size; b++) {
      if (model_out[b] == label_out[b])
        stat.num_correct_predictions++;
    }

    update_train_stat(stat, outputs, labels);
  };

  auto eval_epoch_end = [this, batch_size, max_acc = 0.0f,
                         min_loss = std::numeric_limits<float>::max()](
                          RunStats &stat, DataBuffer &buffer) mutable {
    if (stat.num_iterations != 0) {
      stat.loss /= static_cast<float>(stat.num_iterations);
    } else {
      std::cerr << "stat.num_iterations is 0" << std::endl;
      return;
    }
    stat.accuracy = stat.num_correct_predictions /
                    static_cast<float>(stat.num_iterations * batch_size) *
                    100.0f;

    if (stat.accuracy > max_acc ||
        (stat.accuracy == max_acc && stat.loss < min_loss)) {
      max_acc = stat.accuracy;
      /// @note this is not actually 'the' min loss for whole time but records
      /// when data change
      min_loss = stat.loss;
      auto &save_best_path = std::get<props::SaveBestPath>(model_flex_props);
      if (!save_best_path.empty()) {
        save(save_best_path);
      }
    }
    std::cout << " >> [ Accuracy: " << stat.accuracy
              << "% - Validation Loss : " << stat.loss << " ]";
    ml_logi("[ Accuracy: %.2f %% - Validation Loss: %.5f", stat.accuracy,
            stat.loss);
  };

  PROFILE_MEM_ANNOTATE("TRAIN START");
  auto epochs = getEpochs();
  ml_logd("[NNTrainer] Starts training. Current epoch: %d. Total epochs: %d.",
          epoch_idx + 1, getEpochs());
  for (epoch_idx = epoch_idx + 1; epoch_idx <= epochs; ++epoch_idx) {
    if (stop_cb(stop_user_data)) {
      --epoch_idx;
      break;
    }
    training = run_epoch(train_buffer.get(), true, train_for_iteration,
                         update_train_stat, train_epoch_end, training);
    if (valid_buffer) {
      validation = run_epoch(valid_buffer.get(), false, eval_for_iteration,
                             update_eval_stat, eval_epoch_end, validation);
    }
    std::cout << '\n';
    epoch_complete_cb(epoch_user_data);
  }
  PROFILE_MEM_ANNOTATE("TRAIN END");

  if (test_buffer) {
    std::cout << "Evaluation with test data...\n";
    testing = run_epoch(test_buffer.get(), false, eval_for_iteration,
                        update_eval_stat, eval_epoch_end, testing);
  }

  /** Clear the set inputs and labels */
  model_graph.setInputsLabels({}, {});

  return status;
}

void swap(NeuralNetwork &lhs, NeuralNetwork &rhs) {
  {
    using std::swap;

    swap(lhs.model_props, rhs.model_props);
    swap(lhs.model_flex_props, rhs.model_flex_props);
    swap(lhs.load_path, rhs.load_path);
    swap(lhs.epoch_idx, rhs.epoch_idx);
    swap(lhs.iter, rhs.iter);
    swap(lhs.loss, rhs.loss);
    swap(lhs.opt, rhs.opt);
    swap(lhs.data_buffers, rhs.data_buffers);
    swap(lhs.initialized, rhs.initialized);
    swap(lhs.model_graph, rhs.model_graph);
    swap(lhs.graph_representation, rhs.graph_representation);
    swap(lhs.compiled, rhs.compiled);
    swap(lhs.loadedFromConfig, rhs.loadedFromConfig);
  }
}

int NeuralNetwork::addLayer(NodeType layer) {
  int status = ML_ERROR_NONE;

  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  /** Insert the layer to the graph */
  model_graph.addLayer(layer);
  graph_representation.push_back(layer);

  return status;
}

NeuralNetwork &NeuralNetwork::copyConfiguration(NeuralNetwork &from) {
  if (this != &from) {
    model_props = from.model_props;
    model_flex_props = from.model_flex_props;
    loss = from.loss;
    opt = from.opt;

    NetworkGraph f_graph = from.getNetworkGraph();
    for (auto &l_node : f_graph.getLayerNodes()) {
      addLayer(static_cast<std::shared_ptr<ml::train::Layer>>(
        l_node->cloneConfiguration()));
    }
  }
  return *this;
}

NeuralNetwork::GraphType
NeuralNetwork::getUnsortedLayers(const std::string &input_layer,
                                 const std::string &output_layer) {
  return model_graph.getUnsortedLayers(input_layer, output_layer);
}

int NeuralNetwork::setOptimizer(
  std::shared_ptr<ml::train::Optimizer> optimizer) {
  if (initialized) {
    ml_loge("Cannot set optimizer if already initialized");
    return ML_ERROR_NOT_SUPPORTED;
  }

  opt = std::static_pointer_cast<OptimizerWrapped>(optimizer);

  return ML_ERROR_NONE;
}

int NeuralNetwork::setDataBuffer(const DatasetModeType &mode,
                                 std::shared_ptr<DataBuffer> data_buffer) {
  if (data_buffer == nullptr) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->data_buffers[static_cast<int>(mode)] = data_buffer;

  return ML_ERROR_NONE;
}

int NeuralNetwork::getLayer(const char *name,
                            std::shared_ptr<ml::train::Layer> *layer) {
  // We provide the layer change through the api with user's responsibility.
  //
  // if (compiled) {
  //   ml_loge("Cannot get compiled layer.");
  //   return ML_ERROR_NOT_SUPPORTED;
  // }

  *layer = std::static_pointer_cast<ml::train::Layer>(
    model_graph.getLayerNode(std::string(name)));
  return ML_ERROR_NONE;
}

void NeuralNetwork::printMetrics(std::ostream &out, unsigned int flags) {
  switch (flags) {
  case ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS:
    out << training.loss << std::endl;
    break;

  case ML_TRAIN_SUMMARY_MODEL_VALID_LOSS:
    out << validation.loss << std::endl;
    break;

  case ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY:
    out << validation.accuracy << std::endl;
    break;

  default:
    break;
  }
}

void NeuralNetwork::printPreset(std::ostream &out, unsigned int preset) {
  /** print neuralnet metrics */
  printMetrics(out, preset);
  if (preset > ML_TRAIN_SUMMARY_TENSOR)
    return;

  LayerNode::PrintPreset layer_preset = LayerNode::PrintPreset::PRINT_NONE;

  ///@todo match flags with preset
  unsigned int flags = PRINT_INST_INFO | PRINT_GRAPH_INFO | PRINT_PROP |
                       PRINT_OPTIMIZER | PRINT_METRIC;

  switch (preset) {
  case ML_TRAIN_SUMMARY_TENSOR:
    layer_preset = LayerNode::PrintPreset::PRINT_ALL;
    break;
  case ML_TRAIN_SUMMARY_LAYER:
    layer_preset = initialized ? LayerNode::PrintPreset::PRINT_SUMMARY
                               : LayerNode::PrintPreset::PRINT_SUMMARY_META;
    break;
  case ML_TRAIN_SUMMARY_MODEL:
    break;
  default:
    throw std::invalid_argument("given verbosity is invalid");
  }

  print(out, flags, layer_preset);
}

void NeuralNetwork::addWithReferenceLayers(
  const std::vector<std::shared_ptr<ml::train::Layer>> &reference,
  const std::string &scope, const std::vector<std::string> &input_layers,
  const std::vector<std::string> &start_layers,
  const std::vector<std::string> &end_layers,
  ml::train::ReferenceLayersType type,
  const std::vector<std::string> &type_properties) {
  std::vector<NodeType> casted_reference;
  casted_reference.reserve(reference.size());
  for (auto &node : reference) {
    casted_reference.emplace_back(std::static_pointer_cast<LayerNode>(node));
  }

  addWithReferenceLayers(casted_reference, scope, input_layers, start_layers,
                         end_layers, type, type_properties);
}

void NeuralNetwork::addWithReferenceLayers(
  const std::vector<std::shared_ptr<LayerNode>> &reference,
  const std::string &scope, const std::vector<std::string> &input_layers,
  const std::vector<std::string> &start_layers,
  const std::vector<std::string> &end_layers,
  ml::train::ReferenceLayersType type,
  const std::vector<std::string> &type_properties) {
  /// @todo below configuration should be extracted as a free function to make
  /// it more testable, and reused inside graph interpreter

  /// @note we can exploit connection to connection more fine grained, for now
  /// it is not supported but we can easily make this supported
  std::vector<std::shared_ptr<LayerNode>> nodes;
  nodes.reserve(reference.size());
  for (auto &node : reference) {
    nodes.push_back(node->cloneConfiguration());
  }

  auto start_conns =
    std::vector<Connection>(start_layers.begin(), start_layers.end());
  auto input_conns =
    std::vector<Connection>(input_layers.begin(), input_layers.end());
  auto end_conns =
    std::vector<Connection>(end_layers.begin(), end_layers.end());

  std::vector<std::unique_ptr<GraphRealizer>> realizers;

  realizers.emplace_back(new PreviousInputRealizer(start_conns));
  realizers.emplace_back(new SliceRealizer(start_conns, end_conns));

  if (!input_conns.empty()) {
    realizers.emplace_back(new InputRealizer(start_conns, input_conns));
  }

  if (type == ml::train::ReferenceLayersType::RECURRENT) {
    realizers.emplace_back(
      new RecurrentRealizer(type_properties, input_conns, end_conns));
  }

  if (!scope.empty()) {
    realizers.emplace_back(
      new RemapRealizer([&scope, &input_conns](std::string &name) {
        for (auto &i : input_conns) {
          if (i.getName() == name) {
            return;
          }
        }
        name = scope + "/" + name;
      }));
  }

  for (auto &realizer : realizers) {
    nodes = realizer->realize(nodes);
  }

  for (auto &node : nodes) {
    addLayer(node);
  }
}

void NeuralNetwork::exportTo(Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
  exporter.saveResult(model_props, method, this);
  exporter.saveResult(model_flex_props, method, this);
}

void NeuralNetwork::print(std::ostream &out, unsigned int flags,
                          LayerNode::PrintPreset layerPrintPreset) {
  if (flags & PRINT_INST_INFO) {
    /// @todo uncomment this after implement getProperty (#1875)
    // out << "===================";
    // printInstance(out, this);
  }

  if (flags & PRINT_GRAPH_INFO) {
    unsigned int total_col_size = 80;
    std::vector<unsigned int> column_size = {20, 20, 20, 20};
    auto print_graph_layer_info =
      [column_size](std::ostream &out, std::vector<std::string> layer_info) {
        const auto &trim_string = [](std::string str,
                                     unsigned int column_width) {
          return str.size() < column_width ? str
                                           : str.substr(0, column_width - 1);
        };

        for (unsigned int i = 0; i < column_size.size(); ++i) {
          out << std::setw(column_size[i])
              << trim_string(layer_info[i], column_size[i]);
        }
        out << "\n";
      };

    out << std::string(total_col_size, '=') << '\n';
    print_graph_layer_info(
      out, {"Layer name", "Layer type", "Output dimension", "Input layer"});
    out << std::string(total_col_size, '=') << '\n';
    if (compiled) {
      props::GenericShape dim_property;

      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           iter++) {
        std::string first_dim;
        if (iter->getOutputDimensions().empty()) {
          first_dim = "";
        } else {
          dim_property.set(iter->getOutputDimensions()[0]);
          first_dim = to_string(dim_property);
        }
        const std::vector<std::string> &input_layer_names =
          iter->getInputConnections();
        std::string first_input_name =
          input_layer_names.empty() ? "" : input_layer_names[0];
        print_graph_layer_info(
          out, {iter->getName(), iter->getType(), first_dim, first_input_name});
        for (unsigned int i = 1; i < input_layer_names.size(); ++i) {
          dim_property.set(iter->getInputDimensions()[i]);
          print_graph_layer_info(out, {"", "", "", input_layer_names[i]});
        }
        out << std::string(total_col_size,
                           iter == model_graph.cend() - 1 ? '=' : '-')
            << '\n';
      }
    } else {
      auto &input_connection =
        std::get<std::vector<props::InputConnection>>(model_props);
      auto model_input = std::vector<Connection>(input_connection.begin(),
                                                 input_connection.end());
      auto is_actually_an_input_node =
        [model_input](graph_const_iterator<LayerNode> node) {
          return node->hasInputShapeProperty() or
                 std::any_of(model_input.begin(), model_input.end(),
                             [node](auto &conn) {
                               return node->getName() == conn.getName();
                             });
        };

      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           iter++) {
        const std::vector<std::string> &input_layer_names =
          iter->getInputConnections();

        /// @brief connection information.
        // Intended comment.
        // std::string first_input_name =
        //   input_layer_names.empty()
        //     ? (is_actually_an_input_node(iter) || iter ==
        //     model_graph.cbegin()
        //          ? ""
        //          : (iter - 1)->getName())
        //     : input_layer_names[0];
        print_graph_layer_info(out, {iter->getName(), iter->getType(), "", ""});
        for (unsigned int i = 1; i < input_layer_names.size(); ++i) {
          print_graph_layer_info(out, {"", "", "", ""});
        }
        out << std::string(total_col_size,
                           iter == model_graph.cend() - 1 ? '=' : '-')
            << '\n';
      }
    }
  }

  if (flags & PRINT_PROP) {
    /// @todo print neuralnet property
    /// @todo print mode (if it is eval or training)
  }

  if (flags & PRINT_OPTIMIZER) {
    /// @todo print optimizer (with print optimizer prop)
  }

  if (flags & PRINT_METRIC) {
    /// @todo print metric (currently it is done at printPreset as a
    /// workaround)
    /// @todo print loss function when it is not initialized. (if it is
    /// initialized, loss layer will be printed)
  }

  if (model_graph.empty()) {
    out << "model is empty!" << std::endl;
    return;
  }

  /** print layer properties */
  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++)
    (*iter)->printPreset(out, layerPrintPreset);

  /// @todo Add status to check neuralnet has been run. #290
}

void NeuralNetwork::forEachLayer(
  std::function<void(ml::train::Layer &, RunLayerContext &, void *)> fn,
  void *user_data) {
  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
    auto ln = std::static_pointer_cast<LayerNode>(*iter).get();
    fn(*ln, std::forward<RunLayerContext &>(ln->getRunContext()), user_data);
  };
}

void NeuralNetwork::exports(const ml::train::ExportMethods &method,
                            const std::string file_path) {
  switch (method) {
  case ml::train::ExportMethods::METHOD_TFLITE: {
#ifdef ENABLE_TFLITE_INTERPRETER
    nntrainer::TfliteInterpreter interpreter;

    /// We will call "serialize" method for the model which is already trained
    /// or allocated. So, we need to call deallocateTensors first to make sure
    /// `dealloc_weights == false`
    model_graph.deallocateTensors();
    model_graph.allocateTensors(ExecutionMode::INFERENCE);
    model_graph.setBatchSize(1); // For now, to inference batch size to be 1
    interpreter.serialize(graph_representation, file_path);
    model_graph.deallocateTensors();
#else
    throw std::runtime_error{
      "Export methods METHOD_TFLITE is not supported. Please enable tflite "
      "interpreter by set ENABLE_TFLITE_INTERPRETER=1"};
#endif
    break;
  }
  case ml::train::ExportMethods::METHOD_FLATBUFFER: {

    /**
     * @todo The current FLATBUFFER exporter only supports TRAIN execution mode.
     * It should be updated to support both train and inference mode.
     * It would be more natural to support inference by default since tflite is
     * typically used solely for inference
     */
    model_graph.deallocateTensors();
    model_graph.allocateTensors(ExecutionMode::TRAIN);
    break;
  }
  default:
    throw std::runtime_error{"Unsupported export method"};
  }
}
} /* namespace nntrainer */

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
#include "model.h"
#include "model_common_properties.h"
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <compute_ops.h>
#include <cstdint>
#include <cstdio>
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
#include <int4_utils.h>
#include <model_loader.h>
#include <quantizer.h>
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
#include <safetensors_util.h>
#include <slice_realizer.h>
#include <util_func.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <tflite_interpreter.h>
#endif

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <chrono>
#include <cuda_context_manager.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

/**
 * @brief Internal enum values for nntrainer to summarize model accuracy & loss
 */
#define ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS 101
#define ML_TRAIN_SUMMARY_MODEL_VALID_LOSS 102
#define ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY 103

namespace nntrainer {

namespace {

Tensor mapExternalTensor(float *buf, const TensorDim &dim) {
  const unsigned int bytes = static_cast<unsigned int>(
    static_cast<size_t>(dim.getDataLen()) * dim.getDataTypeSize());

  switch (dim.getDataType()) {
  case TensorDim::DataType::FP16:
  case TensorDim::DataType::UINT16:
  case TensorDim::DataType::QINT16:
    return Tensor::Map<uint16_t>(reinterpret_cast<uint16_t *>(buf), bytes, dim,
                                 0);
  case TensorDim::DataType::UINT8:
  case TensorDim::DataType::UINT4:
  case TensorDim::DataType::QINT8:
  case TensorDim::DataType::QINT4:
  case TensorDim::DataType::Q4_K:
  case TensorDim::DataType::Q6_K:
  case TensorDim::DataType::Q4_0:
    return Tensor::Map<uint8_t>(reinterpret_cast<uint8_t *>(buf), bytes, dim,
                                0);
  case TensorDim::DataType::UINT32:
  case TensorDim::DataType::BCQ:
    return Tensor::Map<uint32_t>(reinterpret_cast<uint32_t *>(buf), bytes, dim,
                                 0);
  case TensorDim::DataType::FP32:
  case TensorDim::DataType::NONE:
  default:
    return Tensor::Map<float>(buf, bytes, dim, 0);
  }
}

} // namespace

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

  // Step 1 (GPU generalization): opt the graph-wide memory pool into the GPU
  // (SVM) allocator so RunContext activation/weight tensors become GPU-resident
  // and avoid the per-layer host round-trip. Conservative and gated:
  //   - only under an OpenCL build,
  //   - only when NNTR_GPU_SVM_POOL is set (default off => zero behavior change),
  //   - only when the graph actually contains an engine=gpu node.
  // Pure-CPU graphs and OpenCL-disabled builds always keep the "cpu" allocator,
  // so CPU execution stays byte-identical. See
  // tensor/cl_operations/GPU_GENERALIZATION_PLAN.md.
  std::string engine_name = "cpu";
#if defined(ENABLE_OPENCL) && ENABLE_OPENCL == 1
  if (std::getenv("NNTR_GPU_SVM_POOL") != nullptr) {
    for (auto &n : graph_representation) {
      if (n->isComputeEngineGPU()) {
        engine_name = "gpu";
        break;
      }
    }
  }
#endif
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // engine=cuda: route the graph's tensor pool through the CUDA Unified Memory
  // allocator (cudaMallocManaged) so weights/activations are device-resident,
  // letting the cuBLAS / QINT4 GPU FC paths engage instead of the host
  // fallback. UVM is host-coherent, so unported host layers keep working on the
  // same pointers. Default ON when the graph has a cuda node;
  // NNTR_CUDA_UVM_POOL=0 forces the host allocator (correct, runs on host).
  if (engine_name == "cpu") {
    const char *uvm = std::getenv("NNTR_CUDA_UVM_POOL");
    if (!(uvm != nullptr && uvm[0] == '0')) {
      for (auto &n : graph_representation) {
        if (n->isComputeEngineCUDA()) {
          engine_name = "cuda";
          break;
        }
      }
    }
  }
#endif

  model_graph = NetworkGraph(fsu, mode, fsu_path, lookahead, tensor_format,
                             tensor_type, engine_name);

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
      {
        // NNTR_LAYER_PROF: per-layer-type HOST time (no clFinish). Host-blocking
        // ops (e.g. host rmsnorm) show their full cost; async GPU ops show only
        // enqueue time -> reveals where the host timeline is spent. Dumps the
        // running per-type totals to stderr every 64 layer calls.
        static const bool lprof = std::getenv("NNTR_LAYER_PROF") != nullptr;
        if (lprof) {
          auto t0 = std::chrono::high_resolution_clock::now();
          node->forwarding(training);
          auto t1 = std::chrono::high_resolution_clock::now();
          static std::unordered_map<std::string, std::pair<double, int>> acc;
          static int total = 0;
          auto &e = acc[node->getType()];
          e.first += std::chrono::duration<double, std::milli>(t1 - t0).count();
          e.second++;
          if (++total % 64 == 0) {
            std::string s;
            for (auto &kv : acc)
              s += kv.first + "=" + std::to_string(kv.second.first) + "ms/" +
                   std::to_string(kv.second.second) + " ";
            std::fprintf(stderr, "[lprof] %s\n", s.c_str());
          }
        } else {
          node->forwarding(training);
        }
      }
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

// recq R4 lightweight feed pass (defined in libnntrainer blas_kernels.cpp):
// true while the decode loop runs a host-only forward to refresh the embedding
// output; lets us skip every non-input-embedding node's host forward.
bool recq_skip_all_active();

// CUDA M2-B embed-only feed flag (decoupled from the OpenCL recq skip): true
// while the single-capture replay re-runs ONLY the embedding nodes to refresh the
// pinned staging buffer for the new token id; the forwarding_op below then skips
// every other node's host forward (the GPU work comes from the replayed graph).
static bool g_m2b_skip_all = false;

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

    // recq R4 feed pass: run ONLY the input-embedding nodes (they refresh the
    // residual seed for this token); skip every other node's host forward so the
    // GPU forward is supplied solely by the recorded-chain replay (lightweight
    // feed -- avoids re-running the full per-layer host iteration).
    static const bool _recq_feed = std::getenv("NNTR_RECQ_REPLAY") != nullptr;
    if ((_recq_feed && recq_skip_all_active()) || g_m2b_skip_all) {
      const std::string &nm = node->getName();
      if (nm != "embedding0" && nm != "per_layer_input_embedding")
        return;
    }

    auto f = std::get<0>(node->getExecutionOrder());
    if (exec_mode == ExecutionMode::TRAIN or
        (exec_mode == ExecutionMode::INFERENCE and !fsu_mode)) {
      // auto start_layer =
      //      std::chrono::high_resolution_clock::now(); // log the
      //      start_prefill time
      model_graph.flushCacheExcept(f);
      {
        // NNTR_LAYER_PROF: per-layer-type HOST time (no clFinish). Host-blocking
        // ops (host rmsnorm etc.) show full cost; async GPU ops show only enqueue
        // time -> where the host timeline goes. Dumps per-type totals every 64.
        static const bool lprof = std::getenv("NNTR_LAYER_PROF") != nullptr;
        if (lprof) {
          auto t0 = std::chrono::high_resolution_clock::now();
          node->incremental_forwarding(from, to, training);
          auto t1 = std::chrono::high_resolution_clock::now();
          static std::unordered_map<std::string, std::pair<double, int>> acc;
          static int total = 0;
          auto &e = acc[node->getType()];
          e.first += std::chrono::duration<double, std::milli>(t1 - t0).count();
          e.second++;
          if (++total % 64 == 0) {
            std::string s;
            for (auto &kv : acc)
              s += kv.first + "=" + std::to_string(kv.second.first) + "ms/" +
                   std::to_string(kv.second.second) + " ";
            std::fprintf(stderr, "[lprof] %s\n", s.c_str());
          }
        } else {
          node->incremental_forwarding(from, to, training);
        }
      }
      // NNTR_DUMP_STATS: per-node output min/max + NaN/Inf flag, to pinpoint the
      // first layer whose output diverges (Orin/sm_87 triage). Draining first so
      // any async GPU op's result is visible to the host stat read.
      static const bool dump_stats = std::getenv("NNTR_DUMP_STATS") != nullptr;
      if (dump_stats) {
        try {
          cudaDeviceSynchronize();
          Tensor &o = node->getOutput(0);
          float mn = o.minValue(), mx = o.maxValue();
          bool bad = std::isnan(mn) || std::isnan(mx) || std::isinf(mn) ||
                     std::isinf(mx);
          std::fprintf(stderr, "[stats] %-30s %-16s min=%.4g max=%.4g%s\n",
                       node->getName().c_str(), node->getType().c_str(), mn, mx,
                       bad ? "  <<< NaN/Inf" : "");
        } catch (...) {
        }
      }
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

namespace {

/**
 * @brief Resolve the data type a weight will actually be stored as.
 *
 * Mirrors the per-weight policy of the layer save overrides: bias-like tensors
 * (height == 1) are not block-quantized and stay in their original type.
 */
TensorDim::DataType resolveStoredDtype(const Tensor &weight,
                                       TensorDim::DataType requested) {
  if (requested == TensorDim::DataType::NONE ||
      requested == weight.getDataType())
    return weight.getDataType();

  if (nntrainer::safetensors::isQuantized(requested) &&
      weight.getDim().height() == 1)
    return weight.getDataType();

  return requested;
}

} // namespace

void NeuralNetwork::save(
  const std::string &file_path, ml::train::ModelFormat format,
  TensorDim::DataType dtype,
  const std::map<std::string, TensorDim::DataType> &layer_dtype_map,
  ml::train::ISA target_isa) {
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
      layer_node->save(model_file, false, exec_mode, target_dtype, target_isa);
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
  case ml::train::ModelFormat::MODEL_FORMAT_ONNX: {
    throw nntrainer::exception::not_supported(
      "saving with ONNX format is not supported yet.");
    break;
  }
  case ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS: {
    // Delegate the data section to the same per-layer save() the BIN path
    // uses so the quantized bytes are byte-identical: each layer override
    // applies its own quantization policy (e.g. embedding/tie-word-embedding
    // do not transpose, shared weights are written once on first access),
    // which a generic quantizer here could not replicate. Bytes go to a temp
    // file first so per-weight sizes are known before the header is written.
    const std::string tmp_path = file_path + ".nntrtmp";
    std::vector<safetensors::TensorEntry> entries;

    {
      auto tmp_file = checkedOpenStream<std::ofstream>(
        tmp_path, std::ios::out | std::ios::binary | std::ios::trunc);

      std::unordered_set<const Tensor *> visited_st;
      size_t data_offset = 0;

      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           iter++) {
        const auto &layer_node = *iter;
        auto it = layer_dtype_map.find(layer_node->getName());
        const auto requested =
          (it != layer_dtype_map.end()) ? it->second : dtype;
        auto &rc = layer_node->getRunContext();

        // Collect the weights this layer will actually write: first-access
        // only (shared weights are saved once), deduped across the graph.
        struct WInfo {
          const Tensor *t;
          TensorDim::DataType stored;
        };
        std::vector<WInfo> wlist;
        for (unsigned int i = 0; i < rc.getNumWeights(); ++i) {
          if (!rc.isGradientFirstAccess(i))
            continue;
          const Tensor &t = rc.getWeight(i);
          if (!visited_st.insert(&t).second)
            continue;
          wlist.push_back({&t, resolveStoredDtype(t, requested)});
        }

        // Write this layer's weights exactly as the BIN path would.
        const auto start = static_cast<size_t>(tmp_file.tellp());
        layer_node->save(tmp_file, false, exec_mode, requested, target_isa);
        const auto layer_bytes = static_cast<size_t>(tmp_file.tellp()) - start;

        // Map the written bytes back to per-weight header entries. At most one
        // weight per layer is block-quantized; the rest are stored as-is, so
        // the quantized weight's size is whatever remains.
        size_t known = 0;
        int quant_count = 0;
        for (const auto &w : wlist) {
          if (safetensors::isQuantized(w.stored))
            ++quant_count;
          else
            known += w.t->getMemoryBytes();
        }
        NNTR_THROW_IF(quant_count > 1, std::runtime_error)
          << "safetensors save: layer '" << layer_node->getName()
          << "' has multiple quantized weights, which is not supported.";

        size_t assigned = 0;
        for (const auto &w : wlist) {
          const auto &dim = w.t->getDim();
          const bool is_quant = safetensors::isQuantized(w.stored);
          const size_t wsize =
            is_quant ? (layer_bytes - known) : w.t->getMemoryBytes();

          safetensors::TensorEntry entry;
          entry.name = w.t->getName();
          entry.offset_start = data_offset;
          entry.offset_end = data_offset + wsize;
          if (is_quant) {
            // Quantized blobs are opaque bytes (U8) with a 1-D byte shape;
            // the native type and logical shape live in extension fields.
            entry.dtype = safetensors::dtypeToString(w.stored); // "U8"
            entry.shape = {wsize};
            entry.nntr_dtype = safetensors::nntrDtypeName(w.stored);
            entry.nntr_shape = {dim.batch(), dim.channel(), dim.height(),
                                dim.width()};
          } else {
            entry.dtype = safetensors::dtypeToString(w.stored);
            entry.shape = {dim.batch(), dim.channel(), dim.height(),
                           dim.width()};
          }
          entries.push_back(std::move(entry));
          data_offset += wsize;
          assigned += wsize;
        }

        NNTR_THROW_IF(assigned != layer_bytes, std::runtime_error)
          << "safetensors save: byte accounting mismatch for layer '"
          << layer_node->getName() << "' (wrote " << layer_bytes << ", mapped "
          << assigned << ").";
      }

      tmp_file.close();
    }

    // Embed an nntrainer dtype summary so a quantized file can be inspected
    // and identified without an accompanying nntr_config.json.
    std::map<std::string, std::string> metadata;
    bool any_quant = false;
    bool any_q4_0 = false;
    for (const auto &e : entries) {
      any_quant = any_quant || !e.nntr_dtype.empty();
      any_q4_0 = any_q4_0 || e.nntr_dtype == "Q4_0";
    }
    if (any_quant)
      metadata["nntr_format"] = "nntr-safetensors-v1";
    // Q4_0 is repacked into an ISA-specific layout (x86: q4_0x8, ARM: q4_0x4)
    // that is indistinguishable from the header alone, so record which one was
    // produced. DEFAULT resolves to the build platform's layout. Only emitted
    // when a Q4_0 tensor is present, since no other type depends on the ISA.
    if (any_q4_0) {
      const char *isa_str;
      switch (target_isa) {
      case ml::train::ISA::X86:
        isa_str = "x86";
        break;
      case ml::train::ISA::ARM:
        isa_str = "arm";
        break;
      default: // DEFAULT -> the compiled backend's layout
#if defined(__aarch64__) || defined(__arm__)
        isa_str = "arm";
#else
        isa_str = "x86";
#endif
        break;
      }
      metadata["nntr_q4_0_isa"] = isa_str;
    }

    // Write: [8-byte header_size][header (padded to 8)][raw weight data]
    const std::string header_json = safetensors::buildHeader(entries, metadata);
    const uint64_t header_size = static_cast<uint64_t>(header_json.size());
    // safetensors layout: [8-byte header length][header JSON][tensor raw data]
    auto st_file = checkedOpenStream<std::ofstream>(
      file_path, std::ios::out | std::ios::binary | std::ios::trunc);
    // [8-byte header length]
    st_file.write(reinterpret_cast<const char *>(&header_size),
                  sizeof(header_size));
    // [header JSON: per-tensor dtype/shape/offsets + __metadata__]
    st_file.write(header_json.data(),
                  static_cast<std::streamsize>(header_json.size()));
    // [tensor raw data]
    {
      std::ifstream data_in(tmp_path, std::ios::in | std::ios::binary);
      st_file << data_in.rdbuf();
    }
    st_file.close();
    std::remove(tmp_path.c_str());
    break;
  }
  default:
    throw nntrainer::exception::not_supported(
      "saving with given format is not supported yet");
  }
}

void NeuralNetwork::load(const std::string &file_path,
                         ml::train::ModelFormat format) {
  /// @todo this switch case should be delegating the function call only. It's
  /// not delegating for now as required logics are manageable for now.

  bool fsu_mode = std::get<props::Fsu>(model_flex_props);

  const std::regex reg_("\\s*\\;\\s*");
  auto v = split(file_path, reg_);

  size_t start_from = 0;
  std::vector<std::pair<size_t, size_t>> file_offset;
  std::unordered_set<const Tensor *> visited_weights;
  // A QINT4 record's on-disk size depends on its container: the shared
  // plain container (qscheme PER_CHANNEL_AFFINE at the record head; the
  // PR#3978 form) carries fp32 scales plus KAI pad and is NOT the in-memory
  // Section A size that getMemoryBytes() reports. Peek each QINT4 record's
  // qscheme so the running offset matches the actual file layout.
  std::ifstream qint4_peek_stream;
  bool qint4_peek_tried = false;
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
      weight->getVariableRef().setFileOffset(start_from);
      ///@todo instead of checking the data type,
      /// we may need to create a common parent class for
      /// quantized tensors, requiring qparam to be saved
      /// and creating a common interface to check if qparam is needed
      /// this kind of type checking should be avoided
      if (tensor_data_type != TensorDim::DataType::FP32 &&
          tensor_data_type != TensorDim::DataType::FP16 &&
          tensor_data_type != TensorDim::DataType::Q6_K &&
          tensor_data_type != TensorDim::DataType::Q4_0) {
        // for tensor with qparam
        size += sizeof(uint16_t);
      }
      if (tensor_data_type == TensorDim::DataType::QINT4) {
        if (!qint4_peek_tried) {
          qint4_peek_tried = true;
          qint4_peek_stream.open((v.size() == 2) ? v[1] : v[0],
                                 std::ios::in | std::ios::binary);
        }
        uint16_t disk_scheme = 0xFFFF;
        if (qint4_peek_stream.is_open()) {
          qint4_peek_stream.clear();
          qint4_peek_stream.seekg(static_cast<std::streamoff>(start_from),
                                  std::ios::beg);
          qint4_peek_stream.read(reinterpret_cast<char *>(&disk_scheme),
                                 sizeof(uint16_t));
        }
        if (qint4_peek_stream.is_open() && qint4_peek_stream.good() &&
            disk_scheme ==
              static_cast<uint16_t>(QScheme::PER_CHANNEL_AFFINE)) {
          const TensorDim &d = weight->getDim();
          size = sizeof(uint16_t) +
                 Int4Utils::plainRecordPayloadBytes(d.width(), d.height());
        }
      }
      file_offset.emplace_back(std::make_pair(start_from, size));
      start_from += size;
    }
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
    auto f_path = (v.size() == 2) ? v[1] : v[0];

    auto model_file =
      checkedOpenStream<std::ifstream>(f_path, std::ios::in | std::ios::binary);

#if defined(_WIN32)
    HANDLE hFile, hMap;
#endif

    if (exec_mode == ml::train::ExecutionMode::INFERENCE) {
      // Always keep a long-lived fd open during inference. Virtual (slim)
      // tensors capture this fd at read-time and use it later in activate()
      // to mmap their backing region on demand. Without it, virtual tensors
      // end up with fd=-1 and activate() returns MAP_FAILED, segfaulting on
      // first use (e.g. SlimMoE expert weights when MMAP_READ=true).
      model_file_fd = open(f_path.c_str(), O_RDONLY);
      NNTR_THROW_IF((model_file_fd == -1), std::invalid_argument)
        << "Cannot open file : " << f_path;

      // Share a single read-only mmap across load workers. Per-worker mmap of
      // the full weight file can exceed Android's virtual memory or mmap-count
      // limits for large models.
      //
      // Each worker reads from its own file_offset, so sharing the mapped
      // region is safe. Drop the region only after all workers have joined.
      void *shared_mmap_ptr = MAP_FAILED;
      size_t shared_mmap_size = 0;
#if !defined(_WIN32)
      if (MMAP_READ) {
        struct stat st {};
        NNTR_THROW_IF((::fstat(model_file_fd, &st) == -1),
                      std::invalid_argument)
          << "Cannot get file info (fstat): " << f_path;
        shared_mmap_size = static_cast<size_t>(st.st_size);
        shared_mmap_ptr = ::mmap(nullptr, shared_mmap_size, PROT_READ,
                                 MAP_PRIVATE, model_file_fd, 0);
        NNTR_THROW_IF((shared_mmap_ptr == MAP_FAILED), std::runtime_error)
          << "mmap failed for " << f_path << " (" << shared_mmap_size
          << " bytes)";
        // Prefetch the whole weight region: the parallel workers each read a
        // node's (sequential) sub-range, so MADV_RANDOM was defeating readahead
        // and every 4 KB page faulted individually (cold cache, dropped by the
        // MADV_DONTNEED below each run) -> ~24s aggregate read vs ~1s for a
        // sequential `cat` of the same file. WILLNEED kicks off readahead so the
        // workers hit warm pages.
        (void)::posix_madvise(shared_mmap_ptr, shared_mmap_size,
                              POSIX_MADV_WILLNEED);
      }
#endif

      // Bounded-concurrency parallel load. Spawning one std::thread per graph
      // node (250+ on a 4B model) oversubscribes the CPU and collapses on the
      // glibc malloc-arena lock — every worker allocates its tensor buffer at
      // the same time — which for large models stalls the load effectively
      // forever (observed: Qwen3-4B host quantize hung with ~250 threads all
      // parked in futex_wait). Cap in-flight workers at the hardware
      // concurrency and pull nodes off a shared atomic cursor. Each node is
      // still read exactly once; only the degree of parallelism changes, so
      // the loaded bytes are identical to the per-node-thread version.
      std::vector<std::shared_ptr<LayerNode>> load_nodes(model_graph.cbegin(),
                                                         model_graph.cend());

      auto read_one = [&](const std::shared_ptr<LayerNode> &node) {
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
                     std::numeric_limits<size_t>::max(), true, model_file_fd);

          // Early unmap: let the OS reclaim the working set ASAP
          UnmapViewOfFile(view);
          CloseHandle(hMap);
          CloseHandle(hFile);
#else
          // POSIX: read from the parent-owned shared mmap. No per-thread
          // mmap/munmap — see the comment on shared_mmap_ptr above.
          char *view = static_cast<char *>(shared_mmap_ptr);
          node->read(view, false, exec_mode, fsu_mode,
                     std::numeric_limits<size_t>::max(), true, model_file_fd);
#endif
        }
      };

      unsigned int hw_threads = std::thread::hardware_concurrency();
      if (hw_threads == 0)
        hw_threads = 4;
      const size_t worker_count =
        std::min<size_t>(load_nodes.size(), static_cast<size_t>(hw_threads));
      std::atomic<size_t> load_cursor{0};
      std::vector<std::thread> threads;
      threads.reserve(worker_count);
      for (size_t worker = 0; worker < worker_count; ++worker) {
        threads.emplace_back([&]() {
          size_t i;
          while ((i = load_cursor.fetch_add(1)) < load_nodes.size())
            read_one(load_nodes[i]);
        });
      }
      for (auto &t : threads) {
        if (t.joinable())
          t.join();
      }

#if !defined(_WIN32)
      if (shared_mmap_ptr != MAP_FAILED) {
        (void)::posix_madvise(shared_mmap_ptr, shared_mmap_size,
                              POSIX_MADV_DONTNEED);
        ::munmap(shared_mmap_ptr, shared_mmap_size);
      }
#endif
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
  case ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS: {
    NNTR_THROW_IF(!initialized, std::runtime_error)
      << "Cannot load safetensors if not initialized yet, path: " << file_path;

    const auto f_path = (v.size() == 2) ? v[1] : v[0];

    // Read header_size (8 bytes) + header JSON
    std::ifstream st_file(f_path, std::ios::in | std::ios::binary);
    NNTR_THROW_IF(!st_file.is_open(), std::runtime_error)
      << "Cannot open safetensors file: " << f_path;

    uint64_t header_size = 0;
    st_file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
    NNTR_THROW_IF(!st_file, std::runtime_error)
      << "Failed to read safetensors header length from: " << f_path;

    std::string header_json(header_size, '\0');
    st_file.read(header_json.data(), static_cast<std::streamsize>(header_size));
    NNTR_THROW_IF(!st_file, std::runtime_error)
      << "Failed to read safetensors header from: " << f_path;
    st_file.close();

    // data_base: byte offset in file where the data section starts
    const size_t data_base =
      sizeof(uint64_t) + static_cast<size_t>(header_size);

    // Parse header: name -> (offset_start, size_in_bytes)
    auto name_offset_map = safetensors::parseHeader(header_json);

    // Assign file offsets to each weight by name
    std::unordered_set<const Tensor *> visited_st;
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      auto weights = (*iter)->getRunContext().getWeights();
      for (auto weight : weights) {
        if (!visited_st.insert(&weight->getVariableRef()).second)
          continue;
        const std::string &name = weight->getName();
        auto it = name_offset_map.find(name);
        if (it == name_offset_map.end())
          continue;
        const size_t file_off = data_base + it->second.first;
        weight->getVariableRef().setFileOffset(file_off);
      }
    }

    if (exec_mode == ml::train::ExecutionMode::INFERENCE) {
      model_file_fd = ::open(f_path.c_str(), O_RDONLY);
      NNTR_THROW_IF((model_file_fd == -1), std::invalid_argument)
        << "Cannot open safetensors file: " << f_path;

      std::vector<std::thread> threads;
      threads.reserve(model_graph.size());
      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           ++iter) {
        auto node = *iter;
        threads.emplace_back([&, node]() {
          if (!MMAP_READ) {
            auto local_file = checkedOpenStream<std::ifstream>(
              f_path, std::ios::in | std::ios::binary);
            node->read(local_file, false, exec_mode, fsu_mode,
                       std::numeric_limits<size_t>::max(), true, model_file_fd);
          } else {
#if defined(_WIN32)
            HANDLE hFile =
              CreateFileA(f_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL,
                          OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
            NNTR_THROW_IF((hFile == INVALID_HANDLE_VALUE), std::runtime_error)
              << "CreateFileA failed for safetensors file: " << f_path;

            HANDLE hMap =
              CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
            NNTR_THROW_IF((hMap == NULL), std::runtime_error)
              << "CreateFileMapping failed for safetensors file: " << f_path;

            char *view =
              static_cast<char *>(MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0));
            NNTR_THROW_IF((view == nullptr), std::runtime_error)
              << "MapViewOfFile failed for safetensors file: " << f_path;

            node->read(view, false, exec_mode, fsu_mode,
                       std::numeric_limits<size_t>::max(), true, model_file_fd);

            UnmapViewOfFile(view);
            CloseHandle(hMap);
            CloseHandle(hFile);
#else
            int fd = ::open(f_path.c_str(), O_RDONLY);
            NNTR_THROW_IF((fd == -1), std::invalid_argument)
              << "Cannot open safetensors file: " << f_path;

            struct stat st {};
            NNTR_THROW_IF((::fstat(fd, &st) == -1), std::invalid_argument)
              << "Cannot stat safetensors file: " << f_path;

            const size_t f_size = static_cast<size_t>(st.st_size);
            void *mmap_ptr =
              ::mmap(nullptr, f_size, PROT_READ, MAP_PRIVATE, fd, 0);
            ::close(fd);
            NNTR_THROW_IF((mmap_ptr == MAP_FAILED), std::runtime_error)
              << "mmap failed for safetensors file: " << f_path;

            (void)::posix_madvise(mmap_ptr, f_size, POSIX_MADV_RANDOM);

            char *view = static_cast<char *>(mmap_ptr);
            node->read(view, false, exec_mode, fsu_mode,
                       std::numeric_limits<size_t>::max(), true, model_file_fd);

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
      // TRAINING mode: sequential read
      std::ifstream st_in(f_path, std::ios::in | std::ios::binary);
      NNTR_THROW_IF(!st_in.is_open(), std::runtime_error)
        << "Cannot open safetensors file for training load: " << f_path;
      for (auto iter = model_graph.cbegin(); iter != model_graph.cend();
           ++iter) {
        (*iter)->read(st_in, false, exec_mode, fsu_mode);
      }
    }

    ml_logi("read safetensors model file: %s", f_path.c_str());
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
    input_tensors.emplace_back(
      MAKE_SHARED_TENSOR(mapExternalTensor(input[idx], in_dim[idx])));
  }

  if (!label.empty()) {
    sharedConstTensors label_tensors;
    auto label_dim = getOutputDimension();
    label_tensors.reserve(label.size());
    for (unsigned int idx = 0; idx < label_dim.size(); idx++) {
      label_dim[idx].batch(batch_size);
      label_tensors.emplace_back(
        MAKE_SHARED_TENSOR(mapExternalTensor(label[idx], label_dim[idx])));
    }
    output_tensors = inference(input_tensors, label_tensors, false);
  } else {
    output_tensors = inference(input_tensors, false);
  }

  std::vector<float *> output;
  output.reserve(output_tensors.size());

  for (auto &out : output_tensors) {
    auto out_t = *out.get();
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // The caller reads the GPU-produced UVM model output on the host; sync first
    // in async mode (no-op in default sync mode).
    nntrainer::cuda::StreamManager::Global().finishIfAsync();
#endif
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

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
  // CUDA-graph capture of a whole DECODE forward (NNTR_CUDA_GRAPH, M1). A decode
  // step issues ~1000 tiny kernels; the CPU launch/dispatch between them is the
  // decode bottleneck (GPU ~30-47% utilized). Capturing the per-token forward
  // into one graph and replaying it collapses that launch overhead. M1
  // re-instantiates every step (still pays cudaGraphInstantiate) purely to prove
  // capture+replay COHERENCE; M2 will cache the graphExec and patch params.
  // Prerequisite: every decode op stays on the backend stream with no host op
  // reading device memory mid-chain -- provided by the NNTR_CUDA_* device-
  // residency flags + pinned embedding staging. Drains inside the captured
  // region are suppressed by StreamManager's capturing_ guard; the single sync
  // below makes the output valid before the caller (sampling) reads it.
  static const char *_cgraph_env = std::getenv("NNTR_CUDA_GRAPH");
  static const bool cuda_graph_decode =
    _cgraph_env != nullptr && _cgraph_env[0] == '1';
  // PREFILL graph (W3): capture the M>1 prefill forward like decode, collapsing
  // the ~190 per-op cudaStreamSynchronize drains (the cMA=0 sync floor) into one
  // submission. Default ON for INTEGRATED GPUs (Orin) when the graph path is
  // enabled; discrete GPUs (RTX) keep their existing eager-async prefill (they
  // are not sync-bound -- isIntegrated()==false). Override: NNTR_CUDA_PREFILL_GRAPH
  // =1/0. A capture abort (e.g. an in-capture cudaMalloc on un-pre-grown scratch)
  // falls back to eager, which also grows the scratch so the NEXT prefill captures.
  static const bool cuda_graph_prefill = []() {
    const char *e = std::getenv("NNTR_CUDA_PREFILL_GRAPH");
    if (e != nullptr)
      return e[0] != '0';
    const char *g = std::getenv("NNTR_CUDA_GRAPH");
    return g != nullptr && g[0] == '1' &&
           nntrainer::cuda::ContextManager::Global().isIntegrated();
  }();
  static const bool cuda_graph_dbg =
    std::getenv("NNTR_CUDA_GRAPH_DBG") != nullptr;
  // Diagnostic: cache the exec from the first captured token and RE-LAUNCH it for
  // subsequent tokens (no re-capture / re-instantiate). Output is INCOHERENT
  // (per-token params are stale) -- this exists ONLY to measure the pure
  // cudaGraphLaunch+sync ceiling that a real cross-token M2-B would hit, free of
  // the per-token graph-upload that contaminates the M1 replay number.
  static const bool cuda_graph_replay =
    std::getenv("NNTR_CUDA_GRAPH_REPLAY") != nullptr;
  static cudaGraphExec_t _cg_cached_exec = nullptr;
  static sharedConstTensors _cg_cached_out;
  static unsigned long _cg_ok = 0, _cg_fallback = 0;
  bool cuda_graph_captured = false;

  // M2-B: single-capture COHERENT decode. Capture the full forward ONCE (first
  // decode token); for every later token, refresh ONLY the embeddings on the host
  // (g_m2b_skip_all feed pass) so the pinned staging holds the new token's rows,
  // update the device position (cuda_set_pos -> d_pos), and REPLAY the cached
  // graph -- skipping the ~350-op C++ dispatch (the decode bottleneck). Every
  // per-token RoPE/attention/KV-write position is read from d_pos by the
  // NNTR_CUDA_M2B kernels, so the frozen graph stays correct across tokens.
  static const bool cuda_m2b = std::getenv("NNTR_CUDA_M2B") != nullptr;
  if (cuda_m2b && from == 0 && _cg_cached_exec != nullptr) {
    // new sequence (prefill boundary): drop the previous sequence's cached graph.
    cudaGraphExecDestroy(_cg_cached_exec);
    _cg_cached_exec = nullptr;
    _cg_cached_out = {};
  }
  if (cuda_m2b && from != 0 && (to - from) == 1) {
    auto &sm = nntrainer::cuda::StreamManager::Global();
    if (_cg_cached_exec != nullptr) {
      // subsequent token: embed-only feed (refresh emb_stage) -> set pos -> replay
      static const bool m2b_light =
        std::getenv("NNTR_CUDA_M2B_LIGHT") != nullptr;
      if (m2b_light) {
        // lighter feed: set the new token input + run ONLY the two embedding
        // nodes directly, bypassing the full ~350-node graph iteration.
        model_graph.setInputsLabels(X, label);
        auto emb0 = model_graph.getLayerNode("embedding0");
        auto ple = model_graph.getLayerNode("per_layer_input_embedding");
        if (emb0)
          emb0->incremental_forwarding(from, to, false);
        if (ple)
          ple->incremental_forwarding(from, to, false);
      } else {
        g_m2b_skip_all = true;
        out = incremental_forwarding(from, to, X, label, false);
        g_m2b_skip_all = false;
      }
      nntrainer::cuda::cuda_set_pos((int)from, (int)from + 1);
      cudaGraphLaunch(_cg_cached_exec, sm.GetStream());
      cudaStreamSynchronize(sm.GetStream());
      out = _cg_cached_out;
      cuda_graph_captured = true;
    } else if (sm.beginCapture()) {
      // first decode token: set pos, capture the full forward, cache the exec
      nntrainer::cuda::cuda_set_pos((int)from, (int)from + 1);
      out = incremental_forwarding(from, to, X, label, false);
      cudaGraph_t graph = nullptr;
      if (sm.endCapture(&graph) && graph != nullptr) {
        if (cudaGraphInstantiate(&_cg_cached_exec, graph, 0) == cudaSuccess) {
          cudaGraphLaunch(_cg_cached_exec, sm.GetStream());
          cudaStreamSynchronize(sm.GetStream());
          _cg_cached_out = out;
          cuda_graph_captured = true;
        }
        cudaGraphDestroy(graph);
      } else {
        cudaGetLastError();
      }
    }
    if (cuda_graph_dbg) {
      static unsigned long _m2b_tok = 0;
      if (++_m2b_tok <= 16)
        std::fprintf(stderr, "[M2B] tok#%lu %s (exec=%p)\n", _m2b_tok,
                     cuda_graph_captured ? "ok" : "FALLBACK",
                     (void *)_cg_cached_exec);
    }
  }

  if (!cuda_graph_captured && cuda_graph_decode && from != 0 &&
      (to - from) == 1) {
    auto &sm = nntrainer::cuda::StreamManager::Global();
    const char *stage = "beginCapture";
    cudaError_t cerr = cudaSuccess;
    using _clk = std::chrono::high_resolution_clock;
    auto _us = [](_clk::time_point a, _clk::time_point b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    long t_rec = 0, t_inst = 0, t_rep = 0;
    if (cuda_graph_replay && _cg_cached_exec != nullptr) {
      // replay-only: relaunch the cached exec (timing ceiling, incoherent)
      auto p2 = _clk::now();
      cudaGraphLaunch(_cg_cached_exec, sm.GetStream());
      cudaStreamSynchronize(sm.GetStream());
      t_rep = _us(p2, _clk::now());
      out = _cg_cached_out; // persistent output tensors, refilled by the replay
      cuda_graph_captured = true;
    } else if (sm.beginCapture()) {
      auto p0 = _clk::now();
      out = incremental_forwarding(from, to, X, label, false);
      cudaGraph_t graph = nullptr;
      bool ended = sm.endCapture(&graph);
      auto p1 = _clk::now();
      t_rec = _us(p0, p1);
      if (ended && graph != nullptr) {
        cudaGraphExec_t exec = nullptr;
        cerr = cudaGraphInstantiate(&exec, graph, 0);
        auto p2 = _clk::now();
        t_inst = _us(p1, p2);
        if (cerr == cudaSuccess) {
          cudaGraphLaunch(exec, sm.GetStream());
          cudaStreamSynchronize(sm.GetStream());
          t_rep = _us(p2, _clk::now());
          if (cuda_graph_replay) {
            _cg_cached_exec = exec; // keep for replay-only relaunch
            _cg_cached_out = out;
          } else {
            cudaGraphExecDestroy(exec);
          }
          cuda_graph_captured = true;
        } else {
          stage = "cudaGraphInstantiate";
        }
        cudaGraphDestroy(graph);
      } else {
        // capture invalidated (e.g. a mid-capture cudaMalloc): record the error
        // and clear the sticky flag so the eager fallback is not falsely flagged.
        stage = "endCapture";
        cerr = cudaGetLastError();
      }
    }
    if (cuda_graph_captured)
      ++_cg_ok;
    else
      ++_cg_fallback;
    if (cuda_graph_dbg && (_cg_ok + _cg_fallback) <= 12) {
      if (cuda_graph_captured)
        std::fprintf(stderr,
                     "[CUDA_GRAPH] tok#%lu %s  record=%ldus instantiate=%ldus "
                     "replay+sync=%ldus\n",
                     _cg_ok, t_rec ? "CAPTURED+REPLAYED" : "REPLAY-ONLY(cached)",
                     t_rec, t_inst, t_rep);
      else
        std::fprintf(stderr,
                     "[CUDA_GRAPH] fell back (captured=%lu fallback=%lu) stage=%s err=%d\n",
                     _cg_ok, _cg_fallback, stage, (int)cerr);
    }
  }
  // PREFILL graph capture (W3): same machinery as the decode M1 branch above,
  // for the M>1 prefill (from==0). One beginCapture -> forward -> endCapture ->
  // instantiate -> launch -> single sync, replacing the ~190 per-op drains. The
  // StreamManager capturing_ guard suppresses the in-forward syncs; an in-capture
  // cudaMalloc (un-pre-grown scratch) invalidates the graph -> clean eager
  // fallback below (which also grows the scratch so the next prefill captures).
  if (!cuda_graph_captured && cuda_graph_prefill && !prefill_capture_disabled_ &&
      from == 0 && (to - from) > 1) {
    auto &sm = nntrainer::cuda::StreamManager::Global();
    using _clk = std::chrono::high_resolution_clock;
    auto _us = [](_clk::time_point a, _clk::time_point b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    long t_rec = 0, t_inst = 0, t_rep = 0;
    const char *stage = "beginCapture";
    cudaError_t cerr = cudaSuccess;
    if (sm.beginCapture()) {
      auto p0 = _clk::now();
      out = incremental_forwarding(from, to, X, label, false);
      cudaGraph_t graph = nullptr;
      bool ended = sm.endCapture(&graph);
      auto p1 = _clk::now();
      t_rec = _us(p0, p1);
      if (ended && graph != nullptr) {
        cudaGraphExec_t exec = nullptr;
        cerr = cudaGraphInstantiate(&exec, graph, 0);
        auto p2 = _clk::now();
        t_inst = _us(p1, p2);
        if (cerr == cudaSuccess) {
          cudaGraphLaunch(exec, sm.GetStream());
          cudaStreamSynchronize(sm.GetStream());
          t_rep = _us(p2, _clk::now());
          cudaGraphExecDestroy(exec);
          cuda_graph_captured = true;
        } else {
          stage = "cudaGraphInstantiate";
        }
        cudaGraphDestroy(graph);
      } else {
        stage = "endCapture";
        cerr = cudaGetLastError();
      }
    }
    if (cuda_graph_dbg) {
      static unsigned long _pf = 0;
      std::fprintf(stderr,
                   "[PREFILL_GRAPH] #%lu M=%u %s record=%ldus instantiate=%ldus "
                   "replay+sync=%ldus stage=%s err=%d\n",
                   ++_pf, (unsigned)(to - from),
                   cuda_graph_captured ? "CAPTURED" : "FALLBACK", t_rec, t_inst,
                   t_rep, stage, (int)cerr);
    }
  }
  if (!cuda_graph_captured)
    out = incremental_forwarding(from, to, X, label, false);
#else
  out = incremental_forwarding(from, to, X, label, false);
#endif

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
    input_tensors.emplace_back(
      MAKE_SHARED_TENSOR(mapExternalTensor(input[idx], in_dim[idx])));
  }

  // auto start_increment = std::chrono::high_resolution_clock::now();
  if (!label.empty()) {
    sharedConstTensors label_tensors;
    auto label_dim = getOutputDimension();
    label_tensors.reserve(label.size());
    for (unsigned int idx = 0; idx < label_dim.size(); idx++) {
      label_dim[idx].batch(batch_size);
      label_tensors.emplace_back(
        MAKE_SHARED_TENSOR(mapExternalTensor(label[idx], label_dim[idx])));
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

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
    // The host reads the GPU-produced UVM model output below (scopy_fp16_to_fp32
    // / memcpy); sync first in async mode (no-op in default sync mode).
    nntrainer::cuda::StreamManager::Global().finishIfAsync();
#endif
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

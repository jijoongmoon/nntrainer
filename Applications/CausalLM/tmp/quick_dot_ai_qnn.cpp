// SPDX-License-Identifier: Apache-2.0
/**
 * @file   quick_dot_ai_qnn.cpp
 * @brief  QNN model implementation for Quick.AI template
 * @note   This file implements a layer that executes QNN binary file within
 *         transformer.h architecture.
 */

#include "quick_dot_ai_qnn.h"
#include "android_memory_allocator.h"
#include "engine.h"
#include "graph_parser.h"

#include <fstream>
#include <sstream>

using namespace ml::train;
using namespace nntrainer;
using namespace causallm;

namespace {

bool is_absolute_path(const std::string &path) {
  return !path.empty() && path[0] == '/';
}

std::string dirname(const std::string &path) {
  auto pos = path.find_last_of('/');
  if (pos == std::string::npos) return "";
  return path.substr(0, pos);
}

std::string rebase_relative_to_model_file(const std::string &path,
                                          const std::string &model_file) {
  if (path.empty() || is_absolute_path(path)) return path;
  auto base_dir = dirname(model_file);
  if (base_dir.empty()) return path;
  return base_dir + "/" + path;
}

} // namespace

std::string qnn_to_nntrainer_datatype(std::string qnn_dtype) {
  if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_16") return "UINT16";
  if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_8")  return "UINT8";
  if (qnn_dtype == "QNN_DATATYPE_FLOAT_16")        return "FP16";
  throw std::invalid_argument("qnn_dtype is " + qnn_dtype);
}

ml::train::TensorDim::IO_TensorType
get_qnn_input_data(TensorInfo tensor_object, std::set<void *> &allocated_ptrs) {
  int size = GraphParser::get_tensor_size(tensor_object);
  std::string qnn_dtype = tensor_object.data_type;

  if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_16" ||
      qnn_dtype == "QNN_DATATYPE_FLOAT_16") {
    auto *ptr = (uint16_t *)allocate(size);
    allocated_ptrs.insert(ptr);
    return ptr;
  } else if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_8") {
    auto *ptr = (uint8_t *)allocate(size);
    allocated_ptrs.insert(ptr);
    return ptr;
  }
  throw std::invalid_argument("qnn_dtype is " + qnn_dtype);
}

void *causallm::Quick_Dot_AI_QNN::tracked_allocate(size_t size) {
  void *ptr = allocate(size);
  allocated_ptrs_.insert(ptr);
  return ptr;
}

void causallm::Quick_Dot_AI_QNN::deallocate_all() {
  LOGD("Quick_Dot_AI_QNN::deallocate_all: freeing %zu tracked pointers",
       allocated_ptrs_.size());
  for (auto *ptr : allocated_ptrs_) {
    deallocate(ptr);
  }
  allocated_ptrs_.clear();
}

causallm::Quick_Dot_AI_QNN::~Quick_Dot_AI_QNN() {
  for (auto &[model_name, model] : models) {
    model.model_handle.reset();
  }
  deallocate_all();
}

void causallm::Quick_Dot_AI_QNN::initialize() {
  int status;

  auto &ct_engine = nntrainer::Engine::Global();
  LOGD("qnn_engine registering .... ");
  NNTR_THROW_IF(ct_engine.registerContext("libqnn_context.so", ""),
                std::runtime_error)
      << "Fail to register QNN Context";
  LOGD("qnn_engine registering done ");

  GraphParser graph_parser = GraphParser();
  auto graphs_info = graph_parser.parseJsonFile(binary_config_path);

  for (const auto &graph_name : graphs_to_use) {
    auto current_model = createModel(ml::train::ModelType::NEURAL_NET);
    std::string out_dim, out_data_format, out_tensor_format;
    std::string input_names, in_quant, out_quant;

    NNTR_THROW_IF(graphs_info.find(graph_name) == graphs_info.end(),
                  std::runtime_error)
        << graph_name << " does not exist in model binary config "
        << binary_config_path << "!";

    auto &current_graphs_info = graphs_info[graph_name];
    std::vector<ml::train::TensorDim::IO_TensorType> model_inputs;

    for (const auto &[tensor_name, tensor_object] :
         current_graphs_info.raw_inputs) {
      if (uses_embedding &&
          (tensor_name == "inputs_embeds" || tensor_name == "input_embeds")) {
        auto input_shape = tensor_object.dimensions;
        int input_size = input_shape[0];
        std::string input_shape_string = std::to_string(input_shape[0]);
        for (int i = 1; i < (int)input_shape.size() - 1; i++) {
          input_shape_string += ":";
          input_shape_string += std::to_string(input_shape[i]);
          input_size *= input_shape[i];
        }

        // Embedding layer properties. When `embedding_file_name` is set
        // we hand the path to the layer via `quantized_lut_path`. The
        // CausalLM EmbeddingLayer auto-detects the format from the
        // extension:
        //   *.json  → 4-bit packed manifest (dequant + requant)
        //   else    → raw UINT16 bin (memcpy, no requant)
        // For the 4-bit path we also pass the consumer's quant params
        // (this graph's input `scale`/`offset` for input_embeds) so the
        // layer can requant into QNN's UINT16 codes; the raw-uint16 path
        // ignores them.
        std::vector<std::string> emb_props = {
            withKey("name", tensor_name),
            withKey("in_dim", vocab_size),
            withKey("input_shape", input_shape_string),
            withKey("out_dim", input_shape.back()),
        };
        if (!embedding_file_name.empty()) {
          emb_props.push_back(
              withKey("quantized_lut_path", embedding_file_name));
          // Pass consumer (graph input) quant params; the layer ignores
          // them in raw-uint16 mode and uses them in 4-bit mode.
          emb_props.push_back(
              withKey("output_quant_scale",
                      std::to_string(tensor_object.scale)));
          emb_props.push_back(
              withKey("output_quant_offset",
                      std::to_string(tensor_object.offset)));
        }
        current_model->addLayer(createLayer("embedding", emb_props));

        model_inputs.push_back(
            (float *)tracked_allocate(sizeof(float) * input_size));
      } else {
        auto input_shape = tensor_object.dimensions;
        std::string input_shape_string = std::to_string(input_shape[0]);
        for (int i = 1; i < (int)input_shape.size(); i++) {
          input_shape_string += ":";
          input_shape_string += std::to_string(input_shape[i]);
        }
        current_model->addLayer(createLayer(
            "input", {withKey("name", tensor_name),
                      withKey("input_shape", input_shape_string)}));
        model_inputs.push_back(
            get_qnn_input_data(tensor_object, allocated_ptrs_));
      }

      if (!input_names.empty()) input_names += ", ";
      input_names += tensor_name;

      if (!in_quant.empty()) in_quant += ",";
      in_quant += tensor_name + ":" +
                  std::to_string(tensor_object.scale) + ":" +
                  std::to_string(tensor_object.offset);
    }

    for (const auto &[tensor_name, tensor_object] :
         current_graphs_info.raw_outputs) {
      if (!out_dim.empty()) out_dim += ",";
      out_dim += std::to_string(tensor_object.dimensions[0]);
      for (int i = 1; i < (int)tensor_object.dimensions.size(); i++) {
        out_dim += ":";
        out_dim += std::to_string(tensor_object.dimensions[i]);
      }
      if (!out_data_format.empty()) out_data_format += ",";
      out_data_format += qnn_to_nntrainer_datatype(tensor_object.data_type);
      if (!out_tensor_format.empty()) out_tensor_format += ",";
      out_tensor_format += "OUT_TENSOR";
      if (!out_quant.empty()) out_quant += ",";
      out_quant += tensor_name + ":" +
                   std::to_string(tensor_object.scale) + ":" +
                   std::to_string(tensor_object.offset);
    }

    LayerHandle qnn_layer = createLayer(
        "qnn_graph",
        {withKey("name", graph_name), withKey("path", model_file_name),
         withKey("dim", out_dim), withKey("tensor_dtype", out_data_format),
         withKey("tensor_type", out_tensor_format),
         withKey("input_layers", input_names),
         withKey("input_quant_param", in_quant),
         withKey("output_quant_param", out_quant), withKey("engine", "qnn")});
    current_model->addLayer(qnn_layer);

    current_model->setProperty({withKey("batch_size", 1), withKey("epochs", 1),
                                withKey("model_tensor_type", "UINT16-UINT16")});

    auto optimizer = createOptimizer("sgd", {withKey("learning_rate", 0.001)});
    current_model->setOptimizer(std::move(optimizer));

    status = current_model->compile(ExecutionMode::INFERENCE);
    if (status) throw std::invalid_argument("Model compilation failed!");

    status = current_model->initialize(ExecutionMode::INFERENCE);
    if (status) throw std::invalid_argument("Model initialization failed!");

    models[graph_name] = {current_graphs_info, std::move(current_model),
                          model_inputs};
  }
}

void causallm::Quick_Dot_AI_QNN::load_weight(const std::string &weight_path) {
  // QNN binaries are loaded for every graph.
  for (const auto &[key, value] : models) {
    value.model_handle->load(model_file_name, ModelFormat::MODEL_FORMAT_QNN);
  }

  // Embedding source is wired into the EmbeddingLayer via the
  // `quantized_lut_path` property at initialize() time. The layer does
  // its own loading (4-bit manifest OR raw uint16 bin, auto-detected
  // by extension) and shares a single in-memory copy across both
  // graphs via a path-keyed weak cache. So nothing additional is
  // needed here for the embedding case.
  if (uses_embedding && !embedding_file_name.empty()) {
    LOGD("Embedding source wired via EmbeddingLayer property: %s",
         embedding_file_name.c_str());
  }

  for (auto &[key, value] : models) {
    value.model_handle->allocate(ExecutionMode::INFERENCE);
  }
  std::cout << "load weight done" << std::endl;
}

void causallm::Quick_Dot_AI_QNN::save_weight(const std::string &) {}

void causallm::Quick_Dot_AI_QNN::setupParameters(json &cfg,
                                                 json &generation_cfg,
                                                 json &nntr_cfg) {
  LOGD("----------------in Quick_Dot_AI_QNN : setupParameters");
  model_file_name      = nntr_cfg["model_file_name"].get<std::string>();
  binary_config_path   = nntr_cfg["binary_config_path"].get<std::string>();
  binary_config_path   =
      rebase_relative_to_model_file(binary_config_path, model_file_name);
  graphs_to_use        = nntr_cfg["graphs_to_use"].get<std::vector<std::string>>();
  vocab_size           = cfg["vocab_size"].get<int>();

  if (nntr_cfg.contains("uses_embedding"))
    uses_embedding = nntr_cfg["uses_embedding"].get<bool>();

  if (nntr_cfg.contains("embedding_file_name")) {
    embedding_file_name = nntr_cfg["embedding_file_name"].get<std::string>();
    embedding_file_name =
        rebase_relative_to_model_file(embedding_file_name, model_file_name);
    LOGD("---------------- embedding_file_name : %s",
         embedding_file_name.c_str());
  }
}

void causallm::Quick_Dot_AI_QNN::constructModel() {}

std::vector<LayerHandle>
causallm::Quick_Dot_AI_QNN::createTransformerDecoderBlock(const int, std::string) {
  return {};
}

std::vector<LayerHandle> causallm::Quick_Dot_AI_QNN::createAttention(
    const int, int, int, int, std::string, std::string, std::string) {
  return {};
}

std::vector<LayerHandle>
causallm::Quick_Dot_AI_QNN::createMlp(const int, int, int, std::string) {
  return {};
}

void causallm::Quick_Dot_AI_QNN::registerCustomLayers() {}

void causallm::Quick_Dot_AI_QNN::quantize_uint16_memcpy(float *src,
                                                        uint16_t *dest,
                                                        int count, float scale,
                                                        int offset) {
  for (int i = 0; i < count; i++) {
    if (std::isfinite(src[i])) {
      int q = src[i] / scale - offset;
      if (q > 65535) q = 65535;
      if (q < 0) q = 0;
      dest[i] = q;
    } else {
      dest[i] = 0;
    }
  }
}

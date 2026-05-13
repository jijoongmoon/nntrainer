// SPDX-License-Identifier: Apache-2.0
/**
 * @file   quick_dot_ai_qnn.cpp
 * @brief  QNN model implementation for Quick.AI template
 * @note   This file implements a layer that executes QNN binary file within
 * transformer.h architecture.
 */

#include "quick_dot_ai_qnn.h"
#include "android_memory_allocator.h"
#include "engine.h"
#include "graph_parser.h"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

using namespace ml::train;
using namespace nntrainer;
using namespace causallm;

namespace
{

bool
is_absolute_path (const std::string &path)
{
  return !path.empty () && path[0] == '/';
}

std::string
dirname (const std::string &path)
{
  auto pos = path.find_last_of ('/');
  if (pos == std::string::npos) {
    return "";
  }
  return path.substr (0, pos);
}

std::string
rebase_relative_to_model_file (const std::string &path, const std::string &model_file)
{
  if (path.empty () || is_absolute_path (path)) {
    return path;
  }

  auto base_dir = dirname (model_file);
  if (base_dir.empty ()) {
    return path;
  }
  return base_dir + "/" + path;
}

// Format a scale value with enough precision for the float that QNN
// will eventually use (`Qnn_QuantizeParams_t::scaleOffsetEncoding::scale`
// is float, parsed with std::stof on the property side). TensorInfo
// stores scale as double, so we round-trip via float — std::to_string
// silently caps at 6 fractional digits and mangles tiny QNN scales
// (e.g. 0.0004169851... → "0.000417"). Across 70+ tensors × 35 layers
// × every token the dequant drift compounds into representation
// collapse after a few dozen tokens.
inline std::string format_float_precise (double v)
{
  std::ostringstream os;
  os << std::setprecision (std::numeric_limits<float>::max_digits10)
     << static_cast<float> (v);
  return os.str ();
}

} // namespace

/**
 * @brief Dequantize 4-bit packed values to UINT16
 * @param packed_data Pointer to packed 4-bit data (2 values per byte)
 * @param size_in_bytes Size of packed data in bytes
 * @param scale Scale factor for dequantization
 * @param offset Offset for dequantization
 * @return Vector of dequantized UINT16 values
 */
std::vector<uint16_t>
dequantize_4bit_packed (
    const uint8_t *packed_data, size_t size_in_bytes, float scale, int offset)
{
  std::vector<uint16_t> result;
  result.reserve (size_in_bytes * 2); // 2 values per byte

  for (size_t i = 0; i < size_in_bytes; ++i) {
    uint8_t byte = packed_data[i];

    // Extract lower 4 bits (first value)
    uint8_t lower_nibble = byte & 0x0F;

    // Extract upper 4 bits (second value)
    uint8_t upper_nibble = (byte >> 4) & 0x0F;

    // Dequantize to float and convert to UINT16
    float lower_float = (static_cast<float> (lower_nibble) + offset) * scale;
    float upper_float = (static_cast<float> (upper_nibble) + offset) * scale;

    // Clamp to UINT16 range and convert
    uint16_t lower_uint16
        = static_cast<uint16_t> (std::max (0.0f, std::min (65535.0f, lower_float)));
    uint16_t upper_uint16
        = static_cast<uint16_t> (std::max (0.0f, std::min (65535.0f, upper_float)));

    result.push_back (lower_uint16);
    result.push_back (upper_uint16);
  }

  return result;
}

/**
 * @brief Parse scale and offset from json file
 */
std::pair<float, int>
parse_scale_offset_from_json (const std::string &json_file_path)
{
  std::ifstream json_file (json_file_path);
  if (!json_file.is_open ()) {
    throw std::runtime_error ("Failed to open json file: " + json_file_path);
  }

  std::stringstream buffer;
  buffer << json_file.rdbuf ();
  std::string json_str = buffer.str ();

  // Simple parsing for "scale" and "offset"
  // This is a very basic parser, assuming the format is fixed
  float scale = 1.0f;
  int offset = 0;

  size_t scale_pos = json_str.find ("\"scale\"");
  if (scale_pos != std::string::npos) {
    size_t colon_pos = json_str.find (":", scale_pos);
    if (colon_pos != std::string::npos) {
      size_t value_start = json_str.find_first_not_of (" \t\n\r", colon_pos + 1);
      if (value_start != std::string::npos) {
        size_t value_end = json_str.find_first_of (",}", value_start);
        if (value_end != std::string::npos) {
          std::string scale_str = json_str.substr (value_start, value_end - value_start);
          scale = std::stof (scale_str);
        }
      }
    }
  }

  size_t offset_pos = json_str.find ("\"offset\"");
  if (offset_pos != std::string::npos) {
    size_t colon_pos = json_str.find (":", offset_pos);
    if (colon_pos != std::string::npos) {
      size_t value_start = json_str.find_first_not_of (" \t\n\r", colon_pos + 1);
      if (value_start != std::string::npos) {
        size_t value_end = json_str.find_first_of (",}", value_start);
        if (value_end != std::string::npos) {
          std::string offset_str = json_str.substr (value_start, value_end - value_start);
          offset = std::stoi (offset_str);
        }
      }
    }
  }

  return std::make_pair (scale, offset);
}

std::string
qnn_to_nntrainer_datatype (std::string qnn_dtype)
{
  if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_16") {
    return "UINT16";
  } else if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_8") {
    return "UINT8";
  } else if (qnn_dtype == "QNN_DATATYPE_FLOAT_16") {
    return "FP16";
  } else {
    throw std::invalid_argument ("qnn_dtype is " + qnn_dtype);
  }
}

ml::train::TensorDim::IO_TensorType
get_qnn_input_data (TensorInfo tensor_object, std::set<void *> &allocated_ptrs)
{
  int size = GraphParser::get_tensor_size (tensor_object);
  std::string qnn_dtype = tensor_object.data_type;

  if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_16" || qnn_dtype == "QNN_DATATYPE_FLOAT_16") {
    auto *ptr = (uint16_t *)allocate (size);
    allocated_ptrs.insert (ptr);
    return ptr;
  } else if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_8") {
    auto *ptr = (uint8_t *)allocate (size);
    allocated_ptrs.insert (ptr);
    return ptr;
  } else {
    throw std::invalid_argument ("qnn_dtype is " + qnn_dtype);
  }
}

void *
causallm::Quick_Dot_AI_QNN::tracked_allocate (size_t size)
{
  void *ptr = allocate (size);
  allocated_ptrs_.insert (ptr);
  return ptr;
}

void
causallm::Quick_Dot_AI_QNN::deallocate_all ()
{
  LOGD ("Quick_Dot_AI_QNN::deallocate_all: freeing %zu tracked pointers",
      allocated_ptrs_.size ());
  for (auto *ptr : allocated_ptrs_) {
    LOGD ("Quick_Dot_AI_QNN::deallocate_all: deallocating ptr=%p", ptr);
    deallocate (ptr);
  }
  allocated_ptrs_.clear ();
}

causallm::Quick_Dot_AI_QNN::~Quick_Dot_AI_QNN ()
{
  // Tear down each graph's NeuralNetwork (and the QNNGraph layer inside it)
  // FIRST, so ~QNNGraph releases its zero-copy references to our input
  // buffers before we free them. Without this, the deallocate loop below
  // would free memory that QNNGraph still tracks, and ~QNNGraph — invoked
  // later as part of `models` member destruction — would touch freed
  // memory.
  for (auto &[model_name, model] : models) {
    model.model_handle.reset ();
  }
  deallocate_all ();
}

void
causallm::Quick_Dot_AI_QNN::initialize ()
{
  int status;

  auto &ct_engine = nntrainer::Engine::Global ();
  LOGD ("qnn_engine registering .... ");

  NNTR_THROW_IF (ct_engine.registerContext ("libqnn_context.so", ""), std::runtime_error)
      << "Fail to register QNN Context";

  LOGD ("qnn_engine registering done ");
  std::cout << "--------------- " << binary_config_path << std::endl;

  Transformer::registerCustomLayers ();

  GraphParser graph_parser = GraphParser ();
  auto graphs_info = graph_parser.parseJsonFile (binary_config_path);
  for (const auto &graph_name : graphs_to_use) {
    auto current_model = createModel (ml::train::ModelType::NEURAL_NET);
    std::string out_dim;
    std::string out_data_format;
    std::string out_tensor_format;
    std::string input_names;
    std::string in_quant;
    std::string out_quant;

    NNTR_THROW_IF (graphs_info.find (graph_name) == graphs_info.end (), std::runtime_error)
        << graph_name << " does not exist in model binary config"
        << binary_config_path << "!";

    auto &current_graphs_info = graphs_info[graph_name];
    std::vector<ml::train::TensorDim::IO_TensorType> model_inputs;

    for (const auto &[tensor_name, tensor_object] : current_graphs_info.raw_inputs) {
      // if (uses_embedding &&
      //     (tensor_name == "inputs_embeds" || tensor_name == "input_embeds")) {
      // auto input_shape = tensor_object.dimensions;
      // int input_size = input_shape[0];
      // std::string input_shape_string = std::to_string(input_shape[0]);
      // for (int i = 1; i < input_shape.size() - 1; i++) {
      //   input_shape_string += ":";
      //   input_shape_string += std::to_string(input_shape[i]);
      //   input_size *= input_shape[i];
      // }
      // current_model->addLayer(createLayer(
      //     "embedding",
      //     {withKey("name", tensor_name), withKey("in_dim", vocab_size),
      //      withKey("input_shape", input_shape_string),
      //      withKey("out_dim", input_shape.back())}));

      // model_inputs.push_back((float *)tracked_allocate(sizeof(float) * input_size));
      if (uses_embedding && (tensor_name == "inputs_embeds" || tensor_name == "input_embeds")) {
        auto input_shape = tensor_object.dimensions;
        int input_size = input_shape[0];
        std::string input_shape_string = std::to_string (input_shape[0]);
        for (int i = 1; i < (int)input_shape.size () - 1; i++) {
          input_shape_string += ":";
          input_shape_string += std::to_string (input_shape[i]);
          input_size *= input_shape[i];
        }
	        // ── Debug: dump inputs_embeds tensor quant params ──
        std::cout << "[EMB-IN-DBG] graph=" << graph_name
                  << " name=" << tensor_name
                  << " dtype=" << tensor_object.data_type
                  << " scale=" << format_float_precise(tensor_object.scale)
                  << " offset=" << tensor_object.offset
                  << " shape=" << input_shape_string
                  << ":" << input_shape.back()
                  << std::endl;

        // Use the CausalLM tensorwise-4bit-aware embedding layer.
        // When `embedding_file_name` points at a JSON manifest with a
        // 4-bit packed LUT, the layer loads the table once via a
        // path-keyed shared cache so peer graphs share one in-memory copy.
        std::vector<std::string> emb_props = {
          withKey ("name", tensor_name),
          withKey ("in_dim", vocab_size),
          withKey ("input_shape", input_shape_string),
          withKey ("out_dim", input_shape.back ()),
        };
        if (uses_embedding && !embedding_file_name.empty ()) {
          emb_props.push_back (withKey ("quantized_lut_path", embedding_file_name));
          // Round-trip-precise float string so the layer's requant uses
          // the exact same scale QNN sees on the input_embeds tensor.
          emb_props.push_back (withKey (
              "output_quant_scale", format_float_precise (tensor_object.scale)));
          emb_props.push_back (
              withKey ("output_quant_offset", std::to_string (tensor_object.offset)));
        }

        current_model->addLayer (createLayer ("embedding_layer", emb_props));
        model_inputs.push_back ((float *)tracked_allocate (sizeof (float) * input_size));
      } else {
        auto input_shape = tensor_object.dimensions;
        std::string input_shape_string = std::to_string (input_shape[0]);

        for (int i = 1; i < input_shape.size (); i++) {
          input_shape_string += ":";
          input_shape_string += std::to_string (input_shape[i]);
        }
        std::cout << tensor_name << " : " << input_shape_string << std::endl;
        current_model->addLayer (createLayer (
            "input", { withKey ("name", tensor_name),
                         //  withKey("input_dtype",
                         //  qnn_to_nntrainer_datatype(tensor_object.data_type)),
                         withKey ("input_shape", input_shape_string) }));
        model_inputs.push_back (get_qnn_input_data (tensor_object, allocated_ptrs_));
      }

      if (!input_names.empty ()) {
        input_names += ", ";
      }
      input_names += tensor_name;

      if (!in_quant.empty ()) {
        in_quant += ",";
      }
      in_quant += tensor_name;
      in_quant += ":";
      in_quant += format_float_precise (tensor_object.scale);
      in_quant += ":";
      in_quant += std::to_string (tensor_object.offset);
    }

    for (const auto &[tensor_name, tensor_object] : current_graphs_info.raw_outputs) {
      if (!out_dim.empty ()) {
        out_dim += ",";
      }
      out_dim += std::to_string (tensor_object.dimensions[0]);
      for (int i = 1; i < tensor_object.dimensions.size (); i++) {
        out_dim += ":";
        out_dim += std::to_string (tensor_object.dimensions[i]);
      }

      if (!out_data_format.empty ()) {
        out_data_format += ",";
      }
      out_data_format += qnn_to_nntrainer_datatype (tensor_object.data_type);

      if (!out_tensor_format.empty ()) {
        out_tensor_format += ",";
      }
      out_tensor_format += "OUT_TENSOR";

      if (!out_quant.empty ()) {
        out_quant += ",";
      }
      out_quant += tensor_name;
      out_quant += ":";
      out_quant += format_float_precise (tensor_object.scale);
      out_quant += ":";
      out_quant += std::to_string (tensor_object.offset);
    }

    std::cout << "graph_name -------------   : " << graph_name << std::endl;

    LayerHandle qnn_layer = createLayer ("qnn_graph",
        { withKey ("name", graph_name), withKey ("path", model_file_name),
            withKey ("dim", out_dim), withKey ("tensor_dtype", out_data_format),
            withKey ("tensor_type", out_tensor_format),
            withKey ("input_layers", input_names), withKey ("input_quant_param", in_quant),
            withKey ("output_quant_param", out_quant), withKey ("engine", "qnn") });
    current_model->addLayer (qnn_layer);
    std::cout << "end qnn graph" << std::endl;
    current_model->setProperty ({ withKey ("batch_size", 1),
        withKey ("epochs", 1), withKey ("model_tensor_type", "UINT16-UINT16") });

    auto optimizer = createOptimizer ("sgd", { withKey ("learning_rate", 0.001) });
    current_model->setOptimizer (std::move (optimizer));

    status = current_model->compile (ExecutionMode::INFERENCE);
    if (status) {
      throw std::invalid_argument ("Model compilation failed!");
    }
    std::cout << "end compile" << std::endl;
    status = current_model->initialize (ExecutionMode::INFERENCE);
    if (status) {
      throw std::invalid_argument ("Model initialization failed!");
    }
    std::cout << "end inititlize" << std::endl;
    current_model->summarize (std::cout, ml_train_summary_type_e::ML_TRAIN_SUMMARY_MODEL);

    // std::shared_ptr<ml::train::Layer>emb;
    // current_model->getLayer("input_embeds", &emb);
    // auto node = std::static_pointer_cast<nntrainer::LayerNode>(emb);
    // auto in_dt = node->getInputDimensions()[0].getDataType();
    // std::cout << "embedding input dtype = %d",(int)in_dt);

    models[graph_name] = { current_graphs_info, std::move (current_model), model_inputs };
    std::cout << "----------------------- end graph" << std::endl;
  }
}

void
causallm::Quick_Dot_AI_QNN::load_weight (const std::string &weight_path)
{
  std::cout << weight_path << " " << std::endl;
  for (const auto &[key, value] : models) {
    std::cout << model_file_name << std::endl;
    value.model_handle->load (model_file_name, ModelFormat::MODEL_FORMAT_QNN);
  }

  std::cout << "load weight done" << std::endl;
  // Allocate tensors for inference - required for input/output buffers
  for (auto &[key, value] : models) {
    value.model_handle->allocate (ExecutionMode::INFERENCE);
  }
}

void
causallm::Quick_Dot_AI_QNN::save_weight (const std::string &weight_path)
{
  // Unimplemented.
}

void
causallm::Quick_Dot_AI_QNN::setupParameters (json &cfg, json &generation_cfg, json &nntr_cfg)
{
  // Read nntr_config parameters
  LOGD ("----------------in Quick_Dot_AI_QNN : setupParameters");
  model_file_name = nntr_cfg["model_file_name"].get<std::string> ();
  LOGD ("----------------binary_config_path : %s", model_file_name.c_str ());
  binary_config_path = nntr_cfg["binary_config_path"].get<std::string> ();
  binary_config_path = rebase_relative_to_model_file (binary_config_path, model_file_name);
  LOGD ("----------------binary_config_path : %s", binary_config_path.c_str ());
  graphs_to_use = nntr_cfg["graphs_to_use"].get<std::vector<std::string>> ();
  for (auto s : graphs_to_use) {
    LOGD ("----------------graphs_to_use : %s", s.c_str ());
  }
  vocab_size = cfg["vocab_size"].get<int> ();
  LOGD ("----------------vocab size : %d", vocab_size);

  // Multimodal opt-in: when uses_embedding=false, the LLM graph's
  // input(s)_embeds tensor is fed with pre-computed uint16 embeddings
  // rather than token IDs via an embedding layer. Derived classes
  // also mmap embedding_file_name for per-token lookup during
  if (nntr_cfg.contains ("uses_embedding")) {
    uses_embedding = nntr_cfg["uses_embedding"].get<bool> ();
  }
  LOGD ("---------------- uses_embedding : %d", uses_embedding);

  if (nntr_cfg.contains ("embedding_file_name")) {
    embedding_file_name = nntr_cfg["embedding_file_name"].get<std::string> ();
    embedding_file_name
        = rebase_relative_to_model_file (embedding_file_name, model_file_name);
    LOGD ("---------------- embedding_file_name : %s", embedding_file_name.c_str ());
  }
}

void
causallm::Quick_Dot_AI_QNN::constructModel ()
{
  // Unimplemented.
}

std::vector<LayerHandle>
causallm::Quick_Dot_AI_QNN::createTransformerDecoderBlock (
    const int layer_id, std::string input_name)
{
  // Unimplemented.
  return std::vector<LayerHandle> ();
}

std::vector<LayerHandle>
causallm::Quick_Dot_AI_QNN::createAttention (const int layer_id, int seq_len, int n_heads,
    int head_dim, std::string query_name, std::string key_name, std::string value_name)
{
  // Unimplemented.
  return std::vector<LayerHandle> ();
}

std::vector<LayerHandle>
causallm::Quick_Dot_AI_QNN::createMlp (
    const int layer_id, int dim, int hidden_dim, std::string input_name)
{
  // Unimplemented.
  return std::vector<LayerHandle> ();
}

void
causallm::Quick_Dot_AI_QNN::registerCustomLayers ()
{
  // Unimplemented.
}

void
causallm::Quick_Dot_AI_QNN::quantize_uint16_memcpy (
    float *src, uint16_t *dest, int count, float scale, int offset)
{
  for (int i = 0; i < count; i++) {
    if (std::isfinite (src[i])) {
      int quantized_value = src[i] / scale - offset;
      if (quantized_value > 65535)
        quantized_value = 65535;
      if (quantized_value < 0)
        quantized_value = 0;
      dest[i] = quantized_value;
    } else {
      // Warning message?
      dest[i] = 0;
    }
  }
}

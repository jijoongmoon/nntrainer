// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jihoon Lee <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   7 August 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include "encoder.hpp"
#include <app_context.h>
#include <rms_norm.h>
#include <rotary_embedding.h>
#include <swiglu.h>
#include <transpose_layer.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

// Hyper params for LLaMA
int const DIM = 2304;
int const NUM_LAYERS = 28;
int const NUM_HEADS = 18;

int const MULTIPLE_OF = 256;

float const NORM_EPS = 0.000001;
int const NUM_VOCAB = 96000;
int MAX_SEQ_LEN = 1024;

int NUM_TO_GENERATE = 100;

unsigned int INIT_SEQ_LEN = 1024;
unsigned int batch_size = 1;
unsigned int epoch = 1;

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;
bool swap = false;

bool optimize = false;

template <typename T>
T unwrap(std::optional<T> &&value, const std::string &error_msg) {
  if (value.has_value()) {
    return value.value();
  } else {
    throw std::runtime_error(error_msg);
  }
}

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

std::vector<LayerHandle> createAttentionLayer(const int layer_id, int seq_len,
                                              int n_heads, int head_dim,
                                              std::string query_name,
                                              std::string key_name,
                                              std::string value_name) {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  if (optimize) {
    // linear transformation of q
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(
        createLayer("fully_connected",
                    {withKey("name", "layer" + std::to_string(layer_id) +
                                       "_wq_" + std::to_string(i)),
                     withKey("unit", head_dim), withKey("disable_bias", "true"),
                     withKey("input_layers", query_name)}));
    }

    // linear transformation of k
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(
        createLayer("fully_connected",
                    {withKey("name", "layer" + std::to_string(layer_id) +
                                       "_wk_" + std::to_string(i)),
                     withKey("unit", head_dim), withKey("disable_bias", "true"),
                     withKey("input_layers", key_name)}));
    }

    // linear transformation of v
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(
        createLayer("fully_connected",
                    {withKey("name", "layer" + std::to_string(layer_id) +
                                       "_wv_" + std::to_string(i)),
                     withKey("unit", head_dim), withKey("disable_bias", "true"),
                     withKey("input_layers", value_name)}));
    }

    std::string concat_input = "";
    // apply rotary embedding and dot_product attention
    for (int i = 0; i < n_heads; i++) {
      // reshape q, k, v (apply num_heads)
      layers.push_back(createLayer(
        "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                      "_q_reshape_" + std::to_string(i)),
                    withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
                    withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wq_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                      "_k_reshape_" + std::to_string(i)),
                    withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
                    withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wk_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                      "_v_reshape_" + std::to_string(i)),
                    withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
                    withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wv_" + std::to_string(i))}));

      // apply rotary embedding to q, k
      layers.push_back(createLayer(
        "rotary_embedding",
        {withKey("name", "layer" + std::to_string(layer_id) + "_q_rotary_" +
                           std::to_string(i)),
         withKey("input_layers", "layer" + std::to_string(layer_id) +
                                   "_q_reshape_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "rotary_embedding",
        {withKey("name", "layer" + std::to_string(layer_id) + "_k_rotary_" +
                           std::to_string(i)),
         withKey("input_layers", "layer" + std::to_string(layer_id) +
                                   "_k_reshape_" + std::to_string(i))}));

      // apply scaled-dot product attention
      layers.push_back(ml::train::layer::Attention(
        {"name=layer" + std::to_string(layer_id) + "_attention_" +
           std::to_string(i),
         "input_layers=layer" + std::to_string(layer_id) + "_q_rotary_" +
           std::to_string(i) + ",layer" + std::to_string(layer_id) +
           "_v_reshape_" + std::to_string(i) + ",layer" +
           std::to_string(layer_id) + "_k_rotary_" + std::to_string(i),
         "scaled_dot_product=true", "causal_mask=true"}));

      layers.push_back(createLayer(
        "reshape",
        {withKey("name", "layer" + std::to_string(layer_id) +
                           "_attention_output_" + std::to_string(i)),
         withKey("target_shape", "1:" + std::to_string(seq_len) + ":" +
                                   std::to_string(head_dim)),
         withKey("input_layers", "layer" + std::to_string(layer_id) +
                                   "_attention_" + std::to_string(i))}));

      concat_input += "layer" + std::to_string(layer_id) +
                      "_attention_output_" + std::to_string(i);

      if (i != n_heads - 1) {
        concat_input += ",";
      }
    }

    // concat attention output
    layers.push_back(createLayer(
      "concat", {withKey("name", "layer" + std::to_string(layer_id) +
                                   "_attention_concat"),
                 withKey("axis", 3), withKey("input_layers", concat_input)}));

    // reshape for flatten
    layers.push_back(createLayer(
      "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                    "_attention_flatten"),
                  withKey("target_shape", "1:" + std::to_string(seq_len) + ":" +
                                            std::to_string(n_heads * head_dim)),
                  withKey("input_layers", "layer" + std::to_string(layer_id) +
                                            "_attention_concat")}));

    // linear transformation of attention output
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
       withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_attention_flatten")}));
  } else {
    layers.push_back(createLayer(
      "multi_head_attention",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
       withKey("num_heads", std::to_string(NUM_HEADS)),
       withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
       withKey("disable_bias", "true"),
       withKey("input_layers", {query_name, key_name, value_name})}));
  }

  return layers;
}

std::vector<LayerHandle> createFeedForwardLayer(const int layer_id, int dim,
                                                int hidden_dim,
                                                std::string input_name,
                                                int multiplier = 1) {
  using ml::train::createLayer;
  std::vector<LayerHandle> layers;

  hidden_dim = 2 * multiplier * hidden_dim / 3;
  hidden_dim = MULTIPLE_OF * ((hidden_dim + MULTIPLE_OF - 1) / MULTIPLE_OF);

  layers.push_back(
    createLayer("fully_connected",
                {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_1"),
                 withKey("unit", hidden_dim), withKey("disable_bias", "true"),
                 withKey("input_layers", input_name)}));
  layers.push_back(
    createLayer("fully_connected",
                {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_2"),
                 withKey("unit", hidden_dim), withKey("disable_bias", "true"),
                 withKey("input_layers", input_name)}));

  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_1," +
                               "layer" + std::to_string(layer_id) +
                               "_ffn_2")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_output"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_swiglu")}));

  return layers;
}

std::vector<LayerHandle> createTransformerDecoder(const int layer_id,
                                                  std::string input_name) {
  using ml::train::createLayer;
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS))}));

  auto att_layer = createAttentionLayer(
    layer_id, INIT_SEQ_LEN, NUM_HEADS, DIM / NUM_HEADS,
    "layer" + std::to_string(layer_id) + "_attention_norm",
    "layer" + std::to_string(layer_id) + "_attention_norm",
    "layer" + std::to_string(layer_id) + "_attention_norm");
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS))}));

  auto ffn_layer = createFeedForwardLayer(
    layer_id, DIM, 4 * DIM, "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_output")}));

  return layers;
}

ModelHandle createLLaMA() {
  using ml::train::createLayer;

  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> layers;

  if (optimize) {
    layers.push_back(createLayer(
      "input",
      {withKey("name", "input0"),
       withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));
  } else {
    layers.push_back(createLayer(
      "input", {withKey("name", "input0"), withKey("input_shape", "1:1:1")}));
  }

  layers.push_back(ml::train::layer::Embedding(
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "out_dim=" + std::to_string(DIM)}));

  for (int i = 0; i < NUM_LAYERS; i++) {
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer = createTransformerDecoder(i, "embedding0");
    else
      transformer = createTransformerDecoder(
        i, "layer" + std::to_string(i - 1) + "_decoder_output");
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  int last_layer = NUM_LAYERS - 1;

  layers.push_back(createLayer(
    "rms_norm", {withKey("name", "output_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("input_layers", "layer" + std::to_string(last_layer) +
                                           "_decoder_output")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "output_of_llama"), withKey("unit", NUM_VOCAB),
     withKey("disable_bias", "true"), withKey("input_layers", "output_norm")}));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  return model;
}

void createAndRun(unsigned int epochs, unsigned int batch_size,
                  std::string text) {
  // setup model
  ModelHandle model = createLLaMA();
  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      // #ifdef ENABLE_FP16
                      withKey("model_tensor_type", "FP16-FP16"),
                      // #endif
                      withKey("save_path", "test_model.bin")});

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  int status = model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  // model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  // std::string weight_path =
  //   optimize ? "./llama_v2.bin" : "./summarization_v2_fp16.bin";
  // std::string weight_path = "./llama_v2.bin";
  std::string weight_path ="./summarization_v2_fp16.bin";

  model->load(weight_path);

  std::vector<float *> input;
  std::vector<float *> label;

  unsigned int input_len = 502;

  int data_size = batch_size * input_len;

  std::string vocab_file_name = "./vocab.json";
  std::string merge_file_name = "./merges.txt";

  float *input_sample = (float *)malloc(sizeof(float) * data_size);

  auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_name, merge_file_name),
                          "Error initializising GPT2 tokenizer\n");

  // auto init_input = tokenizer.encode(text);
  // std::cout << std::endl;

  // float init_input[10] = {1648, 4286, 39, 12330, 6835, 7840 , 50444, 679, 1260, 10224};

  float init_input[502]={ 1648,  4286,    39, 12330,  6835,  7840, 50444,
  679,  1260, 10224,
          663, 17124,  1783,  1014, 40994,  3097,  1324,   277,  8422,  5192,
          288, 82847,   330,   277, 14989, 26464,  2011,   484,   641, 15764,
          321,  2047,  1117,    27,   592, 18835, 72083, 18735,   288, 28383,
          329,  1579,    26,    93, 56094, 49342,   311,  2137,   288, 14989,
          641,  2139, 10326,   412,   277,   885,    29,    26, 22953, 10019,
          663, 28669,   321, 35595, 56502,  2255,  3848,    27, 13126, 50444,
          679,  1260,  3080,   728,  1929,  1783,   288, 28383,   330, 13042,
        15888,    35,    30,    30, 55580,   467,   288,  1579,    26,    93,
        56094,  1206,  7738, 64975, 13889,   329, 23171,  1909,    25,   288,
        15500,   330,  7840,  8264,    27,  2268,  6835,  7840, 28383, 12017,
          277,  1579,    26,    93, 56094,  7738, 64975, 13889,   329, 23171,
         1909,  2776,   499, 18835, 72083,   322,   288, 82847,   433,  3085,
           22,  2268,  6835,  7840, 50631,  5486,   288, 22569,    39,  2046,
        25007,   288, 50444,   679,  1260,  3080,   728,  1929,    27,  2268,
        38035,   898,  1260, 14615,  1324,   611, 26522,   592, 33948, 78076,
          288, 18835, 72083,   641,  7560,   321,   277, 49478,  9835,    25,
          329,  1248,   641,  1350,  7242, 17312,    27,   592, 15500,   330,
         7840,    25, 85873,   277,  4267,    25,  8264,   467,   288, 28383,
          641, 29914,   311,  1783,   501,  5685,   288,  1579,    26,    93,
        56094,   321,  3505,  2046, 52358,  2047,    26,  3516, 16195,   554,
          412,   288, 13042,    25,  2877,   288,  1881,   330, 25977,   426,
         6618,    25,  2047,    26,  3516,  6067,   329, 12782,    27, 14084,
          330,  8244, 17053,   288, 17674,   322, 28669,    25,  1004,  1263,
          679,  7300,   321,   288, 28319,   330,   288, 13042,    25,   288,
        28383, 38291,   288, 14989,   321, 35595, 56502,   329,  1574, 16366,
         6835,  7840,  5553,    27,  9312, 49543, 15243,   679, 14264,   499,
        16697,  1324,   288, 17674,   321,  8931,  4381,  1092,   330,   288,
        10544,  7009,  1640,   428, 91085,    27, 47723,  4694,   467,   288,
        28383,   641, 29914,   311,  1783, 12608,   288,  1579,    26,    93,
        56094,   321,  3505,  2060,  2011,  2047,    26,  3516,  1151,   425,
         3061,   667,   288, 15500,   330,  7840,  1811,   467,   288,  1579,
           26,    93, 56094,   898, 20635,  4929, 82423,   322,   288,  3691,
           27, 19113,  1911,  5870,   501,  5486,   288, 28383,   330,   277,
        13042,   321, 14282,   288, 82847,    25,  2046,  8455,   288, 11756,
          426,  1083, 21155, 37853,   554,   329,  8422,  1909,    25,  2137,
          277, 28249, 17974,  1598,  1911,  5870,   663,  2785, 28383, 50343,
          288,  1579,    26,    93, 56094,   554,    96, 12753,  2938,   329,
        19807,   501,   641,  2046,    95,  4948,   329, 71906,  5835, 15600,
        14416,  3848,   554,    96, 17674,  4148,   596,   277, 17508,  1099,
          412,   288,  8089, 49543,  4753,  3220,   288, 44915, 18735, 11151,
          100,  1130, 13042,   845,    66,    38,    34,    31,    34,    27,
        60685,  4285,  5213,   458,    36,    26,  6132,    26,  1214,  1579,
           26,    93, 56094, 76955, 58893, 11055, 51646, 54343,   288, 14989,
         1324,   288,  8972, 86227,  1573, 24947,  5681,   426,  6618,  1573,
         1783, 45497,   288, 28383,   907,   330,   288, 82847,   426,   277,
        13042,   663, 32635,   321,   552,  2504, 13298, 31517,    27, 11151,
        22569, 59262,  8264,   467, 58893, 11055, 35184,   288,  8867,   412,
         2060,   426, 33586,   329, 22371,  1941,   288,  1431,  2046, 14074,
         4920, 92896, 15600,   212,  9384, 61438,   611,    27,   212,  1648,
  			   5231,    39   };

  // for (auto element: init_input){
  //   std::vector<int64_t> tokens;
  //   tokens.push_back(element);
  //   std::cerr << tokenizer.decode(tokens) << "("<<element<<") "<< std::flush;
  // }

  // exit(0);

  //  INIT_SEQ_LEN = init_input.size();
  // unsigned int input_len = 502;
  // std::cout << input_len << std::endl;
  // unsigned int input_len = 5;

  // float init_data[INIT_SEQ_LEN] = {5058, 10832};
  // float init_data[INIT_SEQ_LEN] = {
  //   0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   10,  20,  30,  40,
  //   50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900};

  if (optimize) {
    //    for (unsigned int i = 0; i < INIT_SEQ_LEN; ++i) {
    //      input_sample[i] = init_input[i];
    //    }

    input.push_back(input_sample);

    auto output = model->inference(1, input, label);

    nntrainer::Tensor output_tensor({batch_size, 1, INIT_SEQ_LEN, NUM_VOCAB},
                                    output[0]);

    for (unsigned int i = 0; i < INIT_SEQ_LEN; ++i) {
      nntrainer::Tensor output_step = output_tensor.getSharedDataTensor(
        {batch_size, 1, 1, NUM_VOCAB}, i * batch_size * NUM_VOCAB);
      std::cerr << output_step << "\n";
    }
  } else {

    input_sample[0] = static_cast<float>(init_input[0]);

    input.push_back(input_sample);

    for (unsigned int i = 1; i < input_len + NUM_TO_GENERATE; ++i) {

      auto output =
        model->incremental_inference(1, input, label, MAX_SEQ_LEN, i - 1);

      nntrainer::Tensor output_tensor({batch_size, 1, 1, NUM_VOCAB}, output[0]);

      std::cout << i << " - " << output[0][0] << " " << output[0][1] << " "
                << output[0][2] << " ----> ";

      std::vector<int64_t> token_ids;

      if (i < input_len) {
        input_sample[0] = static_cast<float>(init_input[i]);
        std::cout << init_input[i] << std::endl;
      } else {

        std::vector<unsigned int> ids = output_tensor.argmax();
        input_sample[0] = static_cast<float>(ids[0]);
	
        token_ids.push_back(static_cast<int64_t>(ids[0]));
      	auto decoded_str = tokenizer.decode(token_ids);
      	std::cerr << decoded_str << " " << std::flush;
      }
    }
  }
  std::cout << std::endl;
}

int main(int argc, char *argv[]) {
  auto &app_context = nntrainer::AppContext::Global();
  try {
    app_context.registerFactory(nntrainer::createLayer<custom::SwiGLULayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    app_context.registerFactory(nntrainer::createLayer<custom::RMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    const std::vector<std::string> args(argv + 1, argv + argc);
    std::string text = args[0];
    createAndRun(epoch, batch_size, text);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  int status = EXIT_SUCCESS;
  return status;
}

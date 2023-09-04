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

#include <app_context.h>
#include <rms_norm.h>
#include <rotary_embedding.h>
#include <swiglu.h>
#include <transpose_layer.h>
#include "encoder.hpp"

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

// Hyper params for LLaMA
int const DIM = 2304;
int const NUM_LAYERS = 28;
int const NUM_HEADS = 18;

int const MULTIPLE_OF = 256;

float const NORM_EPS = 0.000001;
int const NUM_VOCAB = 96000;
int MAX_SEQ_LEN = 1124;
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

void createAndRun(unsigned int epochs, unsigned int batch_size, std::string text) {
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
  // std::string weight_path ="./llama_v2.bin";
  std::string weight_path ="./summarization_v2_fp16.bin";
  
  model->load(weight_path);

  std::vector<float *> input;
  std::vector<float *> label;

  int data_size = batch_size * INIT_SEQ_LEN;


  std::string vocab_file_name = "./vocab.json";
  std::string merge_file_name = "./merges.txt";
  
  float *input_sample = (float *)malloc(sizeof(float) * data_size);


   auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_name, merge_file_name),
			  "Error initializising GPT2 tokenizer\n");

  auto init_input = tokenizer.encode(text);
  std::cout << std::endl;
  // float init_input[485] = {101,  2048,  2250,  2634,  8221,  2031,  2042,  3718,  2013,  4611,
  // 			   2044,  2027,  7283,  2288,  2046,  1037,  2954,  2503,  1996, 13828,
  // 			   1997,  1037,  4946,  3859,  2077,  2009,  2001,  5115,  2000,  2202,
  // 			   2125,  1012,  1996, 11477, 10719,  5994,  1996,  2952,  1998,  2522,
  // 			   1011,  4405, 12591,  2096,  1996,  4946,  2001,  2108,  4810,  2005,
  // 			   1037,  2753,  1011,  3371,  4990,  2013,  6768,  2000, 28355,  2197,
  // 			   2305,  1012,  2119,  8221,  2031,  2042,  4315, 14122,  6850,  2044,
  // 			   1996,  2952,  1997,  3462,  9932,  2575, 14526, 10865,  2008,  1996,
  // 			   2522,  1011,  4405,  2018, 28616,  4783,  3270,  7178,  1998,  4930,
  // 			   2032,  1010,  1996,  2335,  1997,  2634,  2988,  1012,  2019,  2250,
  // 			   2634,  2952,  4447,  1037,  2522,  1011,  4405, 28616,  4783,  3270,
  // 			   7178,  1998,  4930,  2032,  2076,  2019, 11477, 10719,  1999,  1996,
  // 			   13828,  1006,  5371,  1007,  2019,  2250,  2634, 14056,  2409,  1996,
  // 			   3780,  1024,  1520,  2119,  1996,  8221,  2031,  2042,  4315, 14122,
  // 			   6850,  1012,  2019,  9934,  2038,  2042,  3641,  2046,  2023,  1012,
  // 			   1521,  1996,  8582, 16818,  1996, 11477, 10719,  2001,  3132,  2000,
  // 			   1037, 12064,  6685,  1010,  1998,  2045,  2001,  2053,  3558,  4808,
  // 			   1012,  1996,  2335,  1997,  2634,  1010, 27394,  1037,  3120,  1010,
  // 			   2988,  2008,  1996,  2952,  2001, 17536,  2044,  2002,  2356,  1996,
  // 			   2522,  1011,  4405,  2000,  2501,  1520,  4187,  2202,  1011,  2125,
  // 			   4481,  1521,  2005,  1996,  3462,  1010,  2164,  1996,  2193,  1997,
  // 			   5467,  2006,  2604,  1010,  2202,  1011,  2125,  3635,  1998,  4762,
  // 			   1012,  2612,  1997,  3202,  7316,  1996,  5043,  1999,  6768,  1010,
  // 			   2029,  2052,  2031,  2419,  2000,  1996, 16990,  1997,  1996,  3462,
  // 			   1010,  1996,  2952,  5520,  1996,  4946,  2000, 28355,  1998,  2059,
  // 			   6727,  2250,  2634,  3095,  1012,  2796,  5734,  4584,  2031,  3390,
  // 			   2019,  4812,  2046,  1996,  5043,  2000,  5646,  3251,  2151,  1997,
  // 			   1996,  4243,  2920,  2323,  2022, 28675,  1012,  4311,  6592,  2008,
  // 			   1996,  2952,  2001, 17536,  2044,  4851,  1996,  2522,  1011,  4405,
  // 			   2000,  2501,  2592,  2077,  2202,  1011,  2125,  1012,  1037,  3189,
  // 			   2011,  1996,  2335,  1997,  2634,  2056,  2008,  1996,  2522,  1011,
  // 			   4405,  2038,  4320,  2714, 13519,  1999,  1996,  2627,  1012,  2093,
  // 			   2086,  3283,  2002,  2409,  1996,  2952,  1997,  1037,  3462,  2000,
  // 			   6164,  1996, 13828,  1010,  1520,  6366,  1996,  3340,  2006,  2010,
  // 			   3797,  9127,  1521,  1998,  2954,  2032,  1010,  2096,  1037, 12087,
  // 			   6406,  2048,  2086,  3283,  2013,  2178,  2952,  8781,  1996,  2522,
  // 			   1011,  4405,  1521,  1055,  5177,  2740,  1998,  3555,  2002,  2001,
  // 			   1520, 12726,  1998,  4895,  4783, 18935,  1521,  1012,  2197,  2305,
  // 			   1521,  1055,  5043,  3310,  2012,  1037,  7591,  2051,  2005,  1996,
  // 			   3293,  5734,  3068,  2206,  1996, 10576,  5994,  2446,  9328,  2015,
  // 			   3462,  1018,  2226,  2683, 25746,  2629,  1012, 14766,  2903,  2676,
  // 			   1011,  2095,  1011,  2214,  2522,  1011,  4405, 12460, 11320, 16313,
  // 			   2480,  9969,  8007,  1996,  4946,  2046,  1996,  2413, 13698,  1516,
  // 			   4288,  3071,  2006,  2604,  1516,  2044, 14889,  1996,  2952,  2041,
  // 			   1997,  1996, 13828,  2006,  1037,  3462,  2013,  7623,  2000, 18160,
  // 			   1012,  2446,  3780, 12170,  6392,  2988,  2008, 11320, 16313,  2480,
  // 			   9022,  1996,  4274,  2005,  2592,  2006,  5920,  1998,  6245,  2478,
  // 			   1996,  2171,  1520,  3712, 24844,  4014,  1521,  1012,  3531,  7680,
  // 			   7849,  4697,  2023,  1012,   102};

  // for (auto element: init_input){
  //   std::vector<int64_t> tokens;
  //   tokens.push_back(element);
  //   std::cerr << tokenizer.decode(tokens) << "("<<element<<") "<< std::flush;
  // }

  // exit(0);

  // INIT_SEQ_LEN = init_input.size();
  unsigned int input_len = init_input.size();
  /* INIT_SEQ_LEN=490; */

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
    
    ((uint *)(input_sample))[0] = init_input[0];

    input.push_back(input_sample);

    for (unsigned int i = 1; i < iniput_len + NUM_TO_GENERATE; ++i) {

      unsgined int from = i;
      
      if (i >= INIT_SEQ_LEN)
	from = INIT_SEQ_LEN-1;
	
      auto output =
        model->incremental_inference(1, input, label, INIT_SEQ_LEN, from - 1);

      // nntrainer::Tensor output_tensor({batch_size, 1, 1, NUM_VOCAB}, output[0]);


      long diff = std::distance(
          output[0], std::max_element(output[0], output[0] + NUM_VOCAB));
      
      // std::vector<unsigned int> ids = output_tensor.argmax();
      
      std::vector<int64_t> token_ids;
      
      if (i < INIT_SEQ_LEN) {
        ((uint *)(input_sample))[0] = init_input[i];
	
	// for(auto element:ids){
	// token_ids.push_back(static_cast<int64_t>(init_input[i]));
	// }
	// token_ids.push_back(static_cast<int64_t>(diff));
	
	// std::cout<<tokenizer.decode(token_ids) << "("<<((uint*)(input_sample))[0] << "), ";
      } else {
        // ((uint *)(input_sample))[0] = ids[0];
        ((uint *)(input_sample))[0] = diff;
	// for (unsigned int j=0;j<2; j++)
	//    std::cout << output[0][j] << " ";
	// std::cout<<std::endl << "output :                     "<< ids[0] << std::endl;
	
	// for(auto element:ids){
	//   token_ids.push_back(static_cast<int64_t>(ids[0]));
	// }
	token_ids.push_back(static_cast<int64_t>(diff));
      }

      if (i >= INIT_SEQ_LEN) {
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
    const std::vector<std::string> args(argv+1, argv+argc);
    std::string text = args[0];
    createAndRun(epoch, batch_size,text);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  int status = EXIT_SUCCESS;
  return status;
}

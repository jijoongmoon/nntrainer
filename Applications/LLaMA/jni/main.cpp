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

ModelHandle g_model;

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

void run (std::string text){
  std::vector<float *> input;
  std::vector<float *> label;


  std::string vocab_file_name = "./vocab.json";
  std::string merge_file_name = "./merges.txt";
  

  auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_name, merge_file_name),
                          "Error initializising GPT2 tokenizer\n");
  auto init_input = tokenizer.encode(text);

  unsigned int input_len = init_input.size();
  
  int data_size = batch_size * input_len;

  float *input_sample = (float *)malloc(sizeof(float) * data_size);


  if (optimize) {

    input.push_back(input_sample);

    auto output = g_model->inference(1, input, label);

    nntrainer::Tensor output_tensor({batch_size, 1, INIT_SEQ_LEN, NUM_VOCAB},
                                    output[0]);

    for (unsigned int i = 0; i < INIT_SEQ_LEN; ++i) {
      nntrainer::Tensor output_step = output_tensor.getSharedDataTensor(
        {batch_size, 1, 1, NUM_VOCAB}, i * batch_size * NUM_VOCAB);
    }
  } else {

    input_sample[0] = static_cast<float>(init_input[0]);

    input.push_back(input_sample);

    for (unsigned int i = 1; i < input_len + NUM_TO_GENERATE; ++i) {

      auto output =
        g_model->incremental_inference(1, input, label, MAX_SEQ_LEN, i - 1);

      nntrainer::Tensor output_tensor({batch_size, 1, 1, NUM_VOCAB}, output[0]);
      std::vector<unsigned int> ids = output_tensor.argmax();
        
#ifdef ENABLE_FP16        
        for(auto o : output){
          free(o);
        }
#endif        

      std::vector<int64_t> token_ids;
      if (i < input_len) {
        if (i == 1){
          std::cout << "### Input : ("<<input_len<<" words)" << std::endl;
        }
        std::cout <<" Progress Reading: "<< (int)((float)(i)/(float)(input_len)*100.0) << " % \r";
        std::cout.flush();
        input_sample[0] = static_cast<float>(init_input[i]);
        /* std::cout << init_input[i] << " "; */
      } else {
        
        if ( i == input_len ){
          std::cout <<" Progress Reading: 100 % " << std::endl;        
          std::cout <<std::endl<< "### Output : "<< std::endl;
        }
        input_sample[0] = static_cast<float>(ids[0]);
	
        token_ids.push_back(static_cast<int64_t>(ids[0]));
      	auto decoded_str = tokenizer.decode(token_ids);
      	std::cout << decoded_str << " " ;
        std::cout.flush();
      }
      
    }
  }
  std::cout << std::endl;
}

void createAndRun(unsigned int epochs, unsigned int batch_size) {
  // setup model
  g_model = createLLaMA();
  g_model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      // #ifdef ENABLE_FP16
                      withKey("model_tensor_type", "FP16-FP16"),
                      // #endif
                      withKey("save_path", "test_model.bin")});

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  g_model->setOptimizer(std::move(optimizer));

  int status = g_model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = g_model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  // model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
  std::string weight_path ="./summarization_v2_fp16.bin";

  g_model->load(weight_path);
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

    createAndRun(epoch, batch_size);    
    /* std::string text = args[0]; */
    std::vector< std::string > texts;
    texts.push_back( "## input: Two Air India pilots have been removed from duty after they reportedly got into a fight inside the cockpit of a plane shortly before it was scheduled to take off. The altercation involving the captain and co-pilot erupted while the plane was being prepared for a 50-minute journey from Delhi to Jaipur last night. Both pilots have been derostered after the captain of flight AI611 complained that the co-pilot had misbehaved and struck him, the Times of India reported. An Air India captain claims a co-pilot misbehaved and struck him during an altercation in the cockpit (file) An Air India spokesman told the newspaper: \u2018Both the pilots have been derostered. An inquiry has been ordered into this.\u2019 The airline insists the altercation was limited to a verbal argument, and there was no physical violence. The Times of India, quoting a source, reported that the captain was assaulted after he asked the co-pilot to record \u2018critical take-off figures\u2019 for the flight, including the number of passengers on board, take-off weight and fuel. Instead of immediately reporting the incident in Delhi, which would have led to the cancellation of the flight, the captain flew the plane to Jaipur and then informed Air India staff. Indian aviation officials have launched an investigation into the incident to determine whether any of the parties involved should be disciplined. Reports suggest that the captain was assaulted after asking the co-pilot to record information before take-off . A report by the Times of India said that the co-pilot has faced similar accusations in the past. Three years ago he told the captain of a flight to exit the cockpit, \u2018remove the stars on his shirt collar\u2019 and fight him, while a complaint filed two years ago from another captain questioned the co-pilot\u2019s mental health and claimed he was \u2018rude and unbecoming\u2019. Last night\u2019s incident comes at a sensitive time for the commercial aviation industry following the tragedy involving Germanwings flight 4U9525. Investigators believe 27-year-old co-pilot Andreas Lubitz deliberately crashed the plane into the French Alps \u2013 killing everyone on board \u2013 after locking the captain out of the cockpit on a flight from Barcelona to Dusseldorf. German newspaper Bild reported that Lubitz searched the internet for information on suicide and depression using the name \u2018Skydevil\u2019.\nPlease summarize this.\n## output: ");
    //texts.push_back( "## input: A 12-year-old girl battling Leukemia for two years has been kicked out of school for her lack of attendance. 'I didn't do anything wrong, but they still got rid of me,' Rose McGrath of battle Creek, Michigan said tearfully. Last week St. Joseph's Middle School, a private Catholic School, sent a letter to Rose McGrath and her dismissing her from the school for low attendance and poor academic performance. Scroll down for video . Heartbroken: 'I didn't do anything wrong, but they still got rid of me,' Rose McGrath of battle Creek, Michigan said tearfully of her school kicking her out for poor attendance because of her Leukemia . Dismissed: John Fleckenstein, with Battle Creek Area Catholic Schools, claims that the school made many accommodations for Rose and that none of them seemed to help her enough . Rose's mother Barbara McGrath was just as heartbroken to hear the news and said even though her daughter is no longer getting cancer treatment, that her recovery will take some time. 'Even though she's now done with her treatments you still have a very long recovery process because you've basically just put two and a half years of poison into your body. You're not recovering overnight,' said Rose's mother, Barbara McGrath. Rose has been attending the Battle Creek Catholic Schools her whole entire life and when she was diagnosed with lymphoblastic leukemia her world was turned upside down. Rose told WWMT that school was the one place that she actually felt normal. 'When I'm at home, I'm sick, I don't feel well; no one else does that. But when I'm at school I'm like everyone else,' Rose said. A struggle: Rose was diagnosed with leukemia in 2012 and though she is done with her treatment she still feels ill and has trouble attending her classes and finishing her schoolwork . Not having fun: Rose's mother says her daughter isn't skipping out of school 'to have fun' but that she . Long battle: Rose's mother Barbara (right) stands beside her daughter Rose she continues to heal from her debilitating disease . St. Joseph's catholic school that charges as much as $6,983 per year said that they were generous enough to provide Rose with special accommodations because of her illness. It is unclear as to whether Rose was receiving financial assistance and they did not elaborate on what kinds of accommodations were made. 'These were extraordinary circumstances, but so many accommodations were made we felt eventually it became a point where we really had to help Rose, by being able to make sure that she was getting the assistance that she needed and to learn,' said Father John Fleckenstein, with Battle Creek Area Catholic Schools. The school says that Rose only attended school 32 days out of this entire school year. Rose McGrath's parents feel as though the school is seriously failing their child. 'The accommodations which were made were woefully inadequate for a child with such a serious diagnosis,' said Rose's father Tom McGrath to WWMT. 'It's not like she's out at the mall having fun, she's in her bed, sick with nausea, vomiting, abdominal pain. She\u2019s not having fun, she's sick. She\u2019d be at school if she could,' Barbara said. The McGrath's say that they filed a complaint with the Office of Civil Rights and that they are waiting for a response. Trying to help others: Rose raising money for St. Baldrcks to find a cure for childhood cancer . Support: Rose has plenty of support from family and friends but she will need to figure out where she will go to school if she is not allowed back at St. Josephs .\nPlease summarize this.\n## output: ");
    //texts.push_back("## input: Children are being auditioned at school for Britain\u2019s Got Talent to stop them bunking off lessons, it has been revealed. Teachers say that youngsters have been playing truant as they attempt to find instant stardom on the ITV talent show. As a result, for the first time producers have held auditions at dozens of secondary schools and colleges around the UK for the new series. Scroll down for video . Brace yourselves: Britain's Got Talent judges Amanda Holden and Alesha Dixon strike glamorous poses . The move has helped reduce unauthorised absence levels in some schools on the days of auditions. Teacher Angela Butler said Britain\u2019s Got Talent spent two hours at her school \u2013 Newtown High School, in Powys, mid-Wales \u2013 and a local sixth-form college in November. Around 30 Newtown High School pupils aged 11 to 18 sang and played instruments for producers during preliminary auditions after getting permission from their parents. Mrs Butler, a head of year, described similar on-site auditions across the region as a preventative measure to stop youngsters skipping school. Speaking at the NASUWT union\u2019s annual conference in Cardiff, she attacked the shocking truancy levels in the city caused by auditions for talent shows. The Britain's Got Talent judges and presenters are seen on stage before the start of the 2015 series . The 52-year-old said the day for auditions was \u2018the day throughout Wales when most kids are absent\u2019. She added: \u2018I think that is an absolute travesty, that kids think that is the way that they\u2019re going to have a route out of poverty and not education.\u2019 Simon Cowell has defended allowing young children to take part in Britain\u2019s Got Talent, despite fears by child welfare campaigners that many of them are too young to cope with fame. The music mogul says the ITV show \u2013 which has no lower age limit \u2013 provides opportunities to those who might not otherwise have them. Cowell, 55, who has a 14-month-old son, Eric, with partner Lauren Silverman, said: \u2018Every kid who comes on our show is there because the child wants to be on our show and we make sure we look after them. \u2018As for whether it\u2019s the right thing having someone so young singing on the show \u2013 they may not have the opportunities in life other people have.\u2019 Speaking afterwards, Mrs Butler told how Britain\u2019s Got Talent had changed how it auditions to stop pupils \u2018coming down to Cardiff\u2019 on school days. She said: \u2018For a couple of years we had absences. It must have been a problem throughout Wales because this time, for the auditions that have happened for Britain\u2019s Got Talent, they actually sent people into schools. \u2018When the producers from the programme came to our school, they said they were going all around schools to prevent exactly this [truancy]. 'They spent two hours in our school and they must have seen 30 kids, perhaps more than that. Then they moved onto a sixth form college in the same town.\u2019 But she also bemoaned how children now seem to think fame is the only way to succeed: \u2018Children now think The Voice or Britain\u2019s Got Talent or The X Factor or winning the lottery \u2013 that\u2019s how they\u2019re going to get on and have the things that they see celebrities doing. I do think that kids want this quick fix.\u2019 Britain\u2019s Got Talent \u2013 and its judging panel Simon Cowell, David Walliams, Amanda Holden and Alesha Dixon \u2013 is returning on Saturday for its ninth series, with ITV chiefs expecting bumper audience ratings. A source from the show said: \u2018Last year was the first time we\u2019d ever gone to schools. We did go to schools all around the country. \u2018We are the one of the few talent shows that allow kids to enter, so the logical extension of that was going to schools to allow kids to audition as easily as possible.\u2019 BGT \u2013 and its judging panel Simon Cowell, David Walliams, Amanda Holden (pictured) and Alesha Dixon \u2013 is returning on Saturday for its ninth series, with ITV chiefs expecting bumper audience ratings .\nPlease summarize this.\n## output: ");
    //texts.push_back("## input: (CNN)Ever had a headache so big, you felt like drilling a hole in your head to let the pain out? In Neolithic times trepanation -- or drilling a hole into the skull -- was thought to be a cure for everything from epilepsy to migraines. It could even have been a form of emergency surgery for battle wounds. But while there is still conjecture about the real reasons behind the mysterious procedure, what is known is that the implement often used to carry out the primitive surgery was made from one of the sharpest substances found in nature -- obsidian. Obsidian -- a type of volcanic glass -- can produce cutting edges many times finer than even the best steel scalpels. At 30 angstroms -- a unit of measurement equal to one hundred millionth of a centimeter -- an obsidian scalpel can rival diamond in the fineness of its edge. When you consider that most household razor blades are 300-600 angstroms, obsidian can still cut it with the sharpest materials nano-technology can produce. Even today, a small number of surgeons are using an ancient technology to carry out fine incisions that they say heal with minimal scarring. Dr. Lee Green, professor and chair of the Department of Family Medicine at the University of Alberta, says he routinely uses obsidian blades. \"The biggest advantage with obsidian is that it is the sharpest edge there is, it causes very little trauma to tissue, it heals faster and more importantly it heals with less scarring,\" he said. \"It makes for the best cosmetic outcome.\" He explained that steel scalpels at a microscopic level have a rough cutting edge that tears into tissue, a function of the crystals that make up the metal. Obsidian, meanwhile, cleaves into a fine and continuous edge when properly cut. Dr. Green said he once helped documentary makers produce a program on surgical technology in ancient Egyptian, setting up a blind test on the cutting power of obsidian. Using cultured-skin burn dressing, a substance composed of skin cells, he made an incision with a modern scalpel and a parallel incision with an obsidian scalpel. The host of the program was then invited to look at the cuts under a video microscope and tell the difference. \"It wasn't hard to tell the difference at all -- as soon as he turned around everyone in the studio was like 'Ohhh',\" Dr. Green said. \"Under the microscope you could see the obsidian scalpel had divided individual cells in half, and next to it the steel scalpel incision looked like it had been made by a chainsaw.\" Modern obsidian scalpels look nothing like the decorative flint-knapped knives of Neolithic man, often resembling their modern counterparts in everything except for the blade edge, but Dr. Green said they are a very different animal. \"The feel is very different because obsidian has no 'bite,'\" he said. \"If you look under the microscope at a steel scalpel edge it looks almost like a saw, it has teeth, whereas obsidian is smooth even microscopically. \"It's a very different feel to work with and you have to practice before you start using it in surgery. \"You also have to be careful not to nick yourself with it because you don't even feel it!\" And Dr. Green believes incisions made with these blades heal faster. He said a colleague who needed a mole removed agreed to undergo an experiment where half the procedure was carried out with an obsidian scalpel and the other half was removed with steel. \"What's really fun is seeing it heal,\" he said. \"Four weeks later the difference was quite remarkable -- there was very much a difference in scarring.\" In Germany, the manufacturer Fine Science Tools produces obsidian scalpels which can be used in situations where the patient may have an allergy to steel or metal. \"For studies where trace metals from ordinary scalpel blades cannot be tolerated, these very special obsidian scalpels may provide the answer,\" the company says. At \u20ac99 per scalpel ($107.40), they represent a considerable saving on their diamond cousins which the company prices at \u20ac712.50 ($772.60). But there has been little academic research into the efficacy of obsidian blades compared to steel scalpels, and they do have disadvantages: Obsidian scalpels are not Food and Drug Administration (FDA) approved, and they are extremely brittle and prone to breaking if lateral forces are applied -- meaning they are unlikely to ever be in widespread use. Dr. Green, whose scalpels were manufactured for him by an expert flint-knapper and archaeologist Errett Callahan, concedes the Stone Age scalpels are not for everyone. \"If it was let loose on the market there'd be far too many injuries from it,\" he said. \"It's very fragile and it's very easy to break pieces off.\"\nPlease summarize this.\n## output: ");
    //texts.push_back("## input: It may not be a looker, but the Lusitanian toadfish can do much more than croak. The fish, which lives in rocky crevices in the Mediterranean Sea and Atlantic Ocean and glides over the muddy sea floor, can whistle, croak and grunt. Scientists have found that\u00a0Halobatrachus didactylus makes five types of calls not only to attract a mate, but to warn off rivals too may be trying to swipe their nesting site. Lusitanian toadfish (pictured) make five types of calls and males can even sing in choruses to attract mates.\u00a0The fish woo females with long, rhythmical boatwhistles, which also act as a deterrent to love rivals, . Male fish woo females with long, rhythmical noises that sound like boat whistles, which also act as a deterrent to love rivals,\u00a0New Scientist reported. They build nests under rocks during mating season, which runs from May to July and sing to attract female visitors. If they are successful, the pair mate and the male looks after young that hatch from sticky eggs until they are ready to fend for themselves after around a month. Because males, which grow up to 20 inches (50cm) long, are fiercely territorial, their song also serves as a warning for rivals to stay away. Males build nests under rocks during mating season (pictured), which runs from May to July and sing to attract female visitors. The creature is well camouflaged so sound is the best way to get attention . This is important because males nest close together to one another and form singing choruses like frogs or toads to demonstrate their virility and strength. The whistles vary according to the size of the fish, meaning that specific calls from larger specimens are particularly effective at deterring another fish from picking a fight and stealing a nest. Lusitanian toadfish typically weigh more than four lbs (2kg) and their large, flat heads and wide mouths, make them look like toads, giving their their name. Portuguese scientists discovered that the sounds made by Lusitanian toadfish indicate who they are, their motivation and information about their nest. If they are successful, the pair mate and the male looks after young that hatch from sticky eggs until they are ready to fend for themselves after around a month. Here, a female lays eggs, which stick to the roof of a nest . The experts analysed the frequencies of different songs. \u00a0Water temperature, tide level, fish motivation and the level of social interactions affected most acoustic parameters analysed (shown above). For example, during low tide, the sounds made by the fish were shorter in duration and had lower main frequencies . Male mice sing to woo females, scientists claim. They also change their tune depending on whether she is within sight or not. The females, meanwhile, seem to like some of the songs more than others. In a quirky study that could shed help shed light on autism and other conditions that involve difficulties in communication, researchers from Duke University in North Carolina studied male mice that were either placed in a cage with a female \u2013 or one with just her scent. Special equipment was used to record and analyse their squeaks, which are so high-pitched that people can\u2019t hear them. This revealed that they sang one song when they could simply smell a female and another one when they could see her. When they could merely smell a potential mate, they belted out an extremely shrill and complex song, perhaps in an attempt to make themselves known. But when she was within sight, they serenaded her more softly. These songs also had a more simple structure and were longer. Clara Amorim of ISPA University Institute in Portugal studied the different sounds made by the fish, which contracts muscles in its swim bladders to release air, a little like bagpipe. It makes different sounds by contracting its bladders in various ways. To prove that 'boatwhistles' are used as a deterrent as well as for seduction, her team deflated the swim bladders of fish under anaesthesia, so that the creatures could still contract their muscles, but were rendered silent. The team found that mute males were more likely to face intruders in their nests, suggesting that 'boatwhistles' are an effective warning siren. \u2018Boatwhistles are a cheap way to exclude intruders without engaging in a fight,\u2019 Dr Amorim said. \u2018Seeing that a nest is occupied is not as effective as hearing that there is a male in the nest eager to defend its territory.\u2019 The sounds are modified by environmental factors such as tide level and water temperature as well as courtship motivation and by social interactions. For example, during low tide, the water temperature rises slightly and it's harder for sound to travel in shallow waters, which means that fewer fish sing and their noises are shorter in length and lower in pitch. They found that males that are desperate to mate sing a lot more at higher frequencies with longer notes. Males singing as part of a chorus sing more than they do when they are alone. Fish that sing the most, were found to be capable of a contracting their sound-producing muscles more and therefore showed that their body is in good condition. This means that males that can sing for longer have more energy reserves to defend the nest and to care for the eggs, and could enjoy higher social status and reproductive success as a result. Other fish sing too, such as clownfish, which click their jaws together to scare off intruders. Male Lusitanian toadfish\u00a0build nests under rocks during mating season, which runs from May to July and sing to attract female visitors. Other\u00a0fish sing too, such as clownfish (pictured), which click their jaws together to scare off intruders .\nPlease summarize this.\n## output: ");

    for(auto t : texts)
      run(t);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  int status = EXIT_SUCCESS;
  return status;
}

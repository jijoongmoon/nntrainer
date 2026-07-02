// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   spike_bigvgan.cpp
 * @brief  Stage-C spike: build the full Qwen2.5-Omni BigVGAN graph, load
 *         bigvgan.bin, feed the dumped processed_mel, and compare the output to
 *         the dumped reference (default wav; or an intermediate tap for
 *         bisecting a mismatch).
 *
 *   spike_bigvgan [in_dir=/tmp/bigvgan_spike] [weight_bin] [target=wav]
 *   target in {conv_pre, ups0..ups5, stage0..stage5, activation_post, wav}
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>

#include "antialiased_snake.h"
#include "conv1d_transpose.h"
#include "scale_layer.h"

using ml::train::createLayer;
using LayerHandle = std::shared_ptr<ml::train::Layer>;

static std::string kv(const std::string &k, const std::string &v) {
  return k + "=" + v;
}
static std::string S(int v) { return std::to_string(v); }

static std::vector<float> read_bin(const std::string &p, size_t n) {
  std::ifstream f(p, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open " + p);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()),
         static_cast<std::streamsize>(n * sizeof(float)));
  if (static_cast<size_t>(f.gcount()) != n * sizeof(float))
    throw std::runtime_error("short read " + p);
  return v;
}
static size_t file_floats(const std::string &p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot stat " + p);
  return static_cast<size_t>(f.tellg()) / sizeof(float);
}

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "/tmp/bigvgan_spike";
  const std::string wbin =
    argc > 2 ? argv[2]
             : "/home/jijoongmoon/WorkSpace1/nntrainer-p/Applications/CausalLM/"
               "models/qwen2.5-omni-3b-bigvgan/bigvgan.bin";
  const std::string target = argc > 3 ? argv[3] : "wav";

  const int UP_K[6] = {11, 7, 4, 4, 4, 4};
  const int UP_R[6] = {5, 3, 2, 2, 2, 2};
  const int RES_K[3] = {3, 7, 11};
  const int RES_D[3] = {1, 3, 5};
  const int MEL = 80, INIT_CH = 1536, T0 = 128;

  auto &eng = nntrainer::Engine::Global();
  auto *app_context =
    static_cast<nntrainer::AppContext *>(eng.getRegisteredContext("cpu"));
  app_context->registerFactory(
    nntrainer::createLayer<causallm::Conv1DTransposeLayer>);
  app_context->registerFactory(
    nntrainer::createLayer<causallm::AntialiasedSnakeLayer>);
  app_context->registerFactory(nntrainer::createLayer<causallm::ScaleLayer>);

  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  // Build the ordered layer list; stop after `target` so it becomes the output.
  std::vector<LayerHandle> layers;
  std::string out_name;
  bool done = false;
  auto add = [&](const std::string &type, const std::string &name,
                 std::vector<std::string> props, const std::string &inputs) {
    if (done)
      return;
    if (!inputs.empty())
      props.push_back(kv("input_layers", inputs));
    props.push_back(kv("name", name));
    layers.push_back(createLayer(type, props));
    out_name = name;
  };
  auto reached = [&](const std::string &name) {
    if (name == target)
      done = true;
  };

  add("input", "input0",
      {kv("input_shape", S(MEL) + ":1:" + S(T0))}, "");
  add("conv1d", "conv_pre",
      {kv("filters", S(INIT_CH)), kv("kernel_size", "7"), kv("stride", "1"),
       kv("padding", "3")},
      "input0");
  reached("conv_pre");

  std::string prev = "conv_pre";
  int ch = INIT_CH;
  for (int i = 0; i < 6 && !done; ++i) {
    int out_ch = ch / 2;
    std::string ups = "ups" + S(i);
    add("conv1d_transpose", ups,
        {kv("filters", S(out_ch)), kv("kernel_size", S(UP_K[i])),
         kv("stride", S(UP_R[i])), kv("padding", S((UP_K[i] - UP_R[i]) / 2))},
        prev);
    reached(ups);

    std::vector<std::string> amp_outs;
    for (int b = 0; b < 3; ++b) {
      int kb = RES_K[b];
      std::string a = ups;
      for (int kk = 0; kk < 3; ++kk) {
        int d = RES_D[kk];
        std::string base = "s" + S(i) + "_b" + S(b) + "_k" + S(kk);
        add("antialiased_snake", base + "_act1", {}, a);
        add("conv1d", base + "_c1",
            {kv("filters", S(out_ch)), kv("kernel_size", S(kb)),
             kv("stride", "1"), kv("dilation", S(d)),
             kv("padding", S(d * (kb - 1) / 2))},
            base + "_act1");
        add("antialiased_snake", base + "_act2", {}, base + "_c1");
        add("conv1d", base + "_c2",
            {kv("filters", S(out_ch)), kv("kernel_size", S(kb)),
             kv("stride", "1"), kv("dilation", "1"),
             kv("padding", S((kb - 1) / 2))},
            base + "_act2");
        add("addition", base + "_res", {}, a + "," + base + "_c2");
        a = base + "_res";
      }
      amp_outs.push_back(a);
    }
    add("addition", "s" + S(i) + "_sum", {},
        amp_outs[0] + "," + amp_outs[1] + "," + amp_outs[2]);
    add("scale", "s" + S(i) + "_mean", {kv("scale", "0.3333333432674408")},
        "s" + S(i) + "_sum");
    reached("stage" + S(i)); // alias: stage{i} == s{i}_mean output
    if (target == "stage" + S(i))
      out_name = "s" + S(i) + "_mean";
    prev = "s" + S(i) + "_mean";
    ch = out_ch;
  }
  if (!done) {
    add("antialiased_snake", "act_post", {}, prev);
    reached("activation_post");
    if (target == "activation_post")
      out_name = "act_post";
    add("conv1d", "conv_post",
        {kv("filters", "1"), kv("kernel_size", "7"), kv("stride", "1"),
         kv("padding", "3"), kv("disable_bias", "true")},
        "act_post");
  }

  for (auto &l : layers)
    model->addLayer(l);
  model->setProperty({kv("batch_size", "1"), kv("epochs", "1"),
                      kv("model_tensor_type", "FP32-FP32")});
  if (model->compile(ml::train::ExecutionMode::INFERENCE))
    throw std::runtime_error("compile failed");
  if (model->initialize(ml::train::ExecutionMode::INFERENCE))
    throw std::runtime_error("initialize failed");
  model->load(wbin, ml::train::ModelFormat::MODEL_FORMAT_BIN);

  std::vector<float> in = read_bin(dir + "/input.bin", (size_t)MEL * T0);
  const std::string exp_file =
    (target == "wav") ? dir + "/expected.bin" : dir + "/" + target + ".bin";
  const size_t n = file_floats(exp_file);
  std::vector<float> expect = read_bin(exp_file, n);

  std::vector<float *> input{in.data()};
  std::vector<float *> label;
  std::vector<float *> out = model->inference(1, input, label);
  const float *y = out[0];

  double max_abs = 0.0, sum2 = 0.0;
  for (size_t i = 0; i < n; ++i) {
    float yi = y[i];
    if (target == "wav")
      yi = std::max(-1.0f, std::min(1.0f, yi)); // host clamp
    double dd = std::abs(static_cast<double>(yi) - expect[i]);
    max_abs = std::max(max_abs, dd);
    sum2 += dd * dd;
  }
  std::cout << "bigvgan spike[" << target << " <- " << out_name
            << "]: N=" << n << "  max_abs=" << max_abs
            << "  rmse=" << std::sqrt(sum2 / n) << "\n";
  std::cout << "  first5 got: ";
  for (size_t i = 0; i < 5 && i < n; ++i)
    std::cout << y[i] << " ";
  std::cout << "\n  first5 exp: ";
  for (size_t i = 0; i < 5 && i < n; ++i)
    std::cout << expect[i] << " ";
  std::cout << "\n";
  bool ok = max_abs < 1e-3;
  std::cout << (ok ? "PASS" : "FAIL") << " (atol 1e-3)\n";
  return ok ? 0 : 2;
}

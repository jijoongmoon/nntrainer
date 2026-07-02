// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   spike_antialias.cpp
 * @brief  Micro-spike: run the antialiased_snake layer on the dumped BigVGAN
 *         stage5 and compare to the dumped activation_post (HF TorchActivation1d).
 *
 *   spike_antialias [in_dir=/tmp/antialias_spike]
 */

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

using ml::train::createLayer;
using LayerHandle = std::shared_ptr<ml::train::Layer>;

static std::string kv(const std::string &k, const std::string &v) {
  return k + "=" + v;
}

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

int main(int argc, char **argv) {
  const std::string dir = argc > 1 ? argv[1] : "/tmp/antialias_spike";
  const unsigned int B = 1, C = 24, T = 30720;

  auto &eng = nntrainer::Engine::Global();
  auto *app_context =
    static_cast<nntrainer::AppContext *>(eng.getRegisteredContext("cpu"));
  app_context->registerFactory(
    nntrainer::createLayer<causallm::AntialiasedSnakeLayer>);

  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  std::vector<LayerHandle> layers;
  layers.push_back(createLayer(
    "input", {kv("name", "input0"),
              kv("input_shape", std::to_string(C) + ":1:" + std::to_string(T))}));
  layers.push_back(createLayer(
    "antialiased_snake",
    {kv("name", "act_post"), kv("input_layers", "input0")}));
  for (auto &l : layers)
    model->addLayer(l);

  model->setProperty({kv("batch_size", "1"), kv("epochs", "1"),
                      kv("model_tensor_type", "FP32-FP32")});
  if (model->compile(ml::train::ExecutionMode::INFERENCE))
    throw std::runtime_error("compile failed");
  if (model->initialize(ml::train::ExecutionMode::INFERENCE))
    throw std::runtime_error("initialize failed");
  model->load(dir + "/weight.bin", ml::train::ModelFormat::MODEL_FORMAT_BIN);

  std::vector<float> in = read_bin(dir + "/input.bin", (size_t)B * C * T);
  std::vector<float> expect = read_bin(dir + "/expected.bin", (size_t)B * C * T);
  std::vector<float *> input{in.data()};
  std::vector<float *> label;
  std::vector<float *> out = model->inference(B, input, label);
  const float *y = out[0];

  const size_t n = (size_t)B * C * T;
  double max_abs = 0.0, sum2 = 0.0;
  for (size_t i = 0; i < n; ++i) {
    double d = std::abs(static_cast<double>(y[i]) - expect[i]);
    max_abs = std::max(max_abs, d);
    sum2 += d * d;
  }
  std::cout << "antialias spike: N=" << n << "  max_abs=" << max_abs
            << "  rmse=" << std::sqrt(sum2 / n) << "\n";
  std::cout << "  first5 got: ";
  for (int i = 0; i < 5; ++i)
    std::cout << y[i] << " ";
  std::cout << "\n  first5 exp: ";
  for (int i = 0; i < 5; ++i)
    std::cout << expect[i] << " ";
  std::cout << "\n";

  bool ok = max_abs < 1e-3;
  std::cout << (ok ? "PASS" : "FAIL") << " (atol 1e-3)\n";
  return ok ? 0 : 2;
}

// SPDX-License-Identifier: Apache-2.0
/**
 * Standalone validation driver for the real GatedDeltaNetLayer (LayerImpl),
 * checked against the P1 goldens (gdn_p0/bin) through the real nntrainer Layer
 * API (finalize + forwarding via InitLayerContext/RunLayerContext). No meson,
 * no AppContext registration — compiled against the prebuilt libnntrainer.so.
 */

#include <gated_delta_net_layer.h>

#include <layer_context.h>
#include <layer_devel.h>
#include <tensor.h>
#include <var_grad.h>
#include <weight.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace nntrainer;

static const std::string GDIR = "/home/aisjetson/jijoongmoon/gdn_p0/bin";
static std::vector<float> loadBin(const std::string &name) {
  std::ifstream f(GDIR + "/" + name + ".bin",
                  std::ios::in | std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + name);
  std::streamsize bytes = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<float> v(bytes / sizeof(float));
  f.read(reinterpret_cast<char *>(v.data()), bytes);
  return v;
}
// HF [out,in] -> nntrainer [in,out]
static std::vector<float> T2(const std::vector<float> &w, int out, int in) {
  std::vector<float> y(w.size());
  for (int o = 0; o < out; ++o)
    for (int i = 0; i < in; ++i)
      y[i * out + o] = w[o * in + i];
  return y;
}

// Finalize + forward a single-in/single-out LayerImpl with caller weights+input.
static Tensor runStandalone(std::unique_ptr<Layer> layer, const TensorDim &in_dim,
                            const std::vector<std::string> &props,
                            const std::vector<std::vector<float>> &weight_data,
                            const std::vector<float> &input_data) {
  layer->setProperty(props);
  InitLayerContext ic({in_dim}, {true}, false, "standalone", "", 0.0f,
                      {"NCHW", "FP32", "FP32"}, 1.0f,
                      ml::train::ExecutionMode::INFERENCE);
  layer->finalize(ic);

  std::vector<Weight> weights;
  std::vector<Var_Grad> inputs, outputs, tensors;
  for (auto &ws : ic.getWeightsSpec())
    weights.emplace_back(ws, true);
  for (auto &d : ic.getInputDimensions())
    inputs.emplace_back(d, Initializer::NONE, true, true, "in");
  for (auto &os : ic.getOutSpecs())
    outputs.emplace_back(os.variable_spec.dim, Initializer::NONE, true, true, "out");
  for (auto &ts : ic.getTensorsSpec())
    tensors.emplace_back(ts, true);

  if (weight_data.size() != weights.size())
    throw std::runtime_error("weight count mismatch: provided " +
                             std::to_string(weight_data.size()) + " vs requested " +
                             std::to_string(weights.size()));
  std::memcpy(inputs[0].getVariableRef().getData<float>(), input_data.data(),
              input_data.size() * sizeof(float));
  for (size_t i = 0; i < weights.size(); ++i) {
    auto &wt = weights[i].getVariableRef();
    if (weight_data[i].size() != wt.size())
      throw std::runtime_error("weight " + std::to_string(i) + " size " +
                               std::to_string(weight_data[i].size()) + " vs " +
                               std::to_string(wt.size()));
    std::memcpy(wt.getData<float>(), weight_data[i].data(),
                weight_data[i].size() * sizeof(float));
    weights[i].getGradientRef().setZero();
  }

  auto view = [](auto &vec) {
    std::vector<std::remove_reference_t<decltype(vec[0])> *> p;
    for (auto &e : vec)
      p.push_back(&e);
    return p;
  };
  RunLayerContext rc("standalone", true, 0.0f, false, 1.0f, nullptr, false,
                     view(weights), view(inputs), view(outputs), view(tensors));
  layer->forwarding(rc, false);
  return rc.getOutput(0);
}

int main() {
  // tiny GDN config (matches gdn_p0 goldens)
  const int B = 1, S = 8, HID = 32, NVH = 4, NKH = 2, HKD = 8, HVD = 8, KS = 4;
  const int KEY = HKD * NKH, VAL = HVD * NVH, CONV = KEY * 2 + VAL;

  std::vector<std::vector<float>> wd = {
    T2(loadBin("w_in_proj_qkv"), CONV, HID),  // [hidden, conv_dim]
    T2(loadBin("w_in_proj_z"), VAL, HID),
    T2(loadBin("w_in_proj_b"), NVH, HID),
    T2(loadBin("w_in_proj_a"), NVH, HID),
    loadBin("w_conv1d"),                       // [conv_dim,1,K] flat == [conv_dim,K]
    loadBin("A_log"),
    loadBin("dt_bias"),
    loadBin("w_norm"),
    T2(loadBin("w_out_proj"), HID, VAL),       // [value_dim, hidden]
  };
  std::vector<std::string> props = {
    "linear_num_value_heads=4", "linear_num_key_heads=2",
    "linear_key_head_dim=8", "linear_value_head_dim=8",
    "linear_conv_kernel_dim=4"};

  Tensor out = runStandalone(std::make_unique<causallm::GatedDeltaNetLayer>(),
                             TensorDim(B, 1, S, HID), props, wd, loadBin("hidden"));

  auto ref = loadBin("out");
  const float *po = out.getData<float>();
  float d = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i)
    d = std::max(d, std::fabs(po[i] - ref[i]));
  bool ok = d < 1e-5f;
  std::cout << "[GatedDeltaNetLayer] real LayerImpl forward vs P1 golden 'out': "
            << "max|d| = " << d << (ok ? "  [PASS]" : "  [FAIL]") << "\n";
  return ok ? 0 : 1;
}

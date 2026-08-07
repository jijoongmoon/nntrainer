// SPDX-License-Identifier: Apache-2.0
/**
 * Dumps the machine-readable weight manifest (positional load order + dtype +
 * dims) of a Qwen3_5MoeCausalLM built from a real model dir (config.json /
 * generation_config.json / nntr_config.json). The P4 repacker consumes this
 * as the ground truth for the bin layout.
 *
 * Usage: gdn_dump_manifest <model_dir> [weights_bin]
 * Output lines: W|<idx>|<weight_name>|<dtype>|<batch>|<channel>|<height>|<width>
 * With [weights_bin]: also positional-loads the bin after the dump (P4 load
 * validation — exercises the plain->Section-A QINT4 repack at read time).
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <json.hpp>
#include <layer_context.h>
#include <model.h>
#include <qwen3_5_moe_causallm.h>

using json = nlohmann::json;

static const char *dtypeName(ml::train::TensorDim::DataType t) {
  using DT = ml::train::TensorDim::DataType;
  switch (t) {
  case DT::FP32:
    return "FP32";
  case DT::FP16:
    return "FP16";
  case DT::QINT4:
    return "QINT4";
  case DT::QS4CX:
    return "QS4CX";
  case DT::QINT8:
    return "QINT8";
  case DT::QINT16:
    return "QINT16";
  case DT::Q4_K:
    return "Q4_K";
  case DT::Q6_K:
    return "Q6_K";
  case DT::Q4_0:
    return "Q4_0";
  case DT::UINT4:
    return "UINT4";
  case DT::UINT8:
    return "UINT8";
  case DT::UINT16:
    return "UINT16";
  case DT::UINT32:
    return "UINT32";
  default:
    return "UNKNOWN";
  }
}

// expose the protected ml::train::Model handle for weight introspection
struct ManifestModel : causallm::Qwen3_5MoeCausalLM {
  ManifestModel(json &c, json &g, json &n) :
    causallm::Transformer(c, g, n, causallm::ModelType::CAUSALLM),
    causallm::Qwen3_5MoeCausalLM(c, g, n) {}
  void dumpManifest() {
    int wi = 0;
    model->forEachLayer(
      [&](ml::train::Layer &l, nntrainer::RunLayerContext &rc, void *) {
        const unsigned nw = rc.getNumWeights();
        for (unsigned i = 0; i < nw; ++i) {
          auto &w = rc.getWeight(i);
          auto d = w.getDim();
          std::cout << "W|" << wi++ << "|" << rc.getWeightName(i) << "|"
                    << dtypeName(w.getDataType()) << "|" << d.batch() << "|"
                    << d.channel() << "|" << d.height() << "|" << d.width()
                    << "\n";
        }
      },
      nullptr);
    std::cout << "TOTAL|" << wi << "\n";
  }
};

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: gdn_dump_manifest <model_dir>\n";
    return 1;
  }
  const std::string dir = argv[1];
  auto readJson = [](const std::string &p) {
    std::ifstream f(p);
    if (!f)
      throw std::runtime_error("cannot open " + p);
    return json::parse(f);
  };
  json cfg = readJson(dir + "/config.json");
  json gen = readJson(dir + "/generation_config.json");
  json nntr = readJson(dir + "/nntr_config.json");
  nntr["skip_tokenizer"] = true; // headless: no tokenizer needed for manifest

  auto model = std::make_unique<ManifestModel>(cfg, gen, nntr);
  std::cerr << "[manifest] initialize()...\n";
  model->initialize();
  std::cerr << "[manifest] dumping...\n";
  model->dumpManifest();
  if (argc > 2) {
    std::cerr << "[manifest] load_weight(" << argv[2] << ")...\n";
    model->load_weight(argv[2]);
    std::cerr << "[manifest] LOAD OK\n";
  }
  return 0;
}

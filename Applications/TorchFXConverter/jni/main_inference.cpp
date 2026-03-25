// SPDX-License-Identifier: Apache-2.0
/**
 * @file   main_inference.cpp
 * @date   24 Mar 2026
 * @brief  Inference driver for TorchFXConverter-generated NNTrainer models.
 *
 * Loads converted weights, runs a single forward pass with given input,
 * and saves output logits to a binary file for comparison with Python.
 *
 * Usage:
 *   ./converter_model_inference \
 *       --weights model.bin \
 *       --input reference_input.bin \
 *       --output cpp_logits.bin \
 *       --seq-len 8
 */

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <model.h>

#include GENERATED_MODEL_HEADER

struct Args {
  std::string weight_path;
  std::string input_path;
  std::string output_path;
  unsigned int seq_len = 8;
  unsigned int vocab_size = 1000;
};

static Args parse_args(int argc, char *argv[]) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--weights" && i + 1 < argc)
      args.weight_path = argv[++i];
    else if (arg == "--input" && i + 1 < argc)
      args.input_path = argv[++i];
    else if (arg == "--output" && i + 1 < argc)
      args.output_path = argv[++i];
    else if (arg == "--seq-len" && i + 1 < argc)
      args.seq_len = std::stoul(argv[++i]);
    else if (arg == "--vocab-size" && i + 1 < argc)
      args.vocab_size = std::stoul(argv[++i]);
  }
  return args;
}

int main(int argc, char *argv[]) {
  try {
    auto args = parse_args(argc, argv);

    if (args.weight_path.empty() || args.input_path.empty() ||
        args.output_path.empty()) {
      std::cerr << "Usage: " << argv[0]
                << " --weights <path> --input <path> --output <path> "
                   "[--seq-len N] [--vocab-size N]"
                << std::endl;
      return 1;
    }

    // Step 1: Build and initialize model
    GENERATED_MODEL_CLASS model_builder;

    std::cout << "[inference] Initializing model (seq_len=" << args.seq_len
              << ")..." << std::endl;
    model_builder.initialize();

    auto &model = model_builder.getModel();
    std::cout << "[inference] Model initialized successfully." << std::endl;
    model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

    // Step 2: Load weights
    // Note: compile(Tensor, Tensor) already does compile+initialize+allocate
    std::cout << "[inference] Loading weights from: " << args.weight_path
              << std::endl;
    model->load(args.weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    std::cout << "[inference] Weights loaded." << std::endl;

    // Step 4: Read input data (float32 token IDs)
    // Input buffer sized for max_seq_len (INIT_SEQ_LEN from model)
    unsigned int input_len = args.seq_len;
    std::vector<float> input_data(input_len, 0.0f);
    {
      std::ifstream fin(args.input_path, std::ios::binary);
      if (!fin.is_open()) {
        throw std::runtime_error("Cannot open input file: " +
                                 args.input_path);
      }
      fin.read(reinterpret_cast<char *>(input_data.data()),
               input_len * sizeof(float));
      if (!fin) {
        throw std::runtime_error("Failed to read input data");
      }
    }

    std::cout << "[inference] Input tokens:";
    for (unsigned int i = 0; i < input_len; ++i) {
      std::cout << " " << static_cast<int>(input_data[i]);
    }
    std::cout << std::endl;

    // Step 5: Run incremental_inference (prefill mode)
    // This matches how CausalLM runs the prefill phase
    std::vector<float *> input_ptrs = {input_data.data()};
    std::vector<float *> label_ptrs;

    std::cout << "[inference] Running forward pass (prefill)..." << std::endl;
    auto output = model->incremental_inference(
      1,             // batch size
      input_ptrs,    // input
      label_ptrs,    // empty labels
      input_len,     // init_seq_len
      0,             // from
      input_len,     // to
      false          // output_hidden_state = false (return logits)
    );

    if (output.empty()) {
      throw std::runtime_error("Model inference returned empty output");
    }

    std::cout << "[inference] Forward pass completed." << std::endl;

    // Step 6: Compute output size and save logits
    // incremental_inference returns logits for the last position only
    // Output shape: (batch=1, vocab_size)
    unsigned int vocab_size = args.vocab_size;
    unsigned int total_output_size = vocab_size;

    std::cout << "[inference] Output: vocab_size=" << vocab_size
              << " (last position logits)" << std::endl;

    // Write output logits to binary file
    {
      std::ofstream fout(args.output_path, std::ios::binary);
      if (!fout.is_open()) {
        throw std::runtime_error("Cannot open output file: " +
                                 args.output_path);
      }
      fout.write(reinterpret_cast<const char *>(output[0]),
                 total_output_size * sizeof(float));
    }

    std::cout << "[inference] Saved " << total_output_size
              << " logit values to: " << args.output_path << std::endl;

    // Print first few and last few logit values for sanity check
    std::cout << "[inference] First 5 logits:";
    for (unsigned int i = 0; i < std::min(5u, total_output_size); ++i) {
      std::cout << " " << output[0][i];
    }
    std::cout << std::endl;

    std::cout << "[inference] Last 5 logits:";
    unsigned int start =
      total_output_size > 5 ? total_output_size - 5 : 0;
    for (unsigned int i = start; i < total_output_size; ++i) {
      std::cout << " " << output[0][i];
    }
    std::cout << std::endl;

    std::cout << "[inference] Done." << std::endl;
    return 0;

  } catch (const std::exception &e) {
    std::cerr << "[inference] FAILED: " << e.what() << std::endl;
    return 1;
  }
}

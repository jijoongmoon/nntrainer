// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>

 * @file   test_api.cpp
 * @date   21 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @brief  Simple application to test CausalLM API
 * @bug    No known bugs except for NYI items
 *
 */

#include "causal_lm_api.h"
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr const char *COLOR_RESET = "\033[0m";
constexpr const char *COLOR_BOLD = "\033[1m";
constexpr const char *COLOR_CYAN = "\033[36m";
constexpr const char *COLOR_GREEN = "\033[32m";
constexpr const char *COLOR_YELLOW = "\033[33m";
constexpr const char *COLOR_BLUE = "\033[34m";
constexpr const char *COLOR_RED = "\033[31m";
constexpr const char *COLOR_MAGENTA = "\033[35m";

void printLine(const std::string &s, int length = 80) {
  for (int i = 0; i < length; ++i)
    std::cout << s;
  std::cout << std::endl;
}

void printSection(const std::string &section) {
  std::cout << "\n"
            << COLOR_BOLD << COLOR_BLUE
            << "+-------------------------------------------------------------+"
            << COLOR_RESET << "\n";
  std::cout << COLOR_BOLD << COLOR_BLUE << "|  " << section
            << std::string(58 - section.length(), ' ') << "|" << COLOR_RESET
            << "\n";
  std::cout << COLOR_BOLD << COLOR_BLUE
            << "+-------------------------------------------------------------+"
            << COLOR_RESET << "\n\n";
}

void printSuccess(const std::string &msg) {
  std::cout << COLOR_GREEN << "✓ " << COLOR_BOLD << msg << COLOR_RESET
            << "\n\n";
}

void printError(const std::string &msg) {
  std::cerr << COLOR_RED << "✗ " << COLOR_BOLD << "Error: " << COLOR_RESET
            << msg << "\n";
}

void printWarning(const std::string &msg) {
  std::cout << COLOR_YELLOW << "⚠ " << msg << COLOR_RESET << "\n";
}

void printInfo(const std::string &label, const std::string &value) {
  std::cout << COLOR_CYAN << "  " << label << ":" << COLOR_RESET << " " << value
            << "\n";
}

void printLogo() {
  std::cout << "\n";
  std::cout << COLOR_BOLD << COLOR_MAGENTA;
  std::cout << "  ███╗   ██╗███╗   ██╗\n";
  std::cout << "  ████╗  ██║████╗  ██║\n";
  std::cout << "  ██╔██╗ ██║██╔██╗ ██║\n";
  std::cout << "  ██║╚██╗██║██║╚██╗██║\n";
  std::cout << "  ██║ ╚████║██║ ╚████║\n";
  std::cout << "  ╚═╝  ╚═══╝╚═╝  ╚═══╝\n";
  std::cout << COLOR_RESET;
  std::cout << COLOR_BOLD << COLOR_CYAN
            << "  ────────────────────────────────\n";
  std::cout << "      Causal Language Model API\n"
            << "  ────────────────────────────────\n";
  std::cout << COLOR_RESET << "\n";
}

void printUsage(const char *program_name) {
  std::cout << COLOR_YELLOW << "Usage:" << COLOR_RESET << "\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " <model_name> [prompt] [use_chat_template] [quantization]\n\n";

  std::cout << COLOR_CYAN << "Arguments:" << COLOR_RESET << "\n";
  std::cout << "  model_name        " << COLOR_BOLD << "REQUIRED" << COLOR_RESET
            << "  - Model name (e.g., QWEN3-0.6B-W4A32, QWEN3-0.6B)\n";
  std::cout << "  prompt            " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - Input prompt (default: 'Hello, how are you?')\n";
  std::cout << "  use_chat_template " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET << "  - 0/1 or true/false (default: 1)\n";
  std::cout << "  quantization      " << COLOR_GREEN << "OPTIONAL"
            << COLOR_RESET
            << "  - W4A32/W16A16/W8A16/W32A32/UNKNOWN (default: UNKNOWN)\n\n";

  std::cout << COLOR_YELLOW << "Examples:" << COLOR_RESET << "\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " QWEN3-0.6B-W4A32 \"Tell me a joke\"\n";
  std::cout << "  " << COLOR_BOLD << program_name << COLOR_RESET
            << " QWEN3-0.6B \"Write a poem\" 1 W4A32\n\n";
}
} // namespace

int main(int argc, char *argv[]) {
  printLogo();

  if (argc < 2) {
    printSection("ERROR: Missing Required Arguments");
    printUsage(argv[0]);
    return 1;
  }

  const char *model_name = argv[1];
  const char *prompt = (argc >= 3) ? argv[2] : "Hello, how are you?";
  bool use_chat_template = true;
  if (argc >= 4) {
    use_chat_template =
      (std::string(argv[3]) == "1" || std::string(argv[3]) == "true");
  }

  std::string quant_str = "UNKNOWN";
  ModelQuantizationType quant_type = CAUSAL_LM_QUANTIZATION_UNKNOWN;
  if (argc >= 5) {
    quant_str = std::string(argv[4]);
    if (quant_str == "W4A32")
      quant_type = CAUSAL_LM_QUANTIZATION_W4A32;
    else if (quant_str == "W16A16")
      quant_type = CAUSAL_LM_QUANTIZATION_W16A16;
    else if (quant_str == "W8A16")
      quant_type = CAUSAL_LM_QUANTIZATION_W8A16;
    else if (quant_str == "W32A32")
      quant_type = CAUSAL_LM_QUANTIZATION_W32A32;
  }

  printSection("Configuration");
  printInfo("Model Name", model_name);
  printInfo("Use Chat Template", use_chat_template ? "true" : "false");
  printInfo("Quantization", quant_str);
  std::cout << "\n";

  printSection("Initialization");
  std::cout << COLOR_CYAN << "⏳ " << COLOR_RESET << "Configuring options...\n";
  Config config;
  config.use_chat_template = use_chat_template;
  config.debug_mode = true;
  ErrorCode err = setOptions(config);
  if (err != CAUSAL_LM_ERROR_NONE) {
    printError("Failed to set options");
    std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    return 1;
  }
  printSuccess("Options configured successfully");

  printSection("Model Loading");
  std::cout << COLOR_CYAN << "⏳ " << COLOR_RESET
            << "Loading model: " << COLOR_BOLD << model_name << COLOR_RESET
            << "\n";
  err = loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_UNKNOWN, quant_type,
                  model_name);
  if (err != CAUSAL_LM_ERROR_NONE) {
    printError("Failed to load model");
    std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    return 1;
  }
  printSuccess("Model loaded successfully");

  printSection("Inference");
  std::cout << COLOR_CYAN << "📝 " << COLOR_RESET << "Input Prompt:\n";
  std::cout << COLOR_BOLD << COLOR_YELLOW << "  " << prompt << COLOR_RESET
            << "\n\n";

  std::cout << COLOR_CYAN << "⚡ " << COLOR_RESET << "Running inference...\n\n";

  const char *outputText = nullptr;
  err = runModel(prompt, &outputText);
  if (err != CAUSAL_LM_ERROR_NONE) {
    printError("Failed to run model");
    std::cerr << "  Error code: " << static_cast<int>(err) << "\n";
    return 1;
  }

  if (outputText) {
    std::cout << COLOR_CYAN << "💬 " << COLOR_RESET << "Output:\n";
    std::cout << COLOR_BOLD << COLOR_GREEN << "  ";
    std::string out(outputText);
    size_t pos = 0;
    while (pos < out.length()) {
      size_t newlinePos = out.find('\n', pos);
      if (newlinePos == std::string::npos) {
        newlinePos = out.length();
      }
      std::string line = out.substr(pos, newlinePos - pos);
      std::cout << line;
      if (newlinePos < out.length()) {
        std::cout << "\n  ";
        pos = newlinePos + 1;
      } else {
        pos = out.length();
      }
    }
    std::cout << COLOR_RESET << "\n\n";
  } else {
    printWarning("No output generated");
  }

  printSection("Performance Metrics");
  PerformanceMetrics metrics;
  err = getPerformanceMetrics(&metrics);
  if (err != CAUSAL_LM_ERROR_NONE) {
    printWarning("Failed to get metrics");
    std::cout << "  Error code: " << static_cast<int>(err) << "\n";
  } else {
    double prefill_tps =
      metrics.prefill_duration_ms > 0
        ? (metrics.prefill_tokens / metrics.prefill_duration_ms * 1000.0)
        : 0.0;
    double gen_tps =
      metrics.generation_duration_ms > 0
        ? (metrics.generation_tokens / metrics.generation_duration_ms * 1000.0)
        : 0.0;

    std::cout << COLOR_CYAN << "  📊 " << COLOR_RESET << COLOR_BOLD
              << "Prefill Stage" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "    Tokens:" << COLOR_RESET << "       "
              << metrics.prefill_tokens << "\n";
    std::cout << COLOR_CYAN << "    Duration:" << COLOR_RESET << "     "
              << std::fixed << std::setprecision(2)
              << metrics.prefill_duration_ms << " ms\n";
    std::cout << COLOR_CYAN << "    Throughput:" << COLOR_RESET << "   "
              << COLOR_BOLD << COLOR_GREEN << std::fixed << std::setprecision(1)
              << prefill_tps << COLOR_RESET << " tokens/sec\n\n";

    std::cout << COLOR_CYAN << "  📊 " << COLOR_RESET << COLOR_BOLD
              << "Generation Stage" << COLOR_RESET << "\n";
    std::cout << COLOR_CYAN << "    Tokens:" << COLOR_RESET << "       "
              << metrics.generation_tokens << "\n";
    std::cout << COLOR_CYAN << "    Duration:" << COLOR_RESET << "     "
              << std::fixed << std::setprecision(2)
              << metrics.generation_duration_ms << " ms\n";
    std::cout << COLOR_CYAN << "    Throughput:" << COLOR_RESET << "   "
              << COLOR_BOLD << COLOR_GREEN << std::fixed << std::setprecision(1)
              << gen_tps << COLOR_RESET << " tokens/sec\n\n";
  }

  printLine("═", 63);
  std::cout << COLOR_BOLD << COLOR_GREEN << "  ✓ Test completed successfully!"
            << COLOR_RESET << "\n";
  printLine("═", 63);
  std::cout << "\n";

  return 0;
}

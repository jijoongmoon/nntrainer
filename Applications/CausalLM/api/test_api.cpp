
#include "causal_lm_api.h"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [prompt]" << std::endl;
    return 1;
  }

  const char *model_path = argv[1];
  const char *prompt = (argc >= 3) ? argv[2] : "Hello, how are you?";
  bool use_chat_template = true;
  if (argc >= 4) {
    use_chat_template =
      (std::string(argv[3]) == "1" || std::string(argv[3]) == "true");
  }

  std::cout << "Loading model from: " << model_path << std::endl;
  std::cout << "Use chat template: " << (use_chat_template ? "true" : "false")
            << std::endl;

  // 1. Set Options (Optional)
  Config config;
  config.reserved = 0;
  config.use_chat_template = use_chat_template;
  ErrorCode err = setOptions(config);
  if (err != CAUSAL_LM_ERROR_NONE) {
    std::cerr << "Failed to set options: " << err << std::endl;
    return 1;
  }

  // 2. Load model
  // Auto-detect model type or assume one. For test, we pass UNKNOWN or a
  // specific one if known. The implementation falls back to config.json
  // architecture if provided.
  err = loadModel(CAUSAL_LM_BACKEND_CPU, CAUSAL_LM_MODEL_UNKNOWN, model_path);
  if (err != CAUSAL_LM_ERROR_NONE) {
    std::cerr << "Failed to load model: " << err << std::endl;
    return 1;
  }
  std::cout << "Model loaded successfully." << std::endl;

  // 3. Run Inference
  char outputBuffer[1024];
  std::cout << "Running inference with prompt: " << prompt << std::endl;

  err = runModel(prompt, outputBuffer, sizeof(outputBuffer));
  if (err != CAUSAL_LM_ERROR_NONE) {
    std::cerr << "Failed to run model: " << err << std::endl;
    return 1;
  }

  std::cout << "Output: " << outputBuffer << std::endl;

  return 0;
}

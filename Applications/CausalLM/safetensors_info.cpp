// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file   safetensors_info.cpp
 * @date   26 May 2026
 * @brief  Header-only inspector for nntrainer safetensors weight files.
 *         Prints the embedded metadata and a per-tensor table (including the
 *         native quantization type) without loading any weight data.
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @usage  nntr_safetensors_info <file.safetensors>
 */

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include <safetensors_util.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <file.safetensors>\n";
    return EXIT_FAILURE;
  }

  const std::string path = argv[1];
  std::ifstream file(path, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "error: cannot open file: " << path << "\n";
    return EXIT_FAILURE;
  }

  uint64_t header_size = 0;
  file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
  if (!file) {
    std::cerr << "error: cannot read safetensors header length\n";
    return EXIT_FAILURE;
  }

  std::string header_json(header_size, '\0');
  file.read(header_json.data(), static_cast<std::streamsize>(header_size));
  if (!file) {
    std::cerr << "error: cannot read safetensors header\n";
    return EXIT_FAILURE;
  }

  try {
    std::cout << "file: " << path << "\n";
    std::cout << "header bytes: " << header_size << "\n\n";
    std::cout << nntrainer::safetensors::inspect(header_json);
  } catch (const std::exception &e) {
    std::cerr << "error: failed to parse header: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

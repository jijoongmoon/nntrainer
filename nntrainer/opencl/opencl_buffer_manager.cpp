// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_buffer_manager.cpp
 * @date    01 Dec 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @author  Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains global Buffer objects and manages them
 */

#include <cstring>

#include <opencl_buffer_manager.h>
#include <opencl_loader.h>

namespace nntrainer {

void ClBufferManager::initBuffers() {
  // Lazy: allocate SVM only when the corresponding getSVM* accessor is first
  // called. Eagerly reserving ~205 MiB here (4×32 MiB + 3×1.5 MiB + 3×24 MiB)
  // exhausts the Adreno SVM heap before any test even starts, so subsequent
  // small allocateSVM calls from individual tests return null and the test
  // fixture throws. Sizing each slot on first use keeps tests that never
  // touch the int4/q4_0 paths within the device's SVM budget.
  scale_vec.assign(max_qs, nullptr);
  quant_vec.assign(max_qs, nullptr);
  output_vec.assign(max_qs, nullptr);
}

opencl::Buffer *ClBufferManager::getInBufferA() {
  if (inBufferA == nullptr) {
    inBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferA;
}

opencl::Buffer *ClBufferManager::getInBufferB() {
  if (inBufferB == nullptr) {
    inBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferB;
}

opencl::Buffer *ClBufferManager::getInBufferC() {
  if (inBufferC == nullptr) {
    inBufferC = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferC;
}

opencl::Buffer *ClBufferManager::getOutBufferA() {
  if (outBufferA == nullptr) {
    outBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  }
  return outBufferA;
}

opencl::Buffer *ClBufferManager::getOutBufferB() {
  if (outBufferB == nullptr) {
    outBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  }
  return outBufferB;
}

void *ClBufferManager::getSVMInput() {
  if (data_input == nullptr) {
    data_input = context_inst_.createSVMRegion(buffer_size_bytes);
  }
  return data_input;
}

void *ClBufferManager::getSVMScale(unsigned int idx) {
  if (idx >= scale_vec.size())
    return nullptr;
  if (scale_vec[idx] == nullptr) {
    scale_vec[idx] = context_inst_.createSVMRegion(scale_q4_0_size);
  }
  return scale_vec[idx];
}

void *ClBufferManager::getSVMQuant(unsigned int idx) {
  if (idx >= quant_vec.size())
    return nullptr;
  if (quant_vec[idx] == nullptr) {
    quant_vec[idx] = context_inst_.createSVMRegion(quant_q4_0_size);
  }
  return quant_vec[idx];
}

void *ClBufferManager::getSVMOutput(unsigned int idx) {
  if (idx >= output_vec.size())
    return nullptr;
  if (output_vec[idx] == nullptr) {
    output_vec[idx] = context_inst_.createSVMRegion(buffer_size_bytes);
  }
  return output_vec[idx];
}

ClBufferManager::~ClBufferManager() {
  if (inBufferA) {
    delete inBufferA;
  }
  if (inBufferB) {
    delete inBufferB;
  }
  if (inBufferC) {
    delete inBufferC;
  }
  if (outBufferA) {
    delete outBufferA;
  }
  if (outBufferB) {
    delete outBufferB;
  }

  if (data_input) {
    context_inst_.releaseSVMRegion(data_input);
  }
  for (auto &ptr : scale_vec) {
    context_inst_.releaseSVMRegion(ptr);
  }
  for (auto &ptr : quant_vec) {
    context_inst_.releaseSVMRegion(ptr);
  }
  for (auto &ptr : output_vec) {
    context_inst_.releaseSVMRegion(ptr);
  }
}

} // namespace nntrainer

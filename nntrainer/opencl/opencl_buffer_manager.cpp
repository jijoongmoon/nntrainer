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
#include <mutex>

#include <opencl_buffer_manager.h>
#include <opencl_loader.h>

namespace nntrainer {

namespace {
/**
 * @brief guards the lazy allocation below, which the getters may reach from
 *        a worker thread
 */
std::mutex &bufferMutex() {
  static std::mutex m;
  return m;
}
} // namespace

void ClBufferManager::initBuffers() {
  // Allocation is deferred to the first use of each region, in the getters
  // below. Allocating them here cost data_input plus max_qs x (scale + quant +
  // output) of shared memory -- a couple of hundred megabytes -- for every
  // process that initializes an OpenCL context, whether or not it ever runs a
  // block-quantized GEMM, which is the only thing these regions serve.
}

opencl::Buffer *ClBufferManager::getInBufferA() {
  std::lock_guard<std::mutex> lk(bufferMutex());
  if (inBufferA == nullptr) {
    inBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferA;
}

opencl::Buffer *ClBufferManager::getInBufferB() {
  std::lock_guard<std::mutex> lk(bufferMutex());
  if (inBufferB == nullptr) {
    inBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferB;
}

opencl::Buffer *ClBufferManager::getInBufferC() {
  std::lock_guard<std::mutex> lk(bufferMutex());
  if (inBufferC == nullptr) {
    inBufferC = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  }
  return inBufferC;
}

opencl::Buffer *ClBufferManager::getOutBufferA() {
  std::lock_guard<std::mutex> lk(bufferMutex());
  if (outBufferA == nullptr) {
    outBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  }
  return outBufferA;
}

opencl::Buffer *ClBufferManager::getOutBufferB() {
  std::lock_guard<std::mutex> lk(bufferMutex());
  if (outBufferB == nullptr) {
    outBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  }
  return outBufferB;
}

void *ClBufferManager::getSVMInput() {
  std::lock_guard<std::mutex> lk(bufferMutex());
  if (data_input == nullptr)
    data_input = context_inst_.createSVMRegion(buffer_size_bytes);
  return data_input;
}

void *ClBufferManager::getSVMScale(unsigned int idx) {
  if (idx >= max_qs)
    return nullptr;
  std::lock_guard<std::mutex> lk(bufferMutex());
  while (scale_vec.size() <= idx)
    scale_vec.push_back(context_inst_.createSVMRegion(scale_q4_0_size));
  return scale_vec[idx];
}

void *ClBufferManager::getSVMQuant(unsigned int idx) {
  if (idx >= max_qs)
    return nullptr;
  std::lock_guard<std::mutex> lk(bufferMutex());
  while (quant_vec.size() <= idx)
    quant_vec.push_back(context_inst_.createSVMRegion(quant_q4_0_size));
  return quant_vec[idx];
}

void *ClBufferManager::getSVMOutput(unsigned int idx) {
  if (idx >= max_qs)
    return nullptr;
  std::lock_guard<std::mutex> lk(bufferMutex());
  while (output_vec.size() <= idx)
    output_vec.push_back(context_inst_.createSVMRegion(buffer_size_bytes));
  return output_vec[idx];
}

void ClBufferManager::releaseBuffers() {
  delete inBufferA;
  inBufferA = nullptr;
  delete inBufferB;
  inBufferB = nullptr;
  delete inBufferC;
  inBufferC = nullptr;
  delete outBufferA;
  outBufferA = nullptr;
  delete outBufferB;
  outBufferB = nullptr;

  if (data_input) {
    context_inst_.releaseSVMRegion(data_input);
    data_input = nullptr;
  }
  for (auto &ptr : scale_vec) {
    context_inst_.releaseSVMRegion(ptr);
  }
  scale_vec.clear();
  for (auto &ptr : quant_vec) {
    context_inst_.releaseSVMRegion(ptr);
  }
  quant_vec.clear();
  for (auto &ptr : output_vec) {
    context_inst_.releaseSVMRegion(ptr);
  }
  output_vec.clear();
}

ClBufferManager::~ClBufferManager() { releaseBuffers(); }

} // namespace nntrainer

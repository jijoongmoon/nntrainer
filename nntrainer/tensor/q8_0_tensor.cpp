// SPDX-License-Identifier: Apache-2.0
/**
 * @file        q8_0_tensor.cpp
 * @date        14 May 2026
 * @brief       Q8_0_Tensor implementation (mirror of q4_0_tensor.cpp).
 * @see         https://github.com/nntrainer/nntrainer
 * @author      Claude
 * @bug         No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <q8_0_tensor.h>
#include <tensor.h>

namespace nntrainer {

Q8_0_Tensor::Q8_0_Tensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm) {
  offset = 0;
}

Q8_0_Tensor::Q8_0_Tensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name) :
  TensorBase(d, false, init, name) {
  NNTR_THROW_IF(d.batch() != 1 || d.channel() != 1 || d.width() % QK8_0 != 0,
                std::invalid_argument)
    << "Q8_0_Tensor must be 2-dimensional with batch=1, channel=1 and "
       "width divisible by " << QK8_0;

  if (alloc_now)
    allocate();
  offset = 0;
}

Q8_0_Tensor::Q8_0_Tensor(const TensorDim &d, const void *buf) :
  Q8_0_Tensor(d, true, Initializer::NONE, "") {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy_q80(buf);
  }
}

void Q8_0_Tensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    allocateSrcTensor();
  } else {
    MemoryData *mem_data = new MemoryData((void *)(new uint8_t[size()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<uint8_t>();
      delete mem_data;
    });
    offset = 0;
    initialize();
  }
}

void *Q8_0_Tensor::getData() const {
  if (!data)
    return nullptr;
  data->validate();
  return data->getAddr<uint8_t>() + offset;
}

size_t Q8_0_Tensor::size() const {
  size_t num_blocks = height() * width() / QK8_0;
  return Q8_0_SIZE * num_blocks;
}

size_t Q8_0_Tensor::getMemoryBytes() const { return size() * sizeof(uint8_t); }

void Q8_0_Tensor::copy_q80(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData())
    return;
  scopy(size(), (uint8_t *)buf, 1, (uint8_t *)getData(), 1);
}

void Q8_0_Tensor::setZero() {
  uint8_t *d = (uint8_t *)getData();
  std::fill(d, d + size(), 0);
}

void Q8_0_Tensor::initialize() {
  if (empty() || !isAllocated())
    return;
  setZero();
  putData();
}

QScheme Q8_0_Tensor::q_scheme() const { return QScheme::Q8_0; }

} // namespace nntrainer

// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cl_tensor_view.cpp
 * @date    20 May 2026
 * @brief   TensorBacking implementation. Single cl_mem + cached zero-copy
 *          image2d_from_buffer views.
 */
#include "cl_tensor_view.h"

#include <stdexcept>
#include <string>

namespace nntrainer::tv {

TensorBacking::TensorBacking(cl_context ctx, cl_mem buf, Encoding enc,
                             Layout lay, size_t bytes, bool owned)
  : ctx_(ctx), buf_(buf), enc_(enc), lay_(lay), bytes_(bytes), owned_(owned) {}

TensorBacking::~TensorBacking() {
  for (auto &kv : image_cache_) {
    if (kv.second)
      clReleaseMemObject(kv.second);
  }
  if (owned_ && buf_)
    clReleaseMemObject(buf_);
}

cl_mem TensorBacking::imageView(const ViewSpec &spec) {
  if (!spec.as_image)
    return buf_;

  auto it = image_cache_.find(spec);
  if (it != image_cache_.end())
    return it->second;

  cl_image_format fmt{};
  fmt.image_channel_order = spec.image_channel_order;
  fmt.image_channel_data_type = spec.image_channel_type;

  cl_image_desc desc{};
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = spec.width;
  desc.image_height = spec.height;
  desc.image_row_pitch = spec.row_pitch_bytes;
  desc.buffer = buf_; // zero-copy view over the same cl_mem (image2d_from_buffer)

  cl_int err = CL_SUCCESS;
  cl_mem img = clCreateImage(ctx_, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
  if (err != CL_SUCCESS || img == nullptr) {
    throw std::runtime_error("TensorBacking::imageView clCreateImage failed: " +
                             std::to_string(err));
  }
  image_cache_.emplace(spec, img);
  return img;
}

} // namespace nntrainer::tv

// SPDX-License-Identifier: Apache-2.0
/**
 * @file    cl_tensor_view.h
 * @date    20 May 2026
 * @brief   Paper-style tensor virtualization (minimal): a single cl_mem
 *          backing + multiple zero-copy views (buffer + image2d_from_buffer).
 *          Designed to grow: more Encoding/Layout values added as new ops
 *          come online (attention/RMSNorm/SwiGLU/etc.).
 *
 * Background: arXiv:2505.00232 (ML Drift) §3.1 — tensor virtualization
 * decouples the physical allocation from the per-op view (image1D / image2D /
 * image3D / buffer). We start with image2d_from_buffer over a single cl_mem
 * for v8c FC; extend as needed.
 */
#ifndef __NNTRAINER_CL_TENSOR_VIEW_H__
#define __NNTRAINER_CL_TENSOR_VIEW_H__

#include <CL/cl.h>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace nntrainer::tv {

/**
 * @brief In-memory data encoding. Bits at rest.
 */
enum class Encoding : uint8_t {
  FP32,
  FP16,
  INT8,
  INT4_OFFSET, ///< 4-bit values stored as (value+8) in low 4 bits of each byte
  INT4_2COMP,  ///< 4-bit values in 2's complement (low 4 bits of each byte)
  // grow as needed
};

/**
 * @brief Element axis order / packing convention.
 */
enum class Layout : uint8_t {
  ROW_MAJOR,   ///< [outer][inner] linear (e.g. [N][K] for weight, [M][K] for act)
  OSV32_ISV2,  ///< existing Int4QTensor disk layout
  PHWC4,       ///< paper §3.1 activation layout (future)
  OHWI,        ///< paper §3.8 attention K-cache layout (future)
};

/**
 * @brief Image view spec. as_image=false means raw buffer view (free).
 *        image_format + width + height + row_pitch describe image2d view.
 */
struct ViewSpec {
  bool as_image = false;
  cl_channel_order image_channel_order = CL_RGBA;
  cl_channel_type image_channel_type = CL_UNSIGNED_INT32;
  size_t width = 0;
  size_t height = 0;
  size_t row_pitch_bytes = 0;

  bool operator==(const ViewSpec &o) const {
    return as_image == o.as_image &&
           image_channel_order == o.image_channel_order &&
           image_channel_type == o.image_channel_type &&
           width == o.width && height == o.height &&
           row_pitch_bytes == o.row_pitch_bytes;
  }
};
struct ViewSpecHash {
  size_t operator()(const ViewSpec &s) const noexcept {
    size_t h = s.as_image;
    h = h * 131 + s.image_channel_order;
    h = h * 131 + s.image_channel_type;
    h = h * 131 + s.width;
    h = h * 131 + s.height;
    h = h * 131 + s.row_pitch_bytes;
    return h;
  }
};

/**
 * @brief One physical cl_mem allocation + metadata + cache of zero-copy views.
 *        Buffer view is always available; image views are created on first
 *        request via image2d_from_buffer and cached.
 */
class TensorBacking {
public:
  /**
   * @brief Construct with an existing cl_mem buffer (TensorBacking takes
   *        ownership unless owned=false).
   */
  TensorBacking(cl_context ctx, cl_mem buf, Encoding enc, Layout lay,
                size_t bytes, bool owned = true);
  ~TensorBacking();

  TensorBacking(const TensorBacking &) = delete;
  TensorBacking &operator=(const TensorBacking &) = delete;

  /// raw buffer view (zero-copy, always available)
  cl_mem buffer() const { return buf_; }
  /// image2d view over the same backing (zero-copy via image2d_from_buffer)
  cl_mem imageView(const ViewSpec &spec);

  Encoding encoding() const { return enc_; }
  Layout layout() const { return lay_; }
  size_t bytes() const { return bytes_; }

private:
  cl_context ctx_;
  cl_mem buf_;
  Encoding enc_;
  Layout lay_;
  size_t bytes_;
  bool owned_;
  std::unordered_map<ViewSpec, cl_mem, ViewSpecHash> image_cache_;
};

} // namespace nntrainer::tv

#endif // __NNTRAINER_CL_TENSOR_VIEW_H__

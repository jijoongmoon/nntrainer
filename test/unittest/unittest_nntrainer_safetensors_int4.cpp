// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 nntrainer authors
 *
 * @file   unittest_nntrainer_safetensors_int4.cpp
 * @date   Apr 2026
 * @brief  Smoke tests for the safetensors schema_version 2 path with
 *         per-channel int4 weights. Validates the P2/P3a/P4/P5 plumbing
 *         end-to-end at the structural level: a NeuralNetwork with a
 *         FullyConnected(weight_dtype=QINT4) layer can be saved, the
 *         resulting file carries the new "quant" metadata object, and
 *         a fresh NN with the same topology can load it back without
 *         the strict dtype/quant validation throwing.
 *
 * @note   Forward numerical correctness is intentionally NOT tested
 *         here. The CPU int4 dispatch in FloatTensor::dotQInteger
 *         currently passes Int4QTensor's canonical (raw packed nibbles
 *         + contiguous fp16 scales) layout to the KleidiAI
 *         nntr_gemm_qai8dxp_qsi4cxp_packed variant, which expects a
 *         KleidiAI-specific pre-packed layout. The two layouts are
 *         incompatible and a numerical test would either segfault on
 *         x86 (where the fallback is NYI throw) or produce garbage on
 *         ARM. That mismatch is tracked as a separate follow-up
 *         ("fix int4 dispatch to use the unpacked variant with a
 *         separate fp32 scale buffer or repack Int4QTensor into
 *         KleidiAI-packed layout on first use") and is outside the
 *         scope of P6.
 *
 * @see    https://github.com/nntrainer/nntrainer
 */

#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include <layer.h>
#include <model.h>
#include <neuralnet.h>
#include <optimizer.h>
#include <quantizer.h>
#include <tensor.h>
#include <tensor_dim.h>

using TensorDim = ml::train::TensorDim;
using DataType = TensorDim::DataType;
using ModelFormat = ml::train::ModelFormat;

namespace {

/**
 * @brief Build an initialized Input + FC NeuralNetwork whose FC weight
 *        is QINT4. The FC has no bias so there is exactly one weight
 *        tensor in the saved file, which keeps the structural check
 *        in the test body simple. The `weight_initializer=ones`
 *        property makes initialization deterministic (Int4QTensor
 *        only supports ZEROS/ONES).
 *
 * @param K number of input features (weight height)
 * @param N number of output features (weight width)
 */
std::unique_ptr<nntrainer::NeuralNetwork>
buildInt4FcNN(unsigned int K, unsigned int N) {
  auto nn = std::make_unique<nntrainer::NeuralNetwork>();

  nn->addLayer(ml::train::layer::Input(
    {"name=input", "input_shape=1:1:" + std::to_string(K)}));

  nn->addLayer(ml::train::layer::FullyConnected({
    "name=dense",
    "unit=" + std::to_string(N),
    "weight_dtype=QINT4",
    "disable_bias=true",
    "weight_initializer=ones",
  }));

  nn->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn->setProperty({"loss=mse", "batch_size=1"});

  nn->compile();
  nn->initialize();
  return nn;
}

/**
 * @brief Read the entire contents of a file into a std::string. Used
 *        to inspect the safetensors header bytes for regression
 *        checks (e.g. "schema_version\":\"2" must be present, the
 *        quant object must be emitted for the QINT4 weight).
 */
std::string readFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    return {};
  std::ostringstream oss;
  oss << f.rdbuf();
  return oss.str();
}

} // namespace

// =============================================================================
// P6 smoke tests: safetensors schema_version 2 + per-channel int4 roundtrip
// =============================================================================

/**
 * @brief Saving a NeuralNetwork with a QINT4 FC weight must produce a
 *        safetensors file. Validates the full P2 save path:
 *        - NeuralNetwork::save(SAFETENSORS) does not throw when the
 *          graph contains a quantized weight tensor (which was broken
 *          before P5 because FC::finalize wired a PER_TENSOR_AFFINE
 *          quantizer that then collided with float_tensor.cpp's
 *          PER_CHANNEL-only dispatch in Tensor::dot).
 *        - The resulting file is non-empty and has the magic 8-byte
 *          little-endian header_size prefix.
 */
TEST(SafetensorsInt4, SaveQINT4Fc_smoke) {
  const std::string path = "test_int4_save.safetensors";
  const unsigned int K = 8;
  const unsigned int N = 8;

  auto nn = buildInt4FcNN(K, N);
  ASSERT_NO_THROW(
    nn->save(path, ModelFormat::MODEL_FORMAT_SAFETENSORS));

  std::ifstream f(path, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(f.is_open())
    << "save() claimed success but the file is missing: " << path;
  auto file_size = f.tellg();
  f.close();

  // Minimum: 8-byte header_size + JSON header (at least a few dozen
  // bytes for the __metadata__ + one tensor entry) + weight bytes.
  EXPECT_GT(file_size, static_cast<std::streampos>(8 + 32))
    << "safetensors file unexpectedly small";

  std::remove(path.c_str());
}

/**
 * @brief The safetensors file written by P2 for a QINT4 weight must
 *        carry schema_version 2 and a "quant" object on the weight
 *        entry. This is a textual regression check so we can catch
 *        silent drops of the quant metadata (e.g. if a future commit
 *        accidentally reverts neuralnet.cpp::save's quant emission).
 */
TEST(SafetensorsInt4, SaveQINT4Fc_header_contains_quant) {
  const std::string path = "test_int4_header.safetensors";
  const unsigned int K = 8;
  const unsigned int N = 8;

  auto nn = buildInt4FcNN(K, N);
  ASSERT_NO_THROW(
    nn->save(path, ModelFormat::MODEL_FORMAT_SAFETENSORS));

  // Read the whole file (small) and spot-check the header text. The
  // JSON header lives immediately after the 8-byte header_size field
  // and is readable as-is.
  std::string blob = readFile(path);
  ASSERT_GE(blob.size(), static_cast<size_t>(8));

  // Convert the first 8 bytes to a size_t to find where the JSON ends.
  uint64_t hdr_size = 0;
  std::memcpy(&hdr_size, blob.data(), sizeof(hdr_size));
  ASSERT_LE(static_cast<size_t>(8 + hdr_size), blob.size());

  std::string header_json = blob.substr(8, hdr_size);

  EXPECT_NE(header_json.find("\"format\":\"nntrainer\""), std::string::npos)
    << "missing nntrainer format tag in __metadata__";
  EXPECT_NE(header_json.find("\"schema_version\":\"2\""), std::string::npos)
    << "schema_version 2 must be emitted by P2";
  EXPECT_NE(header_json.find("\"dtype\":\"I4\""), std::string::npos)
    << "QINT4 dtype string missing from header";
  EXPECT_NE(header_json.find("\"quant\""), std::string::npos)
    << "per-entry quant object missing for QINT4 weight";
  EXPECT_NE(header_json.find("\"encoding\":\"axis_scale_offset\""),
            std::string::npos)
    << "axis_scale_offset encoding missing";
  EXPECT_NE(header_json.find("\"bitwidth\":4"), std::string::npos)
    << "bitwidth=4 missing in quant object";
  // axis=1: for nntrainer's [K, N] FC weight layout, per-channel means
  // per-output-column (width), not per-input-row (height). Int4QTensor::
  // scale_size() returns width() and the save path emits axis=1.
  EXPECT_NE(header_json.find("\"axis\":1"), std::string::npos)
    << "axis=1 (per-output-column) missing in quant object";

  std::remove(path.c_str());
}

/**
 * @brief Full round-trip: save a NN with QINT4 FC weight, build a
 *        fresh NN with the same topology, load the file. The
 *        P3a loader must:
 *        - parse __metadata__.schema_version="2" (not throw on
 *          unknown versions);
 *        - emit the file tensor list + C++ weight list debug dumps;
 *        - exact-match the "dense:weight" name;
 *        - pass strict dtype validation (file "I4" == C++ QINT4);
 *        - accept the file's quant object against the C++ model's
 *          PER_CHANNEL_AFFINE Int4QTensor (c_is_quant == true, no
 *          mismatch throw).
 *        None of these should raise an exception for this simple
 *        one-FC topology.
 */
TEST(SafetensorsInt4, SaveThenLoadQINT4Fc_roundtrip) {
  const std::string path = "test_int4_roundtrip.safetensors";
  const unsigned int K = 8;
  const unsigned int N = 8;

  auto nn1 = buildInt4FcNN(K, N);
  ASSERT_NO_THROW(
    nn1->save(path, ModelFormat::MODEL_FORMAT_SAFETENSORS));

  auto nn2 = buildInt4FcNN(K, N);
  ASSERT_NO_THROW(
    nn2->load(path, ModelFormat::MODEL_FORMAT_SAFETENSORS));

  std::remove(path.c_str());
}

// =============================================================================
// Q4_0 regression tests: FC layer must keep working for Q4_0 weights, which
// live on Lane B (GGML-style block-quantized) and go through
// FloatTensor::dotQnK / gemm_q4_0_fp32 rather than the int4 dispatch path
// touched by P5/P6b. These tests guard against accidental breakage of the
// Q4_0 code path when the QINT4 branches are refactored.
// =============================================================================

namespace {

/**
 * @brief Build an initialized Input + FC NeuralNetwork whose FC weight
 *        is Q4_0. Q4_0_Tensor requires batch=1, channel=1 and
 *        `width() % 32 == 0`, so `N` must be a multiple of 32.
 *        The Q4_0 block format carries its fp16 scale embedded in
 *        each 18-byte block, so the layer-side quantizer on FC stays
 *        null (old `default:` branch, unchanged by P5).
 */
std::unique_ptr<nntrainer::NeuralNetwork>
buildQ4_0FcNN(unsigned int K, unsigned int N) {
  auto nn = std::make_unique<nntrainer::NeuralNetwork>();
  nn->addLayer(ml::train::layer::Input(
    {"name=input", "input_shape=1:1:" + std::to_string(K)}));
  nn->addLayer(ml::train::layer::FullyConnected({
    "name=dense",
    "unit=" + std::to_string(N),
    "weight_dtype=Q4_0",
    "disable_bias=true",
    "weight_initializer=ones",
  }));
  nn->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
  nn->setProperty({"loss=mse", "batch_size=1"});
  nn->compile();
  nn->initialize();
  return nn;
}

} // namespace

/**
 * @brief Building and initializing an FC layer with Q4_0 weight dtype
 *        must succeed. Validates that P5's QINT4-specific branch in
 *        FullyConnectedLayer::finalize did not accidentally null out
 *        Q4_0 allocation or selection.
 */
TEST(SafetensorsQ4_0, BuildFcWithQ4_0Weight_smoke) {
  const unsigned int K = 16;
  const unsigned int N = 32; // must be divisible by 32 for Q4_0_Tensor
  ASSERT_NO_THROW({ auto nn = buildQ4_0FcNN(K, N); });
}

/**
 * @brief Save an FC layer whose weight is Q4_0 and verify that
 *        neuralnet.cpp's safetensors writer emits the Q4_0 quant
 *        encoding (schema_version 2). This covers the Lane B
 *        dtype_str + deriveQuantInfo additions from commit 807f18d.
 */
TEST(SafetensorsQ4_0, SaveQ4_0Fc_header_contains_q4_0_quant) {
  const std::string path = "test_q4_0_header.safetensors";
  const unsigned int K = 16;
  const unsigned int N = 32;

  auto nn = buildQ4_0FcNN(K, N);
  ASSERT_NO_THROW(
    nn->save(path, ModelFormat::MODEL_FORMAT_SAFETENSORS));

  std::string blob = readFile(path);
  ASSERT_GE(blob.size(), static_cast<size_t>(8));

  uint64_t hdr_size = 0;
  std::memcpy(&hdr_size, blob.data(), sizeof(hdr_size));
  ASSERT_LE(static_cast<size_t>(8 + hdr_size), blob.size());
  std::string header_json = blob.substr(8, hdr_size);

  EXPECT_NE(header_json.find("\"schema_version\":\"2\""), std::string::npos);
  EXPECT_NE(header_json.find("\"dtype\":\"Q4_0\""), std::string::npos)
    << "Q4_0 dtype string missing from header";
  EXPECT_NE(header_json.find("\"quant\""), std::string::npos)
    << "per-entry quant object missing for Q4_0 weight";
  EXPECT_NE(header_json.find("\"encoding\":\"q4_0\""), std::string::npos)
    << "q4_0 encoding missing in quant object";
  EXPECT_NE(header_json.find("\"bitwidth\":4"), std::string::npos)
    << "bitwidth=4 missing in quant object for Q4_0";
  // group_size for Q4_0 is the GGML block size, 32.
  EXPECT_NE(header_json.find("\"group_size\":32"), std::string::npos)
    << "group_size=32 missing in quant object for Q4_0";

  std::remove(path.c_str());
}

/**
 * @brief Full Q4_0 save/load round-trip. Identical topology on both
 *        sides; P3a strict dtype validation must accept "Q4_0" ==
 *        Q4_0 and the quant object must not be flagged as a mismatch
 *        against a quantized C++ dtype.
 */
TEST(SafetensorsQ4_0, SaveThenLoadQ4_0Fc_roundtrip) {
  const std::string path = "test_q4_0_roundtrip.safetensors";
  const unsigned int K = 16;
  const unsigned int N = 32;

  auto nn1 = buildQ4_0FcNN(K, N);
  ASSERT_NO_THROW(
    nn1->save(path, ModelFormat::MODEL_FORMAT_SAFETENSORS));

  auto nn2 = buildQ4_0FcNN(K, N);
  ASSERT_NO_THROW(
    nn2->load(path, ModelFormat::MODEL_FORMAT_SAFETENSORS));

  std::remove(path.c_str());
}

/**
 * @brief Loading a safetensors file whose dtype contradicts the C++
 *        model must fail loudly (strict P3a dtype validation). We
 *        simulate this by saving as FP32 and trying to load into a
 *        QINT4 model with the same topology, then save as QINT4 and
 *        trying to load into a FP32 model. Either direction should
 *        throw with a dtype mismatch diagnostic.
 */
TEST(SafetensorsInt4, DtypeMismatch_throws) {
  const std::string path = "test_int4_mismatch.safetensors";
  const unsigned int K = 8;
  const unsigned int N = 8;

  // Save a FP32 FC weight (default weight_dtype when not overridden)
  auto build_fp32 = [&]() {
    auto nn = std::make_unique<nntrainer::NeuralNetwork>();
    nn->addLayer(ml::train::layer::Input(
      {"name=input", "input_shape=1:1:" + std::to_string(K)}));
    nn->addLayer(ml::train::layer::FullyConnected({
      "name=dense",
      "unit=" + std::to_string(N),
      "disable_bias=true",
      "weight_initializer=ones",
    }));
    nn->setOptimizer(ml::train::optimizer::SGD({"learning_rate=0.1"}));
    nn->setProperty({"loss=mse", "batch_size=1"});
    nn->compile();
    nn->initialize();
    return nn;
  };

  // FP32 save -> QINT4 load  (must throw)
  {
    auto nn_save = build_fp32();
    ASSERT_NO_THROW(
      nn_save->save(path, ModelFormat::MODEL_FORMAT_SAFETENSORS));
    auto nn_load = buildInt4FcNN(K, N);
    EXPECT_THROW(
      nn_load->load(path, ModelFormat::MODEL_FORMAT_SAFETENSORS),
      std::runtime_error)
      << "loading FP32 file into a QINT4 model should throw";
    std::remove(path.c_str());
  }

  // QINT4 save -> FP32 load  (must throw)
  {
    auto nn_save = buildInt4FcNN(K, N);
    ASSERT_NO_THROW(
      nn_save->save(path, ModelFormat::MODEL_FORMAT_SAFETENSORS));
    auto nn_load = build_fp32();
    EXPECT_THROW(
      nn_load->load(path, ModelFormat::MODEL_FORMAT_SAFETENSORS),
      std::runtime_error)
      << "loading QINT4 file into a FP32 model should throw";
    std::remove(path.c_str());
  }
}

// =============================================================================
// P6b Part B: numerical end-to-end dispatch test for FloatTensor::dotQInteger
// -----------------------------------------------------------------------------
// These tests exercise the CPU KleidiAI qsi4cxp_unpacked path that P5/P6b
// wired into FloatTensor::dotQInteger. They sit at the Tensor API level (no
// NeuralNetwork / FC layer) so the validation is scoped to:
//
//   1. Tensor::dot on FP32 activation x QINT4 weight reaches
//      FloatTensor::dotQInteger (dispatch switch case).
//   2. dotQInteger picks the PER_CHANNEL_AFFINE branch and forwards the
//      raw packed nibble buffer + fp32 scale buffer to
//      nntr_gemm_qai8dxp_qsi4cxp_unpacked without layout translation.
//   3. The P6b transB=false direction matches nntrainer's [K, N] (kxn)
//      weight storage convention.
//
// They do NOT test Int4QTensor::setValue / getValue semantic compatibility
// with KleidiAI. Int4QTensor's setValue packs values as two's-complement
// signed nibbles with the even flat-index in the HIGH nibble, whereas
// KleidiAI's qsi4cxp kxn kernel reads bytes as offset-binary unsigned with
// zero_point=8 and the even n_idx in the LOW nibble. These are genuinely
// incompatible conventions — a separate follow-up ([P6b-2]) must either
// translate at save/load time or rewrite Int4QTensor to natively use
// KleidiAI's convention. To sidestep that unresolved question, these tests
// write the raw packed bytes directly via getData<uint8_t>() in
// KleidiAI-native format (offset-binary, low nibble for even n_idx) and
// set scales directly via getScale<float>().
//
// Gated on ENABLE_FP16 because the CPU KleidiAI dispatch in
// FloatTensor::dotQInteger is currently itself inside `#ifdef ENABLE_FP16`
// (tracked as [C4]).
// =============================================================================

#ifdef ENABLE_FP16

namespace {

/**
 * @brief Allocate a QINT4 nntrainer::Tensor with canonical [1, 1, K, N]
 *        shape, then overwrite its packed-nibble data and fp32 scale
 *        buffer with KleidiAI-native kxn bytes so the KleidiAI kernel
 *        sees uniform real weight `real_value` with uniform scale 1.0
 *        for every output column.
 *
 * Byte layout written:
 *   - K rows x ((N+1)/2) bytes per row, row-major
 *   - For each (k, n): offset = k * ((N+1)/2) + n/2
 *   - Even n_idx -> LOW nibble, odd n_idx -> HIGH nibble
 *   - Stored nibble = (real_value + 8) & 0xf   (offset-binary, zp=8)
 *
 * The scale buffer is a contiguous fp32 array of length
 * Int4QTensor::scale_size(), filled with 1.0f. For a [1,1,K,N] tensor
 * with group_size==0 (pure per-channel) scale_size() returns height() =
 * K, which over-allocates vs KleidiAI's per-N-output expectation for
 * K != N but is harmless with uniform scales. The tests use K == N to
 * keep the count consistent on both sides; mixed-size cases need the
 * [P6b-2] convention fix first.
 */
nntrainer::Tensor makeKleidiAiInt4Weight(unsigned int K, unsigned int N,
                                         int real_value) {
  // Construct a QINT4 tensor on the Tensor API (this is the same path
  // FC layer / safetensors loader exercise internally).
  nntrainer::TensorDim dim(1, 1, K, N,
                            {nntrainer::Tformat::NCHW,
                             nntrainer::Tdatatype::QINT4});
  nntrainer::Tensor weight(dim, true, nntrainer::Initializer::ZEROS, "w",
                           nntrainer::QScheme::PER_CHANNEL_AFFINE);

  // Overwrite the raw packed-nibble buffer with KleidiAI kxn bytes.
  uint8_t *data = weight.getData<uint8_t>();
  const size_t row_stride = (N + 1) / 2;
  const uint8_t stored_nibble =
    static_cast<uint8_t>((real_value + 8) & 0xf); // offset-binary, zp=8
  // Pack both nibbles of each byte to the same stored value.
  const uint8_t packed_byte =
    static_cast<uint8_t>((stored_nibble << 4) | stored_nibble);
  std::memset(data, packed_byte, K * row_stride);

  // Overwrite the fp32 scale buffer with uniform 1.0. getScale<float>()
  // returns the same pointer allocate() uses, so this is a direct
  // in-place write (no re-allocation needed).
  float *scales = weight.getScale<float>();
  const size_t n_scales = weight.scale_size();
  for (size_t i = 0; i < n_scales; ++i)
    scales[i] = 1.0f;

  return weight;
}

} // namespace

/**
 * @brief Weight = all zeros (real value 0), input = [1, 2, 3, 4],
 *        scale = 1.0. Expected output = [0, 0, 0, 0] exactly, since
 *        any input times zero is zero and no scale is ever applied to
 *        a non-zero accumulator. This is the strongest invariant in
 *        KleidiAI's kxn kernel and trips any gross dispatch or layout
 *        error.
 */
TEST(SafetensorsInt4, DotQInt4_zeroWeight_producesZero) {
  const unsigned int K = 4;
  const unsigned int N = 4;

  // FP32 activation [1, 1, 1, 4]
  nntrainer::Tensor act(1, 1, 1, K, nntrainer::Tformat::NCHW,
                        nntrainer::Tdatatype::FP32);
  act.setValue(0, 0, 0, 0, 1.0f);
  act.setValue(0, 0, 0, 1, 2.0f);
  act.setValue(0, 0, 0, 2, 3.0f);
  act.setValue(0, 0, 0, 3, 4.0f);

  nntrainer::Tensor weight = makeKleidiAiInt4Weight(K, N, /*real_value=*/0);

  // Pre-allocate output with [1,1,1,N]. FloatTensor::dotQInteger reads
  // M from this->height() (=1), K from this->width() (=4), and
  // N from output.width() (=4).
  nntrainer::Tensor output(1, 1, 1, N, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP32);
  output.setZero();

  ASSERT_NO_THROW(act.dot(weight, output, false, false));

  for (unsigned int n = 0; n < N; ++n) {
    EXPECT_NEAR(output.getValue(0, 0, 0, n), 0.0f, 1e-5f)
      << "all-zero weight must produce zero output at n=" << n;
  }
}

/**
 * @brief Weight = all +1 (real value 1), input = [1, 2, 3, 4],
 *        scale = 1.0. Expected output per column = sum(input) = 10.
 *        Activation quantization introduces a small rounding error
 *        (~0.04 for this range), so we assert with a generous
 *        tolerance of 0.1.
 */
TEST(SafetensorsInt4, DotQInt4_onesWeight_sumsInputs) {
  const unsigned int K = 4;
  const unsigned int N = 4;

  nntrainer::Tensor act(1, 1, 1, K, nntrainer::Tformat::NCHW,
                        nntrainer::Tdatatype::FP32);
  act.setValue(0, 0, 0, 0, 1.0f);
  act.setValue(0, 0, 0, 1, 2.0f);
  act.setValue(0, 0, 0, 2, 3.0f);
  act.setValue(0, 0, 0, 3, 4.0f);
  const float expected = 1.0f + 2.0f + 3.0f + 4.0f; // = 10

  nntrainer::Tensor weight = makeKleidiAiInt4Weight(K, N, /*real_value=*/1);

  nntrainer::Tensor output(1, 1, 1, N, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP32);
  output.setZero();

  ASSERT_NO_THROW(act.dot(weight, output, false, false));

  for (unsigned int n = 0; n < N; ++n) {
    EXPECT_NEAR(output.getValue(0, 0, 0, n), expected, 0.1f)
      << "ones weight should sum the inputs at n=" << n
      << " (got " << output.getValue(0, 0, 0, n) << ")";
  }
}

/**
 * @brief Weight = all -1 (real value -1), input = [1, 2, 3, 4],
 *        scale = 1.0. Expected output per column = -sum(input) = -10.
 *        The negative branch catches sign-extension errors that the
 *        positive-only ones-weight test would miss (since the
 *        offset-binary encoding for -1 is stored nibble 7, and KleidiAI
 *        subtracts 8 internally to recover -1).
 */
TEST(SafetensorsInt4, DotQInt4_negativeOnesWeight_negSumsInputs) {
  const unsigned int K = 4;
  const unsigned int N = 4;

  nntrainer::Tensor act(1, 1, 1, K, nntrainer::Tformat::NCHW,
                        nntrainer::Tdatatype::FP32);
  act.setValue(0, 0, 0, 0, 1.0f);
  act.setValue(0, 0, 0, 1, 2.0f);
  act.setValue(0, 0, 0, 2, 3.0f);
  act.setValue(0, 0, 0, 3, 4.0f);
  const float expected = -(1.0f + 2.0f + 3.0f + 4.0f); // = -10

  nntrainer::Tensor weight = makeKleidiAiInt4Weight(K, N, /*real_value=*/-1);

  nntrainer::Tensor output(1, 1, 1, N, nntrainer::Tformat::NCHW,
                           nntrainer::Tdatatype::FP32);
  output.setZero();

  ASSERT_NO_THROW(act.dot(weight, output, false, false));

  for (unsigned int n = 0; n < N; ++n) {
    EXPECT_NEAR(output.getValue(0, 0, 0, n), expected, 0.1f)
      << "-1 weight should negate-and-sum the inputs at n=" << n
      << " (got " << output.getValue(0, 0, 0, n) << ")";
  }
}

#endif // ENABLE_FP16

// =============================================================================
// Main function
// =============================================================================

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Failed to initialize google test" << std::endl;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Failed to run all tests" << std::endl;
  }

  return result;
}

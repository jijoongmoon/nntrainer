"""Unit tests for the qsi4cxp channel-wise int4 encoder.

Covers the Python helper `quantize_qsi4cxp_kxn` and the int4 path in
WeightConverter._convert_safetensors, without requiring a HuggingFace
model download. The tests validate:

  1. Output buffer sizing matches nntrainer Int4QTensor::getMemoryBytes()
     for K, N (packed data + per-column fp32 scales).
  2. Round-trip accuracy: a near-int4-representable weight matrix
     quantizes and dequantizes back to itself within the expected
     symmetric-int4 error bound.
  3. Nibble packing convention matches KleidiAI kxn layout exactly:
     even n_idx -> low nibble, odd n_idx -> high nibble, stored
     offset-binary with zero_point=8.
  4. The safetensors writer tags int4 entries with dtype="I4" and
     emits schema_version="2" plus the per-entry quant metadata object.
"""

import json
import math
import os
import struct
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from int4_quant import quantize_qsi4cxp_kxn
from weight_converter import WeightConverter
from nntrainer_layers import NNTrainerLayerDef


# =============================================================================
# quantize_qsi4cxp_kxn unit tests
# =============================================================================

def test_quantize_qsi4cxp_kxn_output_sizes():
    """Packed data size = K * ceil(N/2); scale buffer = 4 * N bytes."""
    for K, N in [(4, 4), (8, 16), (32, 64), (7, 11), (5, 3)]:
        w = torch.randn(K, N)
        packed, scales = quantize_qsi4cxp_kxn(w)
        assert len(packed) == K * ((N + 1) // 2), (
            f"packed size mismatch for K={K}, N={N}: "
            f"got {len(packed)}, expected {K * ((N + 1) // 2)}")
        assert len(scales) == 4 * N, (
            f"scale buffer size mismatch for K={K}, N={N}: "
            f"got {len(scales)}, expected {4 * N}")


def test_quantize_qsi4cxp_kxn_known_pattern():
    """Pack a matrix with handpicked integer values and verify the
    exact byte-level layout against KleidiAI's kxn unpacking rule."""
    # weight[k, n] = n - 4  for k in [0..K-1]
    # All K rows are identical, so the per-column scale should be:
    #   |n-4|.max / 7 = max(|n-4|) / 7 for that column
    #   n=0: |-4|/7 = 4/7, real -4 -> quant (-4 / (4/7)) = -7  -> stored 1
    #   n=1: |-3|/7 = 3/7, real -3 -> quant (-3 / (3/7)) = -7  -> stored 1
    #   n=2: |-2|/7 = 2/7, real -2 -> quant -7 -> stored 1
    #   n=3: |-1|/7 = 1/7, real -1 -> quant -7 -> stored 1
    #   n=4: max=0, scale clipped to 1.0, real 0 -> quant 0 -> stored 8
    #   n=5: 1/7,    real +1 -> quant +7 -> stored 15
    #   n=6: 2/7,    real +2 -> quant +7 -> stored 15
    #   n=7: 3/7,    real +3 -> quant +7 -> stored 15
    K, N = 2, 8
    w = torch.zeros(K, N, dtype=torch.float32)
    for k in range(K):
        for n in range(N):
            w[k, n] = float(n - 4)

    packed, scales_bytes = quantize_qsi4cxp_kxn(w)

    # Verify the scale values (per output column, length N)
    scales = np.frombuffer(scales_bytes, dtype="<f4")
    assert len(scales) == N
    # Column n=4 is all zeros; the quantizer clips the raw absmax to
    # 1.0 to avoid a divide-by-zero, which then gets scaled to 1/7
    # like every other column. The stored nibble is 8 (= real 0) so
    # the dequant output is still 0 regardless of the exact scale.
    expected_scales = np.array(
        [4.0 / 7.0, 3.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0,
         1.0 / 7.0,
         1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0],
        dtype=np.float32,
    )
    np.testing.assert_allclose(scales, expected_scales, rtol=1e-6,
                               err_msg="scales[n] should equal "
                                       "max|w[:,n]|/7 per output column")

    # Verify the packed bytes. Each K-row has ceil(N/2)=4 bytes.
    # Row layout for K=0:
    #   byte 0: low=n0=stored(1), high=n1=stored(1) -> 0x11
    #   byte 1: low=n2=stored(1), high=n3=stored(1) -> 0x11
    #   byte 2: low=n4=stored(8), high=n5=stored(15) -> 0xF8
    #   byte 3: low=n6=stored(15), high=n7=stored(15) -> 0xFF
    packed_arr = np.frombuffer(packed, dtype=np.uint8).reshape(K, 4)
    expected_row = np.array([0x11, 0x11, 0xF8, 0xFF], dtype=np.uint8)
    for k in range(K):
        np.testing.assert_array_equal(
            packed_arr[k], expected_row,
            err_msg=f"packed row {k} does not match KleidiAI kxn layout")


def test_quantize_qsi4cxp_kxn_roundtrip_accuracy():
    """Quantize then manually dequantize using KleidiAI's
    stored_nibble - 8 convention. The reconstructed tensor should
    match the original within half a quantization step per column."""
    torch.manual_seed(0)
    K, N = 8, 16
    w = torch.randn(K, N)

    packed, scales_bytes = quantize_qsi4cxp_kxn(w)
    scales = np.frombuffer(scales_bytes, dtype="<f4")
    assert len(scales) == N

    # Manual unpack + dequant using the KleidiAI convention.
    row_stride = (N + 1) // 2
    packed_arr = np.frombuffer(packed, dtype=np.uint8).reshape(K, row_stride)
    recon = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        for n in range(N):
            b = int(packed_arr[k, n // 2])
            if n % 2 == 0:
                nibble = b & 0x0F
            else:
                nibble = (b >> 4) & 0x0F
            real = nibble - 8  # offset-binary -> signed int4
            recon[k, n] = real * scales[n]

    # Per-column max absolute error should be bounded by scale/2
    # (symmetric rounding) — allow a small epsilon for float rounding.
    max_err_per_col = np.max(np.abs(w.numpy() - recon), axis=0)
    bound = scales / 2.0 + 1e-5
    assert np.all(max_err_per_col <= bound), (
        f"per-column quant error exceeds scale/2: "
        f"err={max_err_per_col}, bound={bound}")


# =============================================================================
# Python <-> C++ golden byte fixture for the GEMM cross-validation test
# =============================================================================
#
# This fixture is the verification bridge that replaces P6b-2 for now.
# The Python quantize_qsi4cxp_kxn() and the C++ KleidiAI forward path
# (FloatTensor::dotQInteger -> nntr_gemm_qai8dxp_qsi4cxp_unpacked) must
# agree on the byte wire format: nibble packing order, offset-binary
# encoding, per-output-column scale layout, and K-row stride.
#
# The C++ side of the fixture lives in
#   test/unittest/unittest_nntrainer_safetensors_int4.cpp ::
#       DotQInt4_gemm_nonUniform_matchesReference
# which hard-codes the same 16 bytes + 4 fp32 scales and runs a small
# GEMM through FloatTensor::dot, comparing against the hand-computed
# reference output documented in GEMM_GOLDEN_* below.
#
# If you change quantize_qsi4cxp_kxn()'s nibble / scale convention you
# MUST update both this test and the C++ test together — the compiler
# will not catch a drift because the two sides communicate through raw
# bytes in a file.

# Signed int4 weight matrix for a K=8, N=4 GEMM. Values are chosen so
# every output column has a different absmax profile:
#   col 0: max|w| = 7  -> scale = 1.0         (exact scale)
#   col 1: max|w| = 7  -> scale = 1.0
#   col 2: max|w| = 7  -> scale = 1.0
#   col 3: all zeros  -> absmax clipped to 1 -> scale = 1/7
GEMM_GOLDEN_K = 8
GEMM_GOLDEN_N = 4
GEMM_GOLDEN_WEIGHT = [
    [ 7., -3.,  0.,  0.],
    [-2.,  5.,  1.,  0.],
    [ 4., -6.,  3.,  0.],
    [-1.,  2., -2.,  0.],
    [ 3., -7.,  5.,  0.],
    [-5.,  4., -4.,  0.],
    [ 0.,  0.,  7.,  0.],
    [ 6., -1., -3.,  0.],
]
# Per-K-row byte pair in KleidiAI kxn order:
#   byte 0: low = w[k,0]+8, high = w[k,1]+8
#   byte 1: low = w[k,2]+8, high = w[k,3]+8
GEMM_GOLDEN_PACKED = bytes([
    0x5F, 0x88,  # k=0  w=( +7, -3,  0,  0)  -> (15|5<<4, 8|8<<4)
    0xD6, 0x89,  # k=1  w=( -2, +5, +1,  0)  -> ( 6|13<<4, 9|8<<4)
    0x2C, 0x8B,  # k=2  w=( +4, -6, +3,  0)  -> (12|2<<4, 11|8<<4)
    0xA7, 0x86,  # k=3  w=( -1, +2, -2,  0)  -> ( 7|10<<4, 6|8<<4)
    0x1B, 0x8D,  # k=4  w=( +3, -7, +5,  0)  -> (11|1<<4, 13|8<<4)
    0xC3, 0x84,  # k=5  w=( -5, +4, -4,  0)  -> ( 3|12<<4, 4|8<<4)
    0x88, 0x8F,  # k=6  w=(  0,  0, +7,  0)  -> ( 8|8<<4, 15|8<<4)
    0x7E, 0x85,  # k=7  w=( +6, -1, -3,  0)  -> (14|7<<4, 5|8<<4)
])
GEMM_GOLDEN_SCALES = [1.0, 1.0, 1.0, 1.0 / 7.0]
# Input [1, 2, 3, 4, 5, 6, 7, 8] dotted with the weight above gives
# the hand-computed per-column reference output:
#   col 0: 1*7 + 2*(-2) + 3*4 + 4*(-1) + 5*3 + 6*(-5) + 7*0 + 8*6     = 44
#   col 1: 1*(-3) + 2*5 + 3*(-6) + 4*2 + 5*(-7) + 6*4 + 7*0 + 8*(-1)  = -22
#   col 2: 1*0 + 2*1 + 3*3 + 4*(-2) + 5*5 + 6*(-4) + 7*7 + 8*(-3)     = 29
#   col 3: 0 * anything                                                = 0
GEMM_GOLDEN_INPUT = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
GEMM_GOLDEN_REFERENCE = [44.0, -22.0, 29.0, 0.0]


def test_gemm_golden_bytes_match_python_encoder():
    """Cross-validation: quantize_qsi4cxp_kxn() applied to the golden
    weight must produce exactly GEMM_GOLDEN_PACKED and
    GEMM_GOLDEN_SCALES. If this test drifts, the C++ GEMM fixture in
    unittest_nntrainer_safetensors_int4.cpp will also drift and the
    Python <-> KleidiAI wire format contract is broken."""
    w = torch.tensor(GEMM_GOLDEN_WEIGHT, dtype=torch.float32)
    packed, scales_bytes = quantize_qsi4cxp_kxn(w)

    assert packed == GEMM_GOLDEN_PACKED, (
        "golden packed bytes drifted. Python encoder output:\n"
        + " ".join(f"0x{b:02X}" for b in packed)
        + "\nexpected:\n"
        + " ".join(f"0x{b:02X}" for b in GEMM_GOLDEN_PACKED)
    )

    scales = np.frombuffer(scales_bytes, dtype="<f4")
    np.testing.assert_allclose(
        scales, np.array(GEMM_GOLDEN_SCALES, dtype=np.float32),
        rtol=1e-6,
        err_msg="golden fp32 scales drifted")


def test_gemm_golden_dequant_matches_reference():
    """Dequantize the golden int4 bytes with the documented KleidiAI
    convention (stored_nibble - 8) and verify the unquantized fp32
    GEMM against GEMM_GOLDEN_INPUT yields GEMM_GOLDEN_REFERENCE.
    This is a pure-Python check; the C++ side runs the same reference
    through the KleidiAI kernel with an additional activation
    quantization error that the C++ test's EXPECT_NEAR absorbs."""
    K = GEMM_GOLDEN_K
    N = GEMM_GOLDEN_N
    row_stride = (N + 1) // 2
    packed_arr = np.frombuffer(
        GEMM_GOLDEN_PACKED, dtype=np.uint8).reshape(K, row_stride)
    scales = np.array(GEMM_GOLDEN_SCALES, dtype=np.float32)

    w_real = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        for n in range(N):
            b = int(packed_arr[k, n // 2])
            nibble = (b & 0x0F) if (n % 2 == 0) else ((b >> 4) & 0x0F)
            w_real[k, n] = (nibble - 8) * scales[n]

    # Check the reconstructed real weight matches the original fp32
    # weight column-by-column. For this hand-picked pattern all
    # non-zero columns have scale=1 so the decode is exact; column 3
    # decodes to 0 (its nibble is 8, and 0 * 1/7 = 0).
    np.testing.assert_allclose(w_real,
                               np.array(GEMM_GOLDEN_WEIGHT,
                                        dtype=np.float32),
                               rtol=1e-6,
                               atol=1e-6)

    # Compute the dot product and verify it matches the reference.
    inp = np.array(GEMM_GOLDEN_INPUT, dtype=np.float32)
    out = inp @ w_real  # [N]
    np.testing.assert_allclose(
        out, np.array(GEMM_GOLDEN_REFERENCE, dtype=np.float32),
        rtol=1e-5, atol=1e-5)


# =============================================================================
# WeightConverter int4 integration tests
# =============================================================================

def _make_tiny_linear_layer_def(name, in_features, out_features):
    """Build a minimal NNTrainerLayerDef with the attributes the weight
    converter touches: name, has_weight, has_bias, shared_from, and a
    hf_key mapping stored on layer.hf_weight_keys / .hf_key.

    The TorchFXConverter's weight_map builder iterates layers in order
    and extracts (hf_key, nntr_layer, transform) tuples. We bypass
    `build_weight_map` by constructing a WeightMap directly and passing
    layer definitions that only need to satisfy `__iter__` and name
    lookups for the weight writer.
    """
    # We construct by subclassing dict-like behavior. NNTrainerLayerDef
    # is a dataclass; different converter versions have slightly
    # different field names. Build the minimum viable object.
    ld = NNTrainerLayerDef(
        name=name,
        layer_type="fully_connected",
        input_names=[],
        output_names=[],
        properties={"unit": str(out_features)},
    )
    ld.in_features = in_features
    ld.out_features = out_features
    return ld


def test_weightconverter_int4_header_has_schema_v2_and_quant():
    """A WeightConverter with int4_linear=True writes an entry whose
    JSON header carries dtype=I4, schema_version=2, and the quant
    metadata object expected by the nntrainer C++ loader."""
    # Bypass layer-driven weight_map construction and hand-build a
    # tiny WeightMap + state_dict so the test does not depend on
    # importing transformers / decomposing a real model.
    from weight_converter import WeightMap

    K, N = 8, 16  # square and rectangular both work; pick non-square
    weight = torch.randn(N, K)  # HF convention [out, in] == [N, K]
    state_dict = {"dense.weight": weight}

    wmap = WeightMap()
    wmap.add("dense.weight", "dense", transform="transpose", is_bias=False)

    # Build a WeightConverter without layer introspection; the
    # constructor accepts `layers` but `_convert_safetensors` only
    # reads from the pre-built weight_map and state_dict.
    wc = WeightConverter.__new__(WeightConverter)
    wc.layers = []
    wc.weight_map = wmap
    wc.name_remap = {}
    wc.weight_order = []
    wc.int4_linear = True
    wc.int4_predicate = None

    with tempfile.NamedTemporaryFile(
            suffix=".safetensors", delete=False) as tmp:
        out_path = tmp.name
    try:
        wc._convert_safetensors(state_dict, out_path,
                                target_dtype=torch.float32,
                                dtype_str="float32")

        with open(out_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8").rstrip()
            header = json.loads(header_json)

        # Schema_version promoted because of the int4 entry
        assert header["__metadata__"].get("schema_version") == "2", (
            f"schema_version should be '2' when any int4 entry is "
            f"present. Got __metadata__={header['__metadata__']}")

        entry = header["dense:weight"]
        assert entry["dtype"] == "I4", (
            f"expected dtype=I4, got {entry['dtype']}")
        # After transpose the shape should be [K, N]
        assert entry["shape"] == [K, N], (
            f"expected shape [K, N] = [{K}, {N}], got {entry['shape']}")
        assert "quant" in entry, "per-entry quant object missing"
        q = entry["quant"]
        assert q["encoding"] == "axis_scale_offset"
        assert q["axis"] == 1
        assert q["bitwidth"] == 4
        assert q["group_size"] == 0  # pure per-channel
        assert q["has_zero_point"] is False

        # Data section size = K*ceil(N/2) + 4*N
        expected_size = K * ((N + 1) // 2) + 4 * N
        d0, d1 = entry["data_offsets"]
        assert d1 - d0 == expected_size, (
            f"data section size mismatch: got {d1 - d0}, "
            f"expected {expected_size}")
    finally:
        os.remove(out_path)


def test_weightconverter_int4_bias_stays_fp32():
    """Biases must NOT be quantized. Even with int4_linear=True a
    bias entry keeps its base fp32 dtype."""
    from weight_converter import WeightMap

    K, N = 4, 8
    weight = torch.randn(N, K)
    bias = torch.randn(N)
    state_dict = {"dense.weight": weight, "dense.bias": bias}

    wmap = WeightMap()
    wmap.add("dense.weight", "dense", transform="transpose", is_bias=False)
    wmap.add("dense.bias",   "dense", transform="none",      is_bias=True)

    wc = WeightConverter.__new__(WeightConverter)
    wc.layers = []
    wc.weight_map = wmap
    wc.name_remap = {}
    wc.weight_order = []
    wc.int4_linear = True
    wc.int4_predicate = None

    with tempfile.NamedTemporaryFile(
            suffix=".safetensors", delete=False) as tmp:
        out_path = tmp.name
    try:
        wc._convert_safetensors(state_dict, out_path,
                                target_dtype=torch.float32,
                                dtype_str="float32")
        with open(out_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode("utf-8").rstrip())

        assert header["dense:weight"]["dtype"] == "I4"
        assert "quant" in header["dense:weight"]
        # Bias must stay fp32
        assert header["dense:bias"]["dtype"] == "F32"
        assert "quant" not in header["dense:bias"]
    finally:
        os.remove(out_path)


if __name__ == "__main__":
    # Minimal test runner so the file can be executed directly without
    # a pytest dependency (matches the existing test_safetensors.py
    # style in this directory).
    tests = [
        test_quantize_qsi4cxp_kxn_output_sizes,
        test_quantize_qsi4cxp_kxn_known_pattern,
        test_quantize_qsi4cxp_kxn_roundtrip_accuracy,
        test_gemm_golden_bytes_match_python_encoder,
        test_gemm_golden_dequant_matches_reference,
        test_weightconverter_int4_header_has_schema_v2_and_quant,
        test_weightconverter_int4_bias_stays_fp32,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    if failed:
        print(f"{failed}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"all {len(tests)} tests passed")

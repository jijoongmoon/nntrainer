"""
Channel-wise int4 (qsi4cxp) quantization — standalone module.

No dependency on nntrainer_layers or any TorchFXConverter internals.
Can be imported from anywhere:

    from int4_quant import quantize_qsi4cxp_kxn

Or via importlib for out-of-tree callers:

    spec = importlib.util.spec_from_file_location(
        "int4_quant", "/path/to/tools/TorchFXConverter/int4_quant.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    quantize_qsi4cxp_kxn = mod.quantize_qsi4cxp_kxn
"""

import numpy as np


def quantize_qsi4cxp_kxn(weight_kxn):
    """Quantize a [K, N] FP32 weight matrix to KleidiAI qsi4cxp format.

    Produces bytes matching nntrainer::Int4QTensor canonical layout +
    KleidiAI qai8dxp_qsi4cxp_unpacked kernel expectation.

    Layout:
      [K * ceil(N/2) bytes] packed int4 nibbles, kxn row-major
      [N * 4 bytes]         fp32 per-output-column scales

    Conventions:
      - Even n_idx -> LOW nibble, odd n_idx -> HIGH nibble
      - stored_nibble = real_int4 + 8 (offset-binary, zero_point=8)
      - scale[n] = max|w[:,n]| / 7 (symmetric, no zero point)

    Args:
        weight_kxn: 2-D torch.Tensor or numpy array in [K, N] layout.
                    HuggingFace [N, K] must be transposed first.

    Returns:
        (packed_bytes, scales_bytes) where
          packed_bytes: bytes, length K * ceil(N/2)
          scales_bytes: bytes, length 4 * N (fp32 little-endian)
    """
    # Accept both torch.Tensor and numpy array
    if hasattr(weight_kxn, 'detach'):
        w = weight_kxn.detach().to("cpu").float().numpy()
    else:
        w = np.asarray(weight_kxn, dtype=np.float32)

    if w.ndim != 2:
        raise ValueError(
            f"quantize_qsi4cxp_kxn expects a 2-D array, got shape {w.shape}")

    K, N = w.shape

    # Per-output-column scale: max|w[:,n]| / 7
    col_absmax = np.abs(w).max(axis=0)
    col_absmax = np.where(col_absmax > 0.0, col_absmax, np.float32(1.0))
    scales = (col_absmax / np.float32(7.0)).astype(np.float32)

    # Quantize + clip to [-8, +7]
    q = np.clip(np.round(w / scales).astype(np.int32), -8, 7)

    # Offset-binary: stored = real + 8 (range 0..15)
    q_u8 = (q + 8).astype(np.uint8)

    # Pack nibbles: even n -> low, odd n -> high
    row_stride = (N + 1) // 2
    packed = np.zeros((K, row_stride), dtype=np.uint8)

    n_even = (N + 1) // 2
    packed[:, :n_even] = q_u8[:, 0:N:2] & 0x0F

    n_odd = N // 2
    if n_odd > 0:
        packed[:, :n_odd] |= (q_u8[:, 1:N:2] & 0x0F) << 4

    packed_bytes = packed.tobytes(order="C")
    scales_bytes = scales.astype("<f4").tobytes(order="C")

    # Prepend 2-byte QScheme header for C++ Int4QTensor::read()
    # compatibility. Int4QTensor::read() always calls
    # read_quantization_info() first, which reads 2 bytes as a
    # uint16_t QScheme enum. PER_CHANNEL_AFFINE = 0x0001.
    QSCHEME_PER_CHANNEL_AFFINE = b'\x01\x00'  # uint16_t LE
    return QSCHEME_PER_CHANNEL_AFFINE + packed_bytes, scales_bytes

"""
Weight converter for NNTrainer TorchFX converter.

Converts HuggingFace model weights to NNTrainer binary or safetensors format.

NNTrainer weight format:
  - Binary file (.bin) with weights in layer creation order
  - Each layer's weights are stored contiguously:
    [weight_data] [bias_data] (if has_bias)
  - Linear layer weights need transposition: [out, in] -> [in, out]
  - Embedding weights are kept as-is: [vocab, dim]
  - RMSNorm / LayerNorm weights are single vectors
  - Tied embeddings share weight reference (stored once)

Safetensors format:
  - [8B] header_size (little-endian uint64)
  - [header_size B] JSON header: {"__metadata__": {...}, "name": {"dtype": "F32", "shape": [...], "data_offsets": [start, end]}, ...}
  - [data section] raw tensor bytes

Phase 4.4 of the TorchFX converter pipeline (DESIGN.md).
"""

import json
import struct
from collections import OrderedDict

from nntrainer_layers import NNTrainerLayerDef


# =============================================================================
# Channel-wise int4 (qsi4cxp) quantization
# =============================================================================
#
# This is the Python counterpart to nntrainer::Int4QTensor's canonical
# on-disk layout (P2 + P4 + P6b, schema_version 2). It produces bytes
# that FloatTensor::dotQInteger can hand directly to KleidiAI's
# qai8dxp_qsi4cxp_unpacked kernel without any translation.
#
# Why this exists: the existing Q4_0 quantizer in quantize.cpp is a C++
# utility that re-quantizes an already-nntrainer-loaded .bin file. It
# cannot be used in a pure-Python HF -> nntrainer pipeline. For
# TorchFXConverter we want a Python quantizer that runs BEFORE load,
# directly on HF torch.Tensor objects.
#
# Layout (matches Int4QTensor::allocate + KleidiAI kxn expectation):
#
#   offset                              | contents
#   ------------------------------------+-------------------------------
#   0 .. K * ceil(N/2) - 1              | packed int4 nibbles, kxn
#                                       | (one K-row of ceil(N/2) bytes,
#                                       |  N outer dim packed 2-per-byte)
#   K*ceil(N/2) .. K*ceil(N/2)+4*N - 1  | fp32 scales, one per output
#                                       | column (length N)
#
# Conventions (KleidiAI-native, NOT Int4QTensor::setValue):
#   - Weight is interpreted as [K, N] with K = input features (reduction
#     axis, = nntrainer weight_dim.height) and N = output features
#     (= width). HF / PyTorch weights are stored as [N, K] so the
#     caller must transpose once before handing the tensor here; this
#     mirrors what WeightConverter already does for FP32 linear layers
#     in `_convert_safetensors`.
#   - Per-output-channel (per-N) symmetric scale: one fp32 scale per N.
#     Within each column, all K reduction elements share that scale.
#     This matches nntrainer::Int4QTensor::scale_size() returning
#     width() and the safetensors schema_version 2 metadata
#     emitting `"encoding":"axis_scale_offset","axis":1`.
#   - Int4 value in range [-8, +7] (symmetric, no zero point), encoded
#     on disk as OFFSET-BINARY with `stored_nibble = real + 8` so the
#     4-bit encoding uses 0..15 unsigned. KleidiAI's qsi4cxp kernel
#     reads this same offset-binary convention internally.
#   - Nibble packing within a K-row: byte `b = k*ceil(N/2) + n/2` holds
#       low nibble (bits 0..3) -> n % 2 == 0 (even n_idx)
#       high nibble (bits 4..7) -> n % 2 == 1 (odd  n_idx)
#     This is KleidiAI's `kxn` ordering. Note: this is the OPPOSITE
#     of Int4QTensor::getValue()'s convention (which places even flat
#     indices in the HIGH nibble as two's-complement signed int4).
#     That discrepancy only matters if downstream C++ code reads the
#     data via getValue() — the KleidiAI forward path uses raw bytes
#     via getData<char>() so it is unaffected.
#
# Follow-up [P6b-2]: align Int4QTensor::setValue/getValue with this
# convention so C++ getValue() prints the semantically correct int4
# values instead of a byte-flipped / sign-flipped view.
def quantize_qsi4cxp_kxn(weight_kxn):
    """Quantize a [K, N] FP32 weight matrix to KleidiAI qsi4cxp format.

    Args:
        weight_kxn: 2-D torch.Tensor in [K, N] layout (K = in_features,
                    N = out_features). Callers convert from HuggingFace
                    [N, K] via `.t().contiguous()` before calling.

    Returns:
        (packed_bytes, scales_fp32) where
          packed_bytes: Python `bytes`, length K * ceil(N/2),
                        packed int4 nibbles in KleidiAI kxn order.
          scales_fp32:  Python `bytes`, length 4*N, per-output-column
                        fp32 scales in little-endian native order.
    """
    import numpy as np

    if weight_kxn.dim() != 2:
        raise ValueError(
            f"quantize_qsi4cxp_kxn expects a 2-D tensor, got shape "
            f"{tuple(weight_kxn.shape)}")

    # Work entirely in numpy so we don't depend on torch quant kernels.
    w = weight_kxn.detach().to("cpu").float().numpy()  # [K, N]
    K, N = w.shape

    # Per-output-column absolute max. Symmetric range -> scale covers
    # -max .. +max in 16 levels (-8 .. +7). Clip the min scale so we
    # never divide by zero for an all-zero column.
    col_absmax = np.abs(w).max(axis=0)                 # [N]
    col_absmax = np.where(col_absmax > 0.0, col_absmax,
                          np.float32(1.0))
    # scale = max_abs / 7 so that the largest value maps to +7. Using
    # 7 (not 8) avoids saturating at the negative extreme -8 when the
    # positive side hits its max; for symmetric signed int4 the
    # reachable positive max is +7.
    scales = (col_absmax / np.float32(7.0)).astype(np.float32)  # [N]

    # Quantize: for each element q[k, n] = round(w[k, n] / scales[n]).
    # Broadcasting scales across K rows.
    q = np.round(w / scales).astype(np.int32)           # [K, N]
    # Clip to the symmetric signed int4 range.
    q = np.clip(q, -8, 7)

    # Convert to offset-binary uint8 nibbles (stored = real + 8).
    q_u8 = (q + 8).astype(np.uint8)                     # [K, N], 0..15

    # Pack two N-values per byte in KleidiAI kxn order:
    #   byte[k, n//2] LOW  nibble  = q_u8[k, n]         (even n_idx)
    #   byte[k, n//2] HIGH nibble  = q_u8[k, n+1]       (odd  n_idx)
    # For odd N the last byte has a valid LOW nibble (last even index)
    # but its HIGH nibble is padding; leave it as 0, which decodes
    # to real -8 but is never read since n_idx never exceeds N - 1.
    row_stride = (N + 1) // 2                           # bytes per K row
    packed = np.zeros((K, row_stride), dtype=np.uint8)

    # Even-index columns occupy the LOW nibble of every byte that
    # has one. For N elements the even indices are 0,2,...,2*m where
    # m = (N+1)//2 - 1, giving (N+1)//2 slots — exactly row_stride.
    n_even = (N + 1) // 2
    packed[:, :n_even] = q_u8[:, 0:N:2] & 0x0F

    # Odd-index columns occupy the HIGH nibble of the first N//2 bytes.
    # For odd N there is no odd index paired with the last even one,
    # so we only touch the first (N//2) bytes, not the full row.
    n_odd = N // 2
    if n_odd > 0:
        packed[:, :n_odd] |= (q_u8[:, 1:N:2] & 0x0F) << 4

    packed_bytes = packed.tobytes(order="C")
    scales_bytes = scales.astype("<f4").tobytes(order="C")

    return packed_bytes, scales_bytes


# =============================================================================
# Weight Mapping
# =============================================================================

class WeightMap:
    """Maps HuggingFace state_dict keys to NNTrainer weight order.

    Each entry describes:
      - hf_key: HuggingFace state_dict key (e.g. "model.layers.0.self_attn.q_proj.weight")
      - nntr_layer: NNTrainer layer name
      - transform: "transpose" | "none" | "skip"
      - dtype: target data type
    """

    def __init__(self):
        self.entries = []

    def add(self, hf_key, nntr_layer, transform="none", is_bias=False):
        self.entries.append({
            "hf_key": hf_key,
            "nntr_layer": nntr_layer,
            "transform": transform,
            "is_bias": is_bias,
        })

    def __iter__(self):
        return iter(self.entries)

    def __len__(self):
        return len(self.entries)


def build_weight_map(layers):
    """Build weight mapping from converter layer list.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.

    Returns:
        WeightMap with entries in layer creation order.
    """
    wmap = WeightMap()
    seen_shared = set()

    for layer in layers:
        # Skip layers without weights
        if not layer.has_weight and not layer.has_bias:
            continue

        # Skip tied weights (shared_from = already stored)
        if layer.shared_from:
            if layer.shared_from in seen_shared:
                continue
            seen_shared.add(layer.shared_from)

        # Weight
        if layer.has_weight and layer.weight_hf_key:
            transform = "transpose" if layer.transpose_weight else "none"
            wmap.add(layer.weight_hf_key, layer.name, transform)

        # Bias
        if layer.has_bias and layer.bias_hf_key:
            wmap.add(layer.bias_hf_key, layer.name, "none", is_bias=True)

    return wmap


# =============================================================================
# Weight Converter
# =============================================================================

class WeightConverter:
    """Converts HuggingFace weights to NNTrainer binary or safetensors format.

    Usage:
        converter = WeightConverter(layers)
        converter.convert(hf_state_dict, "model.bin")
        converter.convert(hf_state_dict, "model.safetensors", output_format="safetensors")
        # or
        converter.convert_from_pretrained("Qwen/Qwen3-0.6B", "model.bin")

    For CausalLM-compatible output:
        name_remap, weight_order = build_causallm_mapping(28, True, "qwen3")
        converter = WeightConverter(layers, name_remap=name_remap,
                                    weight_order=weight_order)
    """

    def __init__(self, layers, name_remap=None, weight_order=None,
                 int4_linear=False, int4_predicate=None):
        """
        Args:
            layers: List of NNTrainerLayerDef from converter pipeline.
            name_remap: Optional dict mapping sanitized HF layer name to
                        target layer name (e.g. CausalLM names).
            weight_order: Optional list of target layer names specifying
                          the output order for binary format.
            int4_linear: If True, quantize all Linear layer weights to
                         qsi4cxp channel-wise int4. Embedding and norm
                         weights are left at the converter's base dtype
                         (fp32 / fp16). Biases always stay at base dtype.
            int4_predicate: Optional callable(entry) -> bool. Overrides
                         int4_linear for fine-grained selection: return
                         True for the entries that should be quantized
                         to int4. The entry is a dict from build_weight_map
                         with keys {hf_key, nntr_layer, transform, is_bias}.
        """
        self.layers = layers
        self.weight_map = build_weight_map(layers)
        self.name_remap = name_remap or {}
        self.weight_order = weight_order or []
        self.int4_linear = int4_linear
        self.int4_predicate = int4_predicate

    def _dtype_to_safetensors(self, dtype_str):
        """Map dtype string to safetensors dtype name."""
        mapping = {
            "float32": "F32",
            "float16": "F16",
            "bfloat16": "BF16",
            "int8": "I8",
            "int16": "I16",
            "int32": "I32",
            "uint8": "U8",
            "qint4": "I4",
        }
        return mapping.get(dtype_str, "F32")

    def _should_quantize_int4(self, entry):
        """Return True if the given weight entry should be emitted as
        qsi4cxp channel-wise int4.

        Policy:
          - Biases stay at base dtype (I4 would lose accuracy with no
            upside, and Int4QTensor layout is weight-only).
          - If a user-supplied predicate is set, it fully controls
            selection. This lets callers exclude specific layers (e.g.
            gate_proj/up_proj) or opt individual layers in.
          - Otherwise `int4_linear=True` turns on int4 for any entry
            that had transform=="transpose" (the TorchFXConverter
            pipeline marks Linear weights as transpose-needed; other
            weight kinds like embedding/norm use transform=="none").
        """
        if entry.get("is_bias"):
            return False
        if self.int4_predicate is not None:
            return bool(self.int4_predicate(entry))
        if self.int4_linear and entry.get("transform") == "transpose":
            return True
        return False

    def convert(self, state_dict, output_path, dtype="float32",
                output_format="auto"):
        """Convert HuggingFace state_dict to NNTrainer format.

        Args:
            state_dict: Dict of parameter_name -> torch.Tensor.
            output_path: Output file path (.bin or .safetensors).
            dtype: Target dtype ("float32" or "float16").
            output_format: "bin", "safetensors", or "auto" (detect from extension).
        """
        import torch
        import numpy as np

        if output_format == "auto":
            if output_path.endswith(".safetensors"):
                output_format = "safetensors"
            else:
                output_format = "bin"

        target_dtype = torch.float32 if dtype == "float32" else torch.float16

        if output_format == "safetensors":
            return self._convert_safetensors(state_dict, output_path,
                                             target_dtype, dtype)
        else:
            return self._convert_bin(state_dict, output_path, target_dtype)

    def _remap_layer_name(self, nntr_layer):
        """Remap a layer name using name_remap if available."""
        return self.name_remap.get(nntr_layer, nntr_layer)

    def _ordered_entries(self):
        """Return weight map entries in the correct output order.

        If weight_order is set, reorders entries to match the specified order.
        Otherwise returns entries in their original order.
        """
        if not self.weight_order:
            return list(self.weight_map)

        # Build lookup: remapped_name:suffix -> entry
        by_name = {}
        for entry in self.weight_map:
            remapped = self._remap_layer_name(entry["nntr_layer"])
            suffix = ":bias" if entry.get("is_bias") else ":weight"
            by_name[remapped + suffix] = entry

        ordered = []
        for layer_name in self.weight_order:
            # Each layer may have weight and/or bias
            wkey = layer_name + ":weight"
            bkey = layer_name + ":bias"
            if wkey in by_name:
                ordered.append(by_name.pop(wkey))
            if bkey in by_name:
                ordered.append(by_name.pop(bkey))

        # Append any remaining entries not in weight_order
        for entry in self.weight_map:
            remapped = self._remap_layer_name(entry["nntr_layer"])
            suffix = ":bias" if entry.get("is_bias") else ":weight"
            key = remapped + suffix
            if key in by_name:
                ordered.append(by_name.pop(key))

        return ordered

    def _convert_bin(self, state_dict, output_path, target_dtype):
        """Convert to raw binary format (legacy)."""
        import torch

        entries = self._ordered_entries()
        with open(output_path, "wb") as f:
            for entry in entries:
                hf_key = entry["hf_key"]
                if hf_key not in state_dict:
                    raise KeyError(
                        f"Weight key '{hf_key}' not found in state_dict. "
                        f"Available keys: {list(state_dict.keys())[:10]}...")

                tensor = state_dict[hf_key].to(target_dtype)

                # Apply transformation
                if entry["transform"] == "transpose" and tensor.dim() == 2:
                    tensor = tensor.t().contiguous()

                # Write raw bytes
                data = tensor.cpu().numpy().tobytes()
                f.write(data)

        return output_path

    def _convert_safetensors(self, state_dict, output_path, target_dtype,
                             dtype_str):
        """Convert to safetensors format with JSON header.

        Safetensors format:
          [8B]  header_size (little-endian uint64)
          [header_size B] JSON header
          [data section]  raw tensor bytes

        When name_remap is set, tensor names use the remapped CausalLM-style
        layer names (e.g. "layer0_wq:weight") instead of sanitized HF names.

        When `int4_linear=True` (or `int4_predicate` selects an entry),
        the selected Linear weights are quantized to channel-wise int4
        via `quantize_qsi4cxp_kxn` and written with schema_version="2"
        + per-entry "quant" metadata object, matching the nntrainer
        Int4QTensor canonical layout. Embeddings, norms, and biases
        stay at the converter's base dtype. The presence of any int4
        entry promotes the whole file to schema_version 2; otherwise
        schema_version stays at 1 (= default, not emitted).
        """
        import torch

        sf_dtype = self._dtype_to_safetensors(dtype_str)

        # Use ordered entries (respects weight_order if set)
        entries = self._ordered_entries()

        # First pass: collect all tensor data and metadata
        tensor_data = []
        header_entries = OrderedDict()
        data_offset = 0
        any_quant = False  # triggers schema_version=2 on the metadata block

        for entry in entries:
            hf_key = entry["hf_key"]
            if hf_key not in state_dict:
                raise KeyError(
                    f"Weight key '{hf_key}' not found in state_dict. "
                    f"Available keys: {list(state_dict.keys())[:10]}...")

            tensor = state_dict[hf_key].to(target_dtype)

            if entry["transform"] == "transpose" and tensor.dim() == 2:
                tensor = tensor.t().contiguous()

            # Use remapped name if available, otherwise sanitized HF name
            nntr_name = self._remap_layer_name(entry["nntr_layer"])
            if entry.get("is_bias"):
                nntr_name += ":bias"
            else:
                nntr_name += ":weight"

            if self._should_quantize_int4(entry):
                # Quantize to qsi4cxp channel-wise int4. The transpose
                # above has already put the tensor in [K, N] order
                # (K = reduction / input features, N = output features),
                # which is exactly what quantize_qsi4cxp_kxn expects.
                if tensor.dim() != 2:
                    raise ValueError(
                        f"int4 quantization requires a 2-D weight, "
                        f"but '{hf_key}' has shape "
                        f"{tuple(tensor.shape)}")

                packed_bytes, scales_bytes = quantize_qsi4cxp_kxn(tensor)
                data = packed_bytes + scales_bytes
                data_size = len(data)
                K, N = int(tensor.shape[0]), int(tensor.shape[1])

                header_entries[nntr_name] = {
                    "dtype": "I4",
                    "shape": [K, N],
                    "data_offsets": [data_offset, data_offset + data_size],
                    # schema_version 2 quant object. Must agree with the
                    # C++ save path in neuralnet.cpp::deriveQuantInfo:
                    #   encoding   = "axis_scale_offset"
                    #   axis       = 1  (per-output-column for [K,N] layout)
                    #   bitwidth   = 4
                    #   group_size = 0  (pure per-channel, == height)
                    #   has_zero_point = false (signed int4, offset-binary
                    #                            is an encoding detail,
                    #                            not an asymmetric zp).
                    "quant": {
                        "encoding": "axis_scale_offset",
                        "axis": 1,
                        "bitwidth": 4,
                        "group_size": 0,
                        "has_zero_point": False,
                    },
                }
                any_quant = True
            else:
                data = tensor.cpu().numpy().tobytes()
                data_size = len(data)
                shape = list(tensor.shape)
                header_entries[nntr_name] = {
                    "dtype": sf_dtype,
                    "shape": shape,
                    "data_offsets": [data_offset, data_offset + data_size],
                }

            tensor_data.append(data)
            data_offset += data_size

        # Build JSON header
        metadata = {"format": "nntrainer"}
        if any_quant:
            # Promote to schema_version 2 so the C++ loader's strict
            # validation accepts the per-entry quant objects.
            metadata["schema_version"] = "2"
        header_dict = {"__metadata__": metadata}
        header_dict.update(header_entries)
        header_json = json.dumps(header_dict, separators=(",", ":"),
                                 ensure_ascii=True)

        # Pad to 8-byte alignment
        header_bytes = header_json.encode("utf-8")
        pad_len = (8 - len(header_bytes) % 8) % 8
        header_bytes += b" " * pad_len

        # Write file
        with open(output_path, "wb") as f:
            # Header size (8 bytes, little-endian)
            f.write(struct.pack("<Q", len(header_bytes)))
            # JSON header
            f.write(header_bytes)
            # Data section
            for data in tensor_data:
                f.write(data)

        return output_path

    def convert_from_pretrained(self, model_name_or_path, output_path,
                                dtype="float32", output_format="auto"):
        """Load HuggingFace model weights and convert.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            output_path: Output file path (.bin or .safetensors).
            dtype: Target dtype.
            output_format: "bin", "safetensors", or "auto".
        """
        from transformers import AutoModelForCausalLM
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32)
        state_dict = model.state_dict()
        return self.convert(state_dict, output_path, dtype, output_format)

    def summary(self):
        """Print weight mapping summary."""
        entries = self._ordered_entries()
        has_remap = bool(self.name_remap)
        print(f"Weight map: {len(entries)} entries"
              f"{' (CausalLM remapped)' if has_remap else ''}")
        if has_remap:
            print(f"{'HF Key':<55} {'Original':<30} {'Remapped':<25} {'Xform'}")
            print("-" * 115)
            for entry in entries:
                remapped = self._remap_layer_name(entry['nntr_layer'])
                print(f"{entry['hf_key']:<55} "
                      f"{entry['nntr_layer']:<30} "
                      f"{remapped:<25} "
                      f"{entry['transform']}")
        else:
            print(f"{'HF Key':<60} {'NNTrainer Layer':<30} {'Transform'}")
            print("-" * 100)
            for entry in entries:
                print(f"{entry['hf_key']:<60} "
                      f"{entry['nntr_layer']:<30} "
                      f"{entry['transform']}")

    def generate_script(self):
        """Generate a standalone Python weight conversion script.

        Returns:
            str: Python script content that can be saved and run independently.
        """
        lines = []
        lines.append('"""Auto-generated weight conversion script."""')
        lines.append("import torch")
        lines.append("import sys")
        lines.append("")
        lines.append("def convert(model_path, output_path, dtype='float32'):")
        lines.append("    from transformers import AutoModel")
        lines.append("    target = torch.float32 if dtype == 'float32' "
                     "else torch.float16")
        lines.append("    model = AutoModel.from_pretrained("
                     "model_path, torch_dtype=torch.float32)")
        lines.append("    sd = model.state_dict()")
        lines.append("")
        lines.append("    # Weight mapping (HF key, transform)")
        lines.append("    WEIGHT_MAP = [")
        for entry in self.weight_map:
            lines.append(f'        ("{entry["hf_key"]}", '
                         f'"{entry["transform"]}"),')
        lines.append("    ]")
        lines.append("")
        lines.append("    with open(output_path, 'wb') as f:")
        lines.append("        for hf_key, transform in WEIGHT_MAP:")
        lines.append("            t = sd[hf_key].to(target)")
        lines.append("            if transform == 'transpose' and "
                     "t.dim() == 2:")
        lines.append("                t = t.t().contiguous()")
        lines.append("            f.write(t.cpu().numpy().tobytes())")
        lines.append("")
        lines.append("    print(f'Saved {len(WEIGHT_MAP)} weight tensors "
                     "to {output_path}')")
        lines.append("")
        lines.append("if __name__ == '__main__':")
        lines.append("    if len(sys.argv) < 3:")
        lines.append("        print('Usage: python convert_weights.py "
                     "<model_path> <output.bin> [float32|float16]')")
        lines.append("        sys.exit(1)")
        lines.append("    dtype = sys.argv[3] if len(sys.argv) > 3 "
                     "else 'float32'")
        lines.append("    convert(sys.argv[1], sys.argv[2], dtype)")
        lines.append("")

        return "\n".join(lines)


# =============================================================================
# CausalLM Name Mapping
# =============================================================================

def build_causallm_mapping(num_layers, tie_word_embeddings=True,
                           model_type="qwen3"):
    """Build name remapping and weight ordering for CausalLM C++ app.

    The CausalLM C++ app (Applications/CausalLM) uses hardcoded layer names
    that differ from TorchFXConverter's sanitized HuggingFace module names.
    This function generates the mapping between them.

    Args:
        num_layers: Number of transformer decoder layers.
        tie_word_embeddings: Whether lm_head shares weights with embedding.
        model_type: Model architecture ("qwen3", "llama", "qwen2").

    Returns:
        tuple: (name_remap, weight_order)
          - name_remap: dict mapping sanitized HF name -> CausalLM layer name
          - weight_order: list of CausalLM layer names in C++ creation order
    """
    name_remap = {}
    weight_order = []

    # Embedding
    name_remap["model_embed_tokens"] = "embedding0"
    weight_order.append("embedding0")

    for i in range(num_layers):
        hf_prefix = f"model_layers_{i}_"
        cl_prefix = f"layer{i}_"

        # Attention norm
        name_remap[f"{hf_prefix}input_layernorm"] = f"{cl_prefix}attention_norm"
        weight_order.append(f"{cl_prefix}attention_norm")

        if model_type == "qwen3":
            # Qwen3 createAttention order: V, K, K_norm, Q, Q_norm, O
            name_remap[f"{hf_prefix}self_attn_v_proj"] = f"{cl_prefix}wv"
            name_remap[f"{hf_prefix}self_attn_k_proj"] = f"{cl_prefix}wk"
            name_remap[f"{hf_prefix}self_attn_k_norm"] = f"{cl_prefix}k_norm"
            name_remap[f"{hf_prefix}self_attn_q_proj"] = f"{cl_prefix}wq"
            name_remap[f"{hf_prefix}self_attn_q_norm"] = f"{cl_prefix}q_norm"
            name_remap[f"{hf_prefix}self_attn_o_proj"] = f"{cl_prefix}attention_out"
            weight_order.extend([
                f"{cl_prefix}wv", f"{cl_prefix}wk", f"{cl_prefix}k_norm",
                f"{cl_prefix}wq", f"{cl_prefix}q_norm",
                f"{cl_prefix}attention_out",
            ])
        else:
            # Default (llama, qwen2): V, K, Q, O
            name_remap[f"{hf_prefix}self_attn_v_proj"] = f"{cl_prefix}wv"
            name_remap[f"{hf_prefix}self_attn_k_proj"] = f"{cl_prefix}wk"
            name_remap[f"{hf_prefix}self_attn_q_proj"] = f"{cl_prefix}wq"
            name_remap[f"{hf_prefix}self_attn_o_proj"] = f"{cl_prefix}attention_out"
            weight_order.extend([
                f"{cl_prefix}wv", f"{cl_prefix}wk", f"{cl_prefix}wq",
                f"{cl_prefix}attention_out",
            ])

        # FFN norm
        name_remap[f"{hf_prefix}post_attention_layernorm"] = f"{cl_prefix}ffn_norm"
        weight_order.append(f"{cl_prefix}ffn_norm")

        # FFN: up, gate, down
        name_remap[f"{hf_prefix}mlp_up_proj"] = f"{cl_prefix}ffn_up"
        name_remap[f"{hf_prefix}mlp_gate_proj"] = f"{cl_prefix}ffn_gate"
        name_remap[f"{hf_prefix}mlp_down_proj"] = f"{cl_prefix}ffn_down"
        weight_order.extend([
            f"{cl_prefix}ffn_up", f"{cl_prefix}ffn_gate",
            f"{cl_prefix}ffn_down",
        ])

    # Output norm
    name_remap["model_norm"] = "output_norm"
    weight_order.append("output_norm")

    # LM head (only if not tied)
    if not tie_word_embeddings:
        name_remap["lm_head"] = "output_of_causallm"
        weight_order.append("output_of_causallm")

    return name_remap, weight_order


# =============================================================================
# Convenience functions
# =============================================================================

def convert_weights(layers, state_dict, output_path, dtype="float32",
                    output_format="auto"):
    """Convert HuggingFace weights to NNTrainer format.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        state_dict: HuggingFace model state_dict.
        output_path: Output file path (.bin or .safetensors).
        dtype: Target dtype.
        output_format: "bin", "safetensors", or "auto" (detect from extension).

    Returns:
        str: Output file path.
    """
    converter = WeightConverter(layers)
    return converter.convert(state_dict, output_path, dtype, output_format)


def convert_weights_causallm(layers, state_dict, output_path, num_layers,
                             tie_word_embeddings=True, model_type="qwen3",
                             dtype="float32", output_format="auto"):
    """Convert HuggingFace weights to NNTrainer CausalLM-compatible format.

    Remaps weight names and reorders to match the CausalLM C++ app's
    layer creation order, ensuring correct loading for both .bin and
    .safetensors formats.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        state_dict: HuggingFace model state_dict.
        output_path: Output file path (.bin or .safetensors).
        num_layers: Number of transformer decoder layers.
        tie_word_embeddings: Whether lm_head shares weights with embedding.
        model_type: Model architecture ("qwen3", "llama", "qwen2").
        dtype: Target dtype.
        output_format: "bin", "safetensors", or "auto" (detect from extension).

    Returns:
        str: Output file path.
    """
    name_remap, weight_order = build_causallm_mapping(
        num_layers, tie_word_embeddings, model_type)
    converter = WeightConverter(layers, name_remap=name_remap,
                                weight_order=weight_order)
    return converter.convert(state_dict, output_path, dtype, output_format)


def convert_causallm_from_pretrained(model_path, output_path, model_type="qwen3",
                                     dtype="float32", output_format="auto"):
    """Convert a local HuggingFace model directly to CausalLM-compatible format.

    This is a standalone function that does NOT require the TorchFXConverter
    tracing pipeline. It reads the HF model config, builds the correct weight
    mapping for CausalLM, and converts weights.

    Args:
        model_path: Local path to HuggingFace model directory.
        output_path: Output file path (.bin or .safetensors).
        model_type: Model architecture ("qwen3", "llama", "qwen2").
        dtype: Target dtype ("float32" or "float16").
        output_format: "bin", "safetensors", or "auto".

    Returns:
        str: Output file path.
    """
    import torch
    import numpy as np
    from transformers import AutoConfig, AutoModelForCausalLM

    if output_format == "auto":
        output_format = "safetensors" if output_path.endswith(".safetensors") else "bin"

    # Load model config and weights
    config = AutoConfig.from_pretrained(model_path)
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    state_dict = model.state_dict()

    num_layers = config.num_hidden_layers
    tie_word = getattr(config, "tie_word_embeddings", True)

    # Auto-detect model_type from config if not specified
    if model_type == "auto":
        arch = getattr(config, "model_type", "").lower()
        if "qwen3" in arch or (hasattr(config, "architectures") and
                any("Qwen3" in a for a in config.architectures)):
            model_type = "qwen3"
        else:
            model_type = "llama"  # default fallback

    name_remap, weight_order = build_causallm_mapping(
        num_layers, tie_word, model_type)

    print(f"Model: {model_path}")
    print(f"  type={model_type}, layers={num_layers}, "
          f"tie_word_embeddings={tie_word}")
    print(f"  Output: {output_path} ({output_format}, {dtype})")

    # Build weight entries in CausalLM order
    # Map: causallm_name -> (hf_key, transform)
    _ATTN_PROJS = {"wv": "v_proj", "wk": "k_proj", "wq": "q_proj",
                   "attention_out": "o_proj"}
    _FFN_PROJS = {"ffn_up": "up_proj", "ffn_gate": "gate_proj",
                  "ffn_down": "down_proj"}

    target_dtype = torch.float32 if dtype == "float32" else torch.float16
    sf_dtype = "F32" if dtype == "float32" else "F16"

    # Collect tensors in CausalLM order
    ordered_tensors = []  # list of (causallm_name, tensor_bytes, shape)

    for cl_name in weight_order:
        if cl_name == "embedding0":
            hf_key = "model.embed_tokens.weight"
            t = state_dict[hf_key].to(target_dtype)
        elif cl_name == "output_norm":
            hf_key = "model.norm.weight"
            t = state_dict[hf_key].to(target_dtype)
        elif cl_name == "output_of_causallm":
            hf_key = "lm_head.weight"
            t = state_dict[hf_key].to(target_dtype).t().contiguous()
        else:
            # Parse layer index: "layer{i}_{suffix}"
            parts = cl_name.split("_", 1)
            layer_idx = int(parts[0].replace("layer", ""))
            suffix = parts[1]
            lp = f"model.layers.{layer_idx}."

            if suffix == "attention_norm":
                hf_key = f"{lp}input_layernorm.weight"
                t = state_dict[hf_key].to(target_dtype)
            elif suffix == "ffn_norm":
                hf_key = f"{lp}post_attention_layernorm.weight"
                t = state_dict[hf_key].to(target_dtype)
            elif suffix == "q_norm":
                hf_key = f"{lp}self_attn.q_norm.weight"
                t = state_dict[hf_key].to(target_dtype)
            elif suffix == "k_norm":
                hf_key = f"{lp}self_attn.k_norm.weight"
                t = state_dict[hf_key].to(target_dtype)
            elif suffix in _ATTN_PROJS:
                hf_key = f"{lp}self_attn.{_ATTN_PROJS[suffix]}.weight"
                t = state_dict[hf_key].to(target_dtype).t().contiguous()
            elif suffix in _FFN_PROJS:
                hf_key = f"{lp}mlp.{_FFN_PROJS[suffix]}.weight"
                t = state_dict[hf_key].to(target_dtype).t().contiguous()
            else:
                raise ValueError(f"Unknown CausalLM layer suffix: {suffix}")

        data = t.cpu().numpy().tobytes()
        ordered_tensors.append((cl_name + ":weight", data, list(t.shape)))

    # Write output
    if output_format == "bin":
        with open(output_path, "wb") as f:
            for _, data, _ in ordered_tensors:
                f.write(data)
    else:
        # Safetensors
        header_entries = OrderedDict()
        data_offset = 0
        for name, data, shape in ordered_tensors:
            header_entries[name] = {
                "dtype": sf_dtype, "shape": shape,
                "data_offsets": [data_offset, data_offset + len(data)],
            }
            data_offset += len(data)

        header_dict = {"__metadata__": {"format": "nntrainer"}}
        header_dict.update(header_entries)
        header_json = json.dumps(header_dict, separators=(",", ":"),
                                 ensure_ascii=True)
        header_bytes = header_json.encode("utf-8")
        pad_len = (8 - len(header_bytes) % 8) % 8
        header_bytes += b" " * pad_len

        with open(output_path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for _, data, _ in ordered_tensors:
                f.write(data)

    print(f"Saved {len(ordered_tensors)} weights to {output_path}")
    return output_path


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model weights to NNTrainer format")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Local path to HuggingFace model")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file path (.bin or .safetensors)")
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "qwen3", "llama", "qwen2"],
                        help="Model architecture (default: auto-detect)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="Target data type")
    args = parser.parse_args()

    convert_causallm_from_pretrained(
        args.model_path, args.output,
        model_type=args.model_type, dtype=args.dtype)

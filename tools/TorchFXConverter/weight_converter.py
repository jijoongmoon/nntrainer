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

    def __init__(self, layers, name_remap=None, weight_order=None):
        """
        Args:
            layers: List of NNTrainerLayerDef from converter pipeline.
            name_remap: Optional dict mapping sanitized HF layer name to
                        target layer name (e.g. CausalLM names).
            weight_order: Optional list of target layer names specifying
                          the output order for binary format.
        """
        self.layers = layers
        self.weight_map = build_weight_map(layers)
        self.name_remap = name_remap or {}
        self.weight_order = weight_order or []

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
        }
        return mapping.get(dtype_str, "F32")

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

        # Build lookup: remapped_name -> entry
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
        """
        import torch

        sf_dtype = self._dtype_to_safetensors(dtype_str)

        # Use ordered entries (respects weight_order if set)
        entries = self._ordered_entries()

        # First pass: collect all tensor data and metadata
        tensor_data = []
        header_entries = OrderedDict()
        data_offset = 0

        for entry in entries:
            hf_key = entry["hf_key"]
            if hf_key not in state_dict:
                raise KeyError(
                    f"Weight key '{hf_key}' not found in state_dict. "
                    f"Available keys: {list(state_dict.keys())[:10]}...")

            tensor = state_dict[hf_key].to(target_dtype)

            if entry["transform"] == "transpose" and tensor.dim() == 2:
                tensor = tensor.t().contiguous()

            data = tensor.cpu().numpy().tobytes()
            data_size = len(data)

            # Use remapped name if available, otherwise sanitized HF name
            nntr_name = self._remap_layer_name(entry["nntr_layer"])
            if entry.get("is_bias"):
                nntr_name += ":bias"
            else:
                nntr_name += ":weight"

            shape = list(tensor.shape)

            header_entries[nntr_name] = {
                "dtype": sf_dtype,
                "shape": shape,
                "data_offsets": [data_offset, data_offset + data_size],
            }

            tensor_data.append(data)
            data_offset += data_size

        # Build JSON header
        header_dict = {"__metadata__": {"format": "nntrainer"}}
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

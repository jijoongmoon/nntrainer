"""
Weight converter for NNTrainer TorchFX converter.

Converts HuggingFace model weights to NNTrainer binary format.

NNTrainer weight format:
  - Binary file (.bin) with weights in layer creation order
  - Each layer's weights are stored contiguously:
    [weight_data] [bias_data] (if has_bias)
  - Linear layer weights need transposition: [out, in] -> [in, out]
  - Conv2D weights are reshaped to 2D matrix form:
    [filters, in_ch, k_h, k_w] -> [filters, in_ch * k_h * k_w]
  - Embedding weights are kept as-is: [vocab, dim]
  - RMSNorm / LayerNorm weights are single vectors
  - Tied embeddings share weight reference (stored once)

Phase 4.4 of the TorchFX converter pipeline (DESIGN.md).
"""

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
      - transform: "transpose" | "reshape_2d" | "none" | "skip"
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
            if layer.reshape_weight_2d:
                transform = "reshape_2d"
            elif layer.squeeze_weight_3d:
                transform = "squeeze_3d"
            elif layer.transpose_weight:
                transform = "transpose"
            else:
                transform = "none"
            # Handle fused weight splitting (e.g., Granite fused gate+up)
            if layer.weight_split:
                transform = transform + "+" + layer.weight_split
            wmap.add(layer.weight_hf_key, layer.name, transform)

        # Bias
        if layer.has_bias and layer.bias_hf_key:
            wmap.add(layer.bias_hf_key, layer.name, "none", is_bias=True)

    return wmap


# =============================================================================
# Weight Converter
# =============================================================================

class WeightConverter:
    """Converts HuggingFace weights to NNTrainer binary format.

    Usage:
        converter = WeightConverter(layers)
        converter.convert(hf_state_dict, "model.bin")
        # or
        converter.convert_from_pretrained("Qwen/Qwen3-0.6B", "model.bin")
    """

    def __init__(self, layers):
        self.layers = layers
        self.weight_map = build_weight_map(layers)

    def convert(self, state_dict, output_path, dtype="float32"):
        """Convert HuggingFace state_dict to NNTrainer binary format.

        Args:
            state_dict: Dict of parameter_name -> torch.Tensor.
            output_path: Output .bin file path.
            dtype: Target dtype ("float32" or "float16").
        """
        import torch
        import numpy as np

        target_dtype = torch.float32 if dtype == "float32" else torch.float16

        with open(output_path, "wb") as f:
            for entry in self.weight_map:
                hf_key = entry["hf_key"]
                if hf_key not in state_dict:
                    raise KeyError(
                        f"Weight key '{hf_key}' not found in state_dict. "
                        f"Available keys: {list(state_dict.keys())[:10]}...")

                tensor = state_dict[hf_key].to(target_dtype)

                # Apply transformation
                transform = entry["transform"]

                # Handle composite transforms (e.g., "transpose+first_half")
                split_mode = None
                if "+first_half" in transform:
                    split_mode = "first_half"
                    transform = transform.replace("+first_half", "")
                elif "+second_half" in transform:
                    split_mode = "second_half"
                    transform = transform.replace("+second_half", "")

                if transform == "transpose" and tensor.dim() == 2:
                    tensor = tensor.t().contiguous()
                elif transform == "reshape_2d" and tensor.dim() == 4:
                    # Conv2D: (filters, in_ch, k_h, k_w) -> (filters, in_ch*k_h*k_w)
                    filters = tensor.shape[0]
                    tensor = tensor.reshape(filters, -1).contiguous()
                elif transform == "squeeze_3d" and tensor.dim() == 3:
                    # Depthwise Conv1D: (channels, 1, ksize) -> (channels, ksize)
                    tensor = tensor.squeeze(1).contiguous()

                # Apply weight splitting (fused gate+up decomposition)
                if split_mode:
                    half = tensor.shape[0] // 2
                    if split_mode == "first_half":
                        tensor = tensor[:half].contiguous()
                    else:
                        tensor = tensor[half:].contiguous()

                # Write raw bytes
                data = tensor.cpu().numpy().tobytes()
                f.write(data)

        return output_path

    def convert_from_pretrained(self, model_name_or_path, output_path,
                                dtype="float32"):
        """Load HuggingFace model weights and convert.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            output_path: Output .bin file path.
            dtype: Target dtype.
        """
        from transformers import AutoModelForCausalLM
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32)
        state_dict = model.state_dict()
        return self.convert(state_dict, output_path, dtype)

    def summary(self):
        """Print weight mapping summary."""
        print(f"Weight map: {len(self.weight_map)} entries")
        print(f"{'HF Key':<60} {'NNTrainer Layer':<30} {'Transform'}")
        print("-" * 100)
        for entry in self.weight_map:
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
        lines.append("            elif transform == 'reshape_2d' and "
                     "t.dim() == 4:")
        lines.append("                t = t.reshape(t.shape[0], "
                     "-1).contiguous()")
        lines.append("            elif transform == 'squeeze_3d' and "
                     "t.dim() == 3:")
        lines.append("                t = t.squeeze(1).contiguous()")
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
# Convenience functions
# =============================================================================

def convert_weights(layers, state_dict, output_path, dtype="float32"):
    """Convert HuggingFace weights to NNTrainer format.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        state_dict: HuggingFace model state_dict.
        output_path: Output .bin file path.
        dtype: Target dtype.

    Returns:
        str: Output file path.
    """
    converter = WeightConverter(layers)
    return converter.convert(state_dict, output_path, dtype)

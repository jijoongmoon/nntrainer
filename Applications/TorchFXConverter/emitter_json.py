"""
JSON configuration emitter for NNTrainer TorchFX converter.

Generates a JSON configuration file describing the model structure,
layer definitions, and weight mapping. This is useful for:
  - Tool-based processing and automation
  - Weight conversion scripts
  - Cross-platform model exchange
  - Debugging and visualization

Phase 4.3 of the TorchFX converter pipeline (DESIGN.md).
"""

import json


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that converts non-serializable objects to strings."""
    def default(self, o):
        return str(o)


def _json_safe(obj):
    """Convert non-JSON-serializable objects to strings."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _safe_properties(props):
    """Return a copy of properties dict with all values JSON-serializable."""
    return {k: _json_safe(v) for k, v in props.items()}


from emitter_base import BaseEmitter
from pattern_detector import ModelStructure
from nntrainer_layers import NNTrainerLayerDef


# =============================================================================
# JSON Emitter
# =============================================================================

class JsonEmitter(BaseEmitter):
    """Generates JSON model configuration from converter output.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        structure: ModelStructure from pattern detection.
    """

    def __init__(self, layers, structure, model_name=None):
        super().__init__(layers, structure, model_name=model_name)

    def emit(self):
        """Generate JSON configuration dict.

        Returns:
            dict: Complete model configuration.
        """
        config = {}
        config["model"] = self._emit_model_info()

        # Build layer groups (attention → mha_core, FFN → mlp, etc.)
        layer_to_group, groups = self._build_layer_groups()

        config["layers"] = self._emit_layers(layer_to_group)
        config["weight_map"] = self._emit_weight_map()
        if self.structure and self.structure.blocks:
            config["structure"] = self._emit_structure()
        if groups:
            config["groups"] = groups
        return config

    def emit_string(self, indent=2):
        """Generate JSON string.

        Returns:
            str: JSON-formatted string.
        """
        return json.dumps(self.emit(), indent=indent, ensure_ascii=False,
                          cls=_SafeEncoder)

    # =========================================================================
    # Model Info
    # =========================================================================

    def _emit_model_info(self):
        s = self.structure
        info = {
            "model_type": s.model_type if s else "",
            "arch_type": s.arch_type if s else "",
            "vocab_size": s.vocab_size if s else 0,
            "hidden_size": s.hidden_size if s else 0,
            "num_layers": s.num_layers if s else 0,
            "num_heads": s.num_heads if s else 0,
            "num_kv_heads": s.num_kv_heads if s else 0,
            "head_dim": s.head_dim if s else 0,
            "intermediate_size": s.intermediate_size if s else 0,
            "norm_eps": s.norm_eps if s else 0,
            "tie_word_embeddings": s.tie_word_embeddings if s else False,
        }
        if s and s.rope_theta:
            info["rope_theta"] = s.rope_theta
        return info

    # =========================================================================
    # Layer Definitions
    # =========================================================================

    def _emit_layers(self, layer_to_group=None):
        """Emit all layers as JSON objects."""
        result = []
        for layer in self.layers:
            entry = {
                "name": layer.name,
                "type": layer.layer_type,
            }
            if layer.input_layers:
                entry["input_layers"] = layer.input_layers
            if layer.properties:
                entry["properties"] = _safe_properties(layer.properties)
            if layer.hf_module_name:
                entry["hf_module_name"] = layer.hf_module_name
            if layer.hf_module_type:
                entry["hf_module_type"] = layer.hf_module_type
            if layer_to_group and layer.name in layer_to_group:
                entry["group"] = layer_to_group[layer.name]
            result.append(entry)
        return result

    # =========================================================================
    # Weight Map
    # =========================================================================

    def _emit_weight_map(self):
        """Emit HF state_dict key -> NNTrainer layer name mapping."""
        weight_map = []
        for layer in self.layers:
            if not layer.has_weight and not layer.has_bias:
                continue
            entry = {
                "layer_name": layer.name,
                "layer_type": layer.layer_type,
            }
            if layer.has_weight:
                entry["weight_key"] = layer.weight_hf_key
                entry["transpose_weight"] = layer.transpose_weight
            if layer.has_bias:
                entry["bias_key"] = layer.bias_hf_key
            if layer.shared_from:
                entry["shared_from"] = layer.shared_from
            if layer.weight_split:
                entry["weight_split"] = layer.weight_split
            weight_map.append(entry)
        return weight_map

    # =========================================================================
    # Layer Groups (collapsible attention / FFN groups)
    # =========================================================================

    def _build_layer_groups(self):
        """Build collapsible layer groups from detected model structure.

        Identifies which graph layers belong to each attention and FFN block,
        so viewers can collapse individual ops (matmul, softmax, reshape, ...)
        into higher-level NNTrainer layers (mha_core, mlp).

        Returns:
            (layer_to_group, groups):
                layer_to_group: dict mapping layer_name -> group_id
                groups: list of group descriptor dicts
        """
        s = self.structure
        if not s or not s.blocks:
            return {}, []

        layer_to_group = {}
        groups = []
        layer_names_set = {l.name for l in self.layers}

        for block in s.blocks:
            role = block.block_role or "block"
            bid = block.block_idx

            # --- Self-Attention group ---
            if block.attention:
                grp = self._make_attention_group(
                    block.attention, bid, role, "self_attention",
                    layer_names_set, layer_to_group)
                if grp:
                    groups.append(grp)

            # --- Cross-Attention group ---
            if block.cross_attention:
                grp = self._make_attention_group(
                    block.cross_attention, bid, role, "cross_attention",
                    layer_names_set, layer_to_group)
                if grp:
                    groups.append(grp)

            # --- FFN group ---
            if block.ffn:
                grp = self._make_ffn_group(
                    block.ffn, bid, role,
                    layer_names_set, layer_to_group)
                if grp:
                    groups.append(grp)

        return layer_to_group, groups

    @staticmethod
    def _common_prefix(a, b):
        """Find the longest common prefix up to a '_' scope boundary.

        For 'DenseReluDense_wi_0' vs 'DenseReluDense_wo', the raw prefix
        is 'DenseReluDense_w'.  We trim back to the last '_' to get
        'DenseReluDense' (the module scope).
        """
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        prefix = a[:i]
        # Trim to last '_' boundary for clean module scope
        last_sep = prefix.rfind("_")
        if last_sep > 0:
            return prefix[:last_sep]
        return prefix

    def _collect_members_by_prefix(self, prefix, layer_names_set,
                                   layer_to_group):
        """Collect all layers whose name starts with prefix + '_'.

        Skips layers already assigned to another group.
        """
        members = []
        prefix_u = prefix + "_"
        for l in self.layers:
            if l.name in layer_to_group:
                continue
            if l.name.startswith(prefix_u) and l.name in layer_names_set:
                members.append(l.name)
        return members

    def _make_attention_group(self, attn, block_idx, role, attn_kind,
                              layer_names_set, layer_to_group):
        """Build a group descriptor for an attention pattern."""
        if not attn.q_proj or not attn.k_proj:
            return None

        scope_prefix = self._common_prefix(attn.q_proj, attn.k_proj)
        if not scope_prefix:
            return None

        members = self._collect_members_by_prefix(
            scope_prefix, layer_names_set, layer_to_group)
        if not members:
            return None

        group_id = f"{role}_{block_idx}_{attn_kind}"
        for m in members:
            layer_to_group[m] = group_id

        grp = {
            "id": group_id,
            "type": attn_kind,
            "collapsed_as": "mha_core",
            "block_idx": block_idx,
            "block_type": role,
            "members": members,
            "properties": {
                "num_heads": attn.num_heads,
                "num_kv_heads": attn.num_kv_heads,
                "head_dim": attn.head_dim,
                "has_rope": attn.has_rope,
                "has_relative_position_bias": attn.has_relative_position_bias,
            },
        }
        return grp

    def _make_ffn_group(self, ffn, block_idx, role,
                        layer_names_set, layer_to_group):
        """Build a group descriptor for an FFN pattern."""
        # Find common prefix from two known FFN layers
        a = ffn.gate_proj or ffn.up_proj
        b = ffn.down_proj
        if not a or not b:
            return None

        scope_prefix = self._common_prefix(a, b)
        if not scope_prefix:
            return None

        members = self._collect_members_by_prefix(
            scope_prefix, layer_names_set, layer_to_group)
        if not members:
            return None

        group_id = f"{role}_{block_idx}_ffn"
        for m in members:
            layer_to_group[m] = group_id

        grp = {
            "id": group_id,
            "type": "ffn",
            "collapsed_as": "mlp",
            "block_idx": block_idx,
            "block_type": role,
            "members": members,
            "properties": {
                "ffn_type": ffn.ffn_type,
                "intermediate_size": ffn.intermediate_size,
            },
        }
        return grp

    # =========================================================================
    # Structure (from pattern detection)
    # =========================================================================

    def _emit_structure(self):
        """Emit detected model structure summary."""
        s = self.structure
        result = {
            "embedding": s.embedding,
            "final_norm": s.final_norm,
            "lm_head": s.lm_head,
            "blocks": [],
        }

        for block in s.blocks:
            b = {
                "block_idx": block.block_idx,
                "norm_type": block.norm_type,
                "pre_attn_norm": block.pre_attn_norm,
                "attn_residual": block.attn_residual,
                "pre_ffn_norm": block.pre_ffn_norm,
                "ffn_residual": block.ffn_residual,
            }
            if block.post_attn_norm:
                b["post_attn_norm"] = block.post_attn_norm
            if block.post_ffn_norm:
                b["post_ffn_norm"] = block.post_ffn_norm

            if block.attention:
                attn = block.attention
                b["attention"] = {
                    "type": attn.attention_type,
                    "q_proj": attn.q_proj,
                    "k_proj": attn.k_proj,
                    "v_proj": attn.v_proj,
                    "o_proj": attn.o_proj,
                    "has_qk_norm": attn.has_qk_norm,
                    "has_rope": attn.has_rope,
                    "num_heads": attn.num_heads,
                    "num_kv_heads": attn.num_kv_heads,
                    "head_dim": attn.head_dim,
                }
                if attn.use_sliding_window:
                    b["attention"]["use_sliding_window"] = True
                    if attn.sliding_window:
                        b["attention"]["sliding_window"] = attn.sliding_window
                if attn.q_norm:
                    b["attention"]["q_norm"] = attn.q_norm
                if attn.k_norm:
                    b["attention"]["k_norm"] = attn.k_norm

            if block.cross_attention:
                xattn = block.cross_attention
                b["cross_attention"] = {
                    "type": xattn.attention_type,
                    "q_proj": xattn.q_proj,
                    "k_proj": xattn.k_proj,
                    "v_proj": xattn.v_proj,
                    "o_proj": xattn.o_proj,
                }

            if block.ffn:
                ffn = block.ffn
                b["ffn"] = {
                    "type": ffn.ffn_type,
                    "intermediate_size": ffn.intermediate_size,
                }
                if ffn.ffn_type in ("swiglu", "geglu") or ffn.ffn_type.startswith("gated_"):
                    b["ffn"]["gate_proj"] = ffn.gate_proj
                    b["ffn"]["up_proj"] = ffn.up_proj
                    b["ffn"]["down_proj"] = ffn.down_proj
                else:
                    b["ffn"]["fc1"] = ffn.up_proj
                    b["ffn"]["fc2"] = ffn.down_proj

            result["blocks"].append(b)

        return result


# =============================================================================
# Convenience function
# =============================================================================

def emit_json(layers, structure):
    """Generate JSON model configuration.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        structure: ModelStructure from pattern detection.

    Returns:
        dict: Complete model configuration.
    """
    emitter = JsonEmitter(layers, structure)
    return emitter.emit()


def emit_json_string(layers, structure, indent=2):
    """Generate JSON string.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        structure: ModelStructure from pattern detection.
        indent: JSON indentation.

    Returns:
        str: JSON-formatted string.
    """
    emitter = JsonEmitter(layers, structure)
    return emitter.emit_string(indent=indent)

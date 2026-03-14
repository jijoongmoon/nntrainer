"""
Pattern detector for NNTrainer TorchFX converter.

Detects high-level structural patterns from flat layer lists produced by
the NodeMapper/AdaptiveConverter pipeline. Rather than matching fixed
subgraph templates, detection is driven by **module hierarchy** (hf_module_name)
and **data flow** (input_layers connections).

Detected patterns:
  - AttentionPattern: Q/K/V/O projections, optional Q/K norms, SDPA, RoPE
  - FFNPattern: SwiGLU (gate+up+act+mul+down) or GELU-FFN (fc1+act+fc2)
  - TransformerBlockPattern: Norm + Attention + Residual + Norm + FFN + Residual
  - ModelStructure: Full model overview (embedding, blocks, LM head)

Phase 3 of the TorchFX converter pipeline (DESIGN.md).
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from nntrainer_layers import (
    NNTrainerLayerDef,
    LAYER_FC, LAYER_EMBEDDING, LAYER_TIE_WORD_EMBEDDINGS,
    LAYER_RMS_NORM, LAYER_LAYER_NORM,
    LAYER_ADDITION, LAYER_ADD, LAYER_ACTIVATION, LAYER_MULTIPLY,
    LAYER_DROPOUT,
    OP_SDPA, OP_NOOP, OP_RESHAPE, OP_TRANSPOSE, OP_PERMUTE,
)


# =============================================================================
# Pattern Dataclasses
# =============================================================================

@dataclass
class AttentionPattern:
    """Detected multi-head attention pattern."""
    block_idx: int                          # Layer index (0, 1, 2, ...)
    q_proj: str = ""                        # Layer name of Q projection
    k_proj: str = ""                        # Layer name of K projection
    v_proj: str = ""                        # Layer name of V projection
    o_proj: str = ""                        # Layer name of O projection
    q_norm: str = ""                        # Optional Q post-projection norm
    k_norm: str = ""                        # Optional K post-projection norm
    sdpa: str = ""                          # SDPA layer name
    attention_type: str = "mha"             # "mha", "gqa", "mqa"
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    has_rope: bool = False                  # Whether RoPE ops are present
    has_qk_norm: bool = False               # Whether Q/K norms exist
    # Layer indices in the flat list (for slicing)
    layer_names: list = field(default_factory=list)

    @property
    def scope(self):
        """Common HF module scope (e.g. 'model.layers.0.self_attn')."""
        if self.q_proj:
            # "model_layers_0_self_attn_q_proj" -> need hf_module_name
            return ""
        return ""


@dataclass
class FFNPattern:
    """Detected feed-forward network pattern."""
    block_idx: int
    ffn_type: str = "standard"              # "swiglu", "gelu_ffn", "standard"
    gate_proj: str = ""                     # SwiGLU gate projection
    up_proj: str = ""                       # SwiGLU up projection (or fc1)
    down_proj: str = ""                     # SwiGLU down projection (or fc2)
    activation: str = ""                    # Activation layer name
    gate_multiply: str = ""                 # gate * up multiply layer
    intermediate_size: int = 0
    layer_names: list = field(default_factory=list)


@dataclass
class TransformerBlockPattern:
    """Detected transformer block (norm + attn + residual + norm + ffn + residual)."""
    block_idx: int
    block_role: str = ""                    # "encoder", "decoder", or "" (single-stack)
    pre_attn_norm: str = ""                 # Pre-attention normalization
    attention: Optional[AttentionPattern] = None
    post_attn_norm: str = ""                # Post-attention norm (Gemma3 style)
    attn_residual: str = ""                 # Attention residual addition
    pre_ffn_norm: str = ""                  # Pre-FFN normalization
    ffn: Optional[FFNPattern] = None
    post_ffn_norm: str = ""                 # Post-FFN norm (Gemma3 style)
    ffn_residual: str = ""                  # FFN residual addition
    norm_type: str = "pre_norm"             # "pre_norm" or "post_norm"
    # For encoder-decoder models
    cross_attention: Optional[AttentionPattern] = None
    cross_attn_norm: str = ""
    cross_attn_residual: str = ""


@dataclass
class ModelStructure:
    """Full model structure overview."""
    arch_type: str = ""                     # "decoder_only", "encoder_only", "encoder_decoder"
    model_type: str = ""                    # "qwen3", "bert", "t5", etc.
    embedding: str = ""                     # Embedding layer name
    blocks: list = field(default_factory=list)  # List of TransformerBlockPattern
    lm_head: str = ""                       # LM head layer name (if causal LM)
    final_norm: str = ""                    # Final normalization before LM head
    tie_word_embeddings: bool = False
    vocab_size: int = 0
    hidden_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    intermediate_size: int = 0
    rope_theta: float = 0.0
    norm_eps: float = 0.0
    max_position_embeddings: int = 0
    num_encoder_layers: int = 0
    num_decoder_layers: int = 0

    @property
    def encoder_blocks(self):
        """Return only encoder blocks."""
        return [b for b in self.blocks if b.block_role == "encoder"]

    @property
    def decoder_blocks(self):
        """Return only decoder blocks."""
        return [b for b in self.blocks if b.block_role == "decoder"]

    def summary(self):
        """Print detected model structure."""
        print(f"\n{'='*70}")
        print(f"DETECTED MODEL STRUCTURE")
        print(f"{'='*70}")
        print(f"Architecture: {self.arch_type} ({self.model_type})")
        print(f"Config: hidden={self.hidden_size}, layers={self.num_layers}, "
              f"heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
              f"head_dim={self.head_dim}")
        print(f"Intermediate: {self.intermediate_size}, "
              f"vocab: {self.vocab_size}")
        if self.rope_theta:
            print(f"RoPE theta: {self.rope_theta}")
        print(f"Embedding: {self.embedding}")
        if self.final_norm:
            print(f"Final norm: {self.final_norm}")
        if self.lm_head:
            print(f"LM head: {self.lm_head} "
                  f"(tied={self.tie_word_embeddings})")

        for block in self.blocks:
            _print_block(block)
        print(f"{'='*70}")


def _print_block(block):
    """Print a single transformer block."""
    role = f" [{block.block_role}]" if block.block_role else ""
    print(f"\n  Block {block.block_idx}{role}:")
    print(f"    Norm type: {block.norm_type}")
    if block.pre_attn_norm:
        print(f"    Pre-attn norm: {block.pre_attn_norm}")
    if block.post_attn_norm:
        print(f"    Post-attn norm: {block.post_attn_norm}")
    if block.attention:
        attn = block.attention
        print(f"    Attention ({attn.attention_type}):")
        print(f"      Q: {attn.q_proj}, K: {attn.k_proj}, V: {attn.v_proj}")
        if attn.has_qk_norm:
            print(f"      Q-norm: {attn.q_norm}, K-norm: {attn.k_norm}")
        print(f"      O: {attn.o_proj}")
        print(f"      heads={attn.num_heads}, kv_heads={attn.num_kv_heads}, "
              f"head_dim={attn.head_dim}")
        if attn.has_rope:
            print(f"      RoPE: yes")
    if block.attn_residual:
        print(f"    Attn residual: {block.attn_residual}")

    if block.cross_attention:
        xattn = block.cross_attention
        print(f"    Cross-attention ({xattn.attention_type}):")
        print(f"      Q: {xattn.q_proj}, K: {xattn.k_proj}, V: {xattn.v_proj}")
        print(f"      O: {xattn.o_proj}")

    if block.pre_ffn_norm:
        print(f"    Pre-FFN norm: {block.pre_ffn_norm}")
    if block.post_ffn_norm:
        print(f"    Post-FFN norm: {block.post_ffn_norm}")
    if block.ffn:
        ffn = block.ffn
        print(f"    FFN ({ffn.ffn_type}):")
        if ffn.ffn_type in ("swiglu", "geglu") or ffn.ffn_type.startswith("gated_"):
            print(f"      gate: {ffn.gate_proj}, up: {ffn.up_proj}, "
                  f"down: {ffn.down_proj}")
        else:
            print(f"      fc1: {ffn.up_proj}, fc2: {ffn.down_proj}")
        print(f"      intermediate_size={ffn.intermediate_size}")
    if block.ffn_residual:
        print(f"    FFN residual: {block.ffn_residual}")


# =============================================================================
# Pattern Detection Engine
# =============================================================================

class PatternDetector:
    """Detects structural patterns from a flat list of NNTrainerLayerDef.

    Detection is based on:
      1. Module hierarchy (hf_module_name) - identifies block/attention/MLP scope
      2. Layer types and connections (input_layers) - validates data flow
      3. Model config (optional) - provides architectural constants
    """

    def __init__(self, layers, model_config=None):
        """
        Args:
            layers: List of NNTrainerLayerDef from converter pipeline.
            model_config: HuggingFace model config (optional, for metadata).
        """
        self.layers = layers
        self.config = model_config
        # Build lookup: name -> layer
        self._by_name = {l.name: l for l in layers}
        # Build lookup: name -> index
        self._idx_by_name = {l.name: i for i, l in enumerate(layers)}

    def detect(self):
        """Run full pattern detection pipeline.

        Returns:
            ModelStructure with all detected patterns.
        """
        structure = ModelStructure()

        # Step 1: Extract model metadata from config
        self._extract_config_metadata(structure)

        # Step 2: Detect embedding and LM head
        self._detect_embedding_and_head(structure)

        # Step 3: Detect transformer blocks
        block_scopes = self._find_block_scopes()

        # Separate encoder and decoder scopes for proper numbering
        encoder_scopes = [s for s in block_scopes if "encoder" in s]
        decoder_scopes = [s for s in block_scopes if "decoder" in s]
        other_scopes = [s for s in block_scopes
                        if "encoder" not in s and "decoder" not in s]

        # Process encoder blocks first, then decoder blocks
        enc_idx = 0
        for scope in encoder_scopes:
            block = self._detect_block(enc_idx, scope)
            if block:
                block.block_role = "encoder"
                structure.blocks.append(block)
                enc_idx += 1
        structure.num_encoder_layers = enc_idx

        dec_idx = 0
        for scope in decoder_scopes:
            block = self._detect_block(dec_idx, scope)
            if block:
                block.block_role = "decoder"
                structure.blocks.append(block)
                dec_idx += 1
        structure.num_decoder_layers = dec_idx

        # Single-stack models (Qwen3, BERT, GPT-2, etc.)
        for block_idx, scope in enumerate(other_scopes):
            block = self._detect_block(block_idx, scope)
            if block:
                structure.blocks.append(block)

        structure.num_layers = len(structure.blocks)

        # Step 4: Detect final normalization
        self._detect_final_norm(structure)

        # Step 5: Infer architecture type
        self._infer_arch_type(structure)

        return structure

    # =========================================================================
    # Config Metadata Extraction
    # =========================================================================

    @staticmethod
    def _safe_cfg_int(cfg, *attrs, default=0):
        """Safely get an int config value, falling back through attrs."""
        for attr in attrs:
            val = getattr(cfg, attr, None)
            if val is not None and isinstance(val, (int, float)):
                return int(val)
        return default

    @staticmethod
    def _safe_cfg_float(cfg, *attrs, default=0.0):
        """Safely get a float config value, falling back through attrs."""
        for attr in attrs:
            val = getattr(cfg, attr, None)
            if val is not None and isinstance(val, (int, float)):
                return float(val)
        return default

    def _extract_config_metadata(self, structure):
        """Extract model metadata from HF config."""
        cfg = self.config
        if cfg is None:
            return

        model_type = getattr(cfg, "model_type", "")
        structure.model_type = model_type if isinstance(model_type, str) else str(model_type)
        structure.vocab_size = self._safe_cfg_int(cfg, "vocab_size")
        structure.hidden_size = self._safe_cfg_int(cfg, "hidden_size", "d_model")
        structure.num_heads = self._safe_cfg_int(cfg, "num_attention_heads", "num_heads")
        structure.num_kv_heads = self._safe_cfg_int(
            cfg, "num_key_value_heads", default=structure.num_heads)
        structure.head_dim = self._safe_cfg_int(
            cfg, "head_dim", "d_kv",
            default=(structure.hidden_size // structure.num_heads
                     if structure.num_heads else 0))
        structure.intermediate_size = self._safe_cfg_int(
            cfg, "intermediate_size", "d_ff")
        # rope_theta may be in config directly or in rope_parameters dict
        structure.rope_theta = self._safe_cfg_float(cfg, "rope_theta")
        if not structure.rope_theta:
            rope_params = getattr(cfg, "rope_parameters", None)
            if isinstance(rope_params, dict):
                rt = rope_params.get("rope_theta", rope_params.get("base", None))
                if isinstance(rt, (int, float)):
                    structure.rope_theta = float(rt)
        structure.norm_eps = self._safe_cfg_float(
            cfg, "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon")
        tie = getattr(cfg, "tie_word_embeddings", False)
        structure.tie_word_embeddings = bool(tie) if not callable(tie) else False
        structure.max_position_embeddings = self._safe_cfg_int(
            cfg, "max_position_embeddings", default=2048)

    # =========================================================================
    # Embedding & Head Detection
    # =========================================================================

    def _detect_embedding_and_head(self, structure):
        """Find embedding layer and LM head."""
        for layer in self.layers:
            if layer.layer_type == LAYER_EMBEDDING:
                structure.embedding = layer.name
                if not structure.vocab_size and layer.properties:
                    structure.vocab_size = int(layer.properties.get(
                        "num_embeddings", 0))
                break

        # LM head: FC layer at the end, or tied embedding
        for layer in reversed(self.layers):
            if layer.layer_type == LAYER_TIE_WORD_EMBEDDINGS:
                structure.lm_head = layer.name
                structure.tie_word_embeddings = True
                break
            if layer.layer_type == LAYER_FC:
                # Check if it looks like LM head (name contains "lm_head")
                if "lm_head" in layer.hf_module_name:
                    structure.lm_head = layer.name
                    break

    # =========================================================================
    # Block Scope Discovery
    # =========================================================================

    def _find_block_scopes(self):
        """Find all transformer block scopes from layer names.

        Looks for patterns like:
          - model.layers.0, model.layers.1, ... (Qwen3, LLaMA)
          - encoder.layer.0, encoder.layer.1, ... (BERT)
          - encoder.block.0, decoder.block.0, ... (T5/mT5)

        Returns:
            Sorted list of block scope strings.
        """
        block_scopes = set()
        # Common block path patterns
        block_patterns = [
            # Qwen3/LLaMA/Mistral: model.layers.N
            r"(model\.layers\.\d+)",
            # Gemma/LLaMA base model (AutoModel): layers.N (no model. prefix)
            r"(layers\.\d+)",
            # BERT: encoder.layer.N / bert.encoder.layer.N
            r"((?:bert\.)?encoder\.layer\.\d+)",
            # T5/mT5: encoder.block.N / decoder.block.N
            r"((?:encoder|decoder)\.block\.\d+)",
            # GPT-2: transformer.h.N / h.N
            r"((?:transformer\.)?h\.\d+)",
        ]

        for layer in self.layers:
            name = layer.hf_module_name
            if not name:
                continue
            for pattern in block_patterns:
                m = re.match(pattern, name)
                if m:
                    block_scopes.add(m.group(1))
                    break

        # Sort by block index numerically
        def _sort_key(scope):
            # Extract all numbers, use them as sort key
            parts = scope.split(".")
            key = []
            for p in parts:
                if p.isdigit():
                    key.append((0, int(p)))
                else:
                    key.append((1, p))
            return key

        return sorted(block_scopes, key=_sort_key)

    # =========================================================================
    # Block Detection
    # =========================================================================

    def _detect_block(self, block_idx, scope):
        """Detect a single transformer block's internal structure.

        Args:
            block_idx: Block number (0-based).
            scope: HF module scope (e.g. "model.layers.0").
        """
        block_layers = self._get_layers_in_scope(scope)
        if not block_layers:
            return None

        block = TransformerBlockPattern(block_idx=block_idx)

        # Detect attention sub-pattern
        attn_scope = self._find_attention_scope(scope, block_layers)
        if attn_scope:
            block.attention = self._detect_attention(block_idx, attn_scope,
                                                     block_layers)

        # Detect cross-attention (encoder-decoder models)
        cross_attn_scope = self._find_cross_attention_scope(scope, block_layers)
        if cross_attn_scope:
            block.cross_attention = self._detect_attention(
                block_idx, cross_attn_scope, block_layers)

        # Detect FFN sub-pattern
        ffn_scope = self._find_ffn_scope(scope, block_layers)
        if ffn_scope:
            block.ffn = self._detect_ffn(block_idx, ffn_scope, block_layers)

        # Detect norms and residuals
        self._detect_norms_and_residuals(block, scope, block_layers)

        return block

    def _get_layers_in_scope(self, scope):
        """Get all layers whose hf_module_name or sanitized name starts with scope."""
        # Sanitized scope for name matching (model.layers.0 -> model_layers_0)
        sanitized = scope.replace(".", "_")
        result = []
        for l in self.layers:
            if (l.hf_module_name.startswith(scope + ".")
                or l.hf_module_name == scope
                or (not l.hf_module_name and l.name.startswith(sanitized + "_"))):
                result.append(l)
        return result

    def _find_attention_scope(self, block_scope, block_layers):
        """Find the self-attention scope within a block."""
        # Match attention scope by finding it anywhere in the hf_module_name
        # after the block scope prefix
        attn_keywords = [
            "self_attn",          # Qwen3, LLaMA, Mistral
            "attention.self",     # BERT (encoder.layer.N.attention.self.query)
            "SelfAttention",      # T5 (block.N.layer.0.SelfAttention)
            "self_attention",
            "attn",               # GPT-2
        ]

        for layer in block_layers:
            name = layer.hf_module_name
            if not name.startswith(block_scope):
                continue
            remainder = name[len(block_scope):]
            for kw in attn_keywords:
                idx = remainder.find(kw)
                if idx >= 0:
                    return block_scope + remainder[:idx + len(kw)]

        # Generic fallback: "attention" (less specific, try last)
        for layer in block_layers:
            name = layer.hf_module_name
            if not name.startswith(block_scope):
                continue
            remainder = name[len(block_scope):]
            idx = remainder.find("attention")
            if idx >= 0:
                return block_scope + remainder[:idx + len("attention")]

        return None

    def _find_cross_attention_scope(self, block_scope, block_layers):
        """Find cross-attention scope (T5/mT5 encoder-decoder)."""
        cross_keywords = ["EncDecAttention", "crossattention",
                          "cross_attn", "encoder_attn"]
        for layer in block_layers:
            name = layer.hf_module_name
            if not name.startswith(block_scope):
                continue
            for kw in cross_keywords:
                if kw in name:
                    idx = name.find(kw)
                    return name[:idx + len(kw)]
        return None

    def _find_ffn_scope(self, block_scope, block_layers):
        """Find the FFN/MLP scope within a block."""
        # For BERT-style models, FFN spans intermediate + output
        # We return the block_scope itself and detect FC layers by name
        ffn_patterns = ["mlp", "feed_forward", "ffn", "DenseReluDense"]
        for pat in ffn_patterns:
            full = f"{block_scope}.{pat}"
            for layer in block_layers:
                name = layer.hf_module_name
                if name.startswith(full + ".") or name == full:
                    return full

        # T5-style: block.N.layer.1.DenseReluDense or block.N.layer.2.DenseReluDense
        for layer in block_layers:
            name = layer.hf_module_name
            if "DenseReluDense" in name:
                idx = name.find("DenseReluDense")
                return name[:idx + len("DenseReluDense")]

        # BERT-style: intermediate.dense + output.dense under block scope
        has_intermediate = any(
            l.hf_module_name.startswith(f"{block_scope}.intermediate")
            for l in block_layers)
        if has_intermediate:
            return block_scope  # FFN is directly under block scope

        return None

    # =========================================================================
    # Attention Pattern Detection
    # =========================================================================

    def _detect_attention(self, block_idx, attn_scope, block_layers):
        """Detect attention pattern within the given scope."""
        attn = AttentionPattern(block_idx=block_idx)

        # Find Q/K/V/O projections by name
        attn_layers = [l for l in block_layers
                       if l.hf_module_name.startswith(attn_scope)]

        for layer in attn_layers:
            name_suffix = layer.hf_module_name[len(attn_scope):].lstrip(".")

            if layer.layer_type == LAYER_FC:
                if name_suffix in ("q_proj", "q", "query", "q_proj.0"):
                    attn.q_proj = layer.name
                    attn.layer_names.append(layer.name)
                    # Extract num_heads from Q output size
                    unit = int(layer.properties.get("unit", 0))
                    if unit and self.config:
                        attn.num_heads = getattr(self.config,
                                                 "num_attention_heads", 0)
                        attn.head_dim = getattr(self.config, "head_dim",
                                                unit // attn.num_heads
                                                if attn.num_heads else 0)
                elif name_suffix in ("k_proj", "k", "key", "k_proj.0"):
                    attn.k_proj = layer.name
                    attn.layer_names.append(layer.name)
                    # Extract num_kv_heads from K output size
                    unit = int(layer.properties.get("unit", 0))
                    if unit and attn.head_dim:
                        attn.num_kv_heads = unit // attn.head_dim
                elif name_suffix in ("v_proj", "v", "value", "v_proj.0"):
                    attn.v_proj = layer.name
                    attn.layer_names.append(layer.name)
                elif name_suffix in ("o_proj", "out_proj", "o"):
                    attn.o_proj = layer.name
                    attn.layer_names.append(layer.name)

            elif layer.layer_type in (LAYER_RMS_NORM, LAYER_LAYER_NORM):
                if name_suffix in ("q_norm", "q_layernorm"):
                    attn.q_norm = layer.name
                    attn.has_qk_norm = True
                    attn.layer_names.append(layer.name)
                elif name_suffix in ("k_norm", "k_layernorm"):
                    attn.k_norm = layer.name
                    attn.layer_names.append(layer.name)

            elif layer.layer_type == OP_SDPA:
                attn.sdpa = layer.name
                attn.layer_names.append(layer.name)

        # O projection may be outside the self-attention scope
        # (BERT: attention.output.dense is outside attention.self scope)
        if not attn.o_proj:
            parent_scope = attn_scope.rsplit(".", 1)[0] if "." in attn_scope else ""
            for layer in block_layers:
                if layer.layer_type != LAYER_FC:
                    continue
                hf = layer.hf_module_name
                if parent_scope and hf.startswith(parent_scope):
                    suffix = hf[len(parent_scope):].lstrip(".")
                    if suffix in ("output.dense", "dense"):
                        attn.o_proj = layer.name
                        attn.layer_names.append(layer.name)
                        break

        # Detect RoPE: look for sin/cos ops in block scope, attention scope,
        # or global rotary_emb (Qwen3/LLaMA share rotary_emb across layers)
        for layer in self.layers:
            if layer.layer_type in ("sin", "cos"):
                name_lower = layer.name.lower()
                scope_lower = layer.hf_module_name.lower()
                if ("rotary" in name_lower or "rotary" in scope_lower
                    or "rope" in name_lower or "rope" in scope_lower
                    or scope_lower.startswith(attn_scope)):
                    attn.has_rope = True
                    break
        # Also check config for rope_theta
        if not attn.has_rope and self.config:
            if getattr(self.config, "rope_theta", 0) > 0:
                attn.has_rope = True

        # Determine attention type
        if attn.num_heads and attn.num_kv_heads:
            if attn.num_kv_heads == 1:
                attn.attention_type = "mqa"
            elif attn.num_kv_heads < attn.num_heads:
                attn.attention_type = "gqa"
            else:
                attn.attention_type = "mha"

        # Fallback: use config for kv_heads if not detected from shapes
        if not attn.num_kv_heads and self.config:
            attn.num_kv_heads = getattr(self.config, "num_key_value_heads",
                                        attn.num_heads)

        return attn

    # =========================================================================
    # FFN Pattern Detection
    # =========================================================================

    def _detect_ffn(self, block_idx, ffn_scope, block_layers):
        """Detect FFN pattern within the given scope."""
        ffn = FFNPattern(block_idx=block_idx)

        ffn_layers = [l for l in block_layers
                      if l.hf_module_name.startswith(ffn_scope)]

        # Collect FC layers in FFN scope
        fc_layers = [(l, l.hf_module_name[len(ffn_scope):].lstrip("."))
                     for l in ffn_layers if l.layer_type == LAYER_FC]

        # Detect SwiGLU pattern: gate_proj + up_proj + down_proj
        gate = up = down = None
        for layer, suffix in fc_layers:
            if suffix in ("gate_proj", "gate"):
                gate = layer
            elif suffix in ("up_proj", "up", "wi_1", "w1"):
                up = layer
            elif suffix in ("down_proj", "down", "wo", "w2"):
                down = layer
            # BERT/T5 style: wi_0 = gate, wi = up (T5 DenseReluDense)
            elif suffix in ("wi_0",):
                gate = layer
            elif suffix in ("wi",):
                up = layer
            # BERT style: intermediate.dense = fc1, output.dense = fc2
            elif suffix in ("intermediate.dense",):
                up = layer
            elif suffix in ("output.dense",):
                down = layer

        if gate and up and down:
            ffn.ffn_type = "swiglu"
            ffn.gate_proj = gate.name
            ffn.up_proj = up.name
            ffn.down_proj = down.name
            ffn.intermediate_size = int(gate.properties.get("unit", 0))
            ffn.layer_names = [gate.name, up.name, down.name]

            # Find activation and multiply in the block
            for layer in block_layers:
                if (layer.layer_type == LAYER_ACTIVATION
                    and layer.hf_module_name.startswith(ffn_scope)):
                    ffn.activation = layer.name
                    ffn.layer_names.append(layer.name)
                    # Distinguish GeGLU from SwiGLU based on activation
                    act_type = layer.properties.get("activation", "")
                    if act_type == "gelu":
                        ffn.ffn_type = "geglu"
                    elif act_type in ("relu", "tanh", "sigmoid"):
                        ffn.ffn_type = f"gated_{act_type}"
                elif (layer.layer_type == LAYER_MULTIPLY
                      and layer.hf_module_name.startswith(ffn_scope)):
                    ffn.gate_multiply = layer.name
                    ffn.layer_names.append(layer.name)

        elif len(fc_layers) >= 2 and not gate:
            # GELU-FFN or standard FFN: fc1 -> act -> fc2
            # T5-style: wi -> act -> wo
            ffn.ffn_type = "gelu_ffn"
            if up and down:
                ffn.up_proj = up.name
                ffn.down_proj = down.name
            else:
                # Use positional: first FC = fc1/up, last FC = fc2/down
                ffn.up_proj = fc_layers[0][0].name
                ffn.down_proj = fc_layers[-1][0].name
            ffn.intermediate_size = int(
                self._by_name.get(ffn.up_proj, NNTrainerLayerDef(
                    layer_type="", name="")).properties.get("unit", 0))
            ffn.layer_names = [ffn.up_proj, ffn.down_proj]

            # Find activation
            for layer in block_layers:
                if (layer.layer_type == LAYER_ACTIVATION
                    and layer.hf_module_name.startswith(ffn_scope)):
                    ffn.activation = layer.name
                    ffn.layer_names.append(layer.name)
                    act_type = layer.properties.get("activation", "")
                    if act_type == "gelu":
                        ffn.ffn_type = "gelu_ffn"
                    elif act_type == "relu":
                        ffn.ffn_type = "standard"

        return ffn

    # =========================================================================
    # Norm & Residual Detection
    # =========================================================================

    def _detect_norms_and_residuals(self, block, scope, block_layers):
        """Detect normalization layers and residual connections."""
        norms = []
        residuals = []

        for layer in block_layers:
            if layer.layer_type in (LAYER_RMS_NORM, LAYER_LAYER_NORM):
                # Skip Q/K norms (already detected in attention)
                if block.attention and layer.name in (
                    block.attention.q_norm, block.attention.k_norm):
                    continue
                norms.append(layer)
            elif layer.layer_type in (LAYER_ADDITION, LAYER_ADD):
                residuals.append(layer)

        # Assign norms: first = pre-attention, second = pre-FFN
        # (In T5-style blocks, there may be 3: pre-attn, pre-cross-attn, pre-ffn)
        norm_names_suffixes = []
        for n in norms:
            suffix = n.hf_module_name[len(scope):].lstrip(".")
            norm_names_suffixes.append((n, suffix))

        for norm, suffix in norm_names_suffixes:
            # T5-style: layer.0 = self-attn, layer.1 = cross-attn, layer.2 = FFN
            if any(kw in suffix for kw in ("input_layernorm",
                                           "layer.0.layer_norm",
                                           "pre_attention")):
                block.pre_attn_norm = norm.name
            elif "post_attention_layernorm" in suffix:
                block.post_attn_norm = norm.name
            elif "pre_feedforward_layernorm" in suffix:
                block.pre_ffn_norm = norm.name
            elif "post_feedforward_layernorm" in suffix:
                block.post_ffn_norm = norm.name
            elif any(kw in suffix for kw in ("pre_ffn", "final_layer_norm")):
                block.pre_ffn_norm = norm.name
            elif "layer.1.layer_norm" in suffix:
                # T5: layer.1 = cross-attn norm (decoder) or FFN norm (encoder)
                if block.cross_attention:
                    block.cross_attn_norm = norm.name
                else:
                    block.pre_ffn_norm = norm.name
            elif "layer.2.layer_norm" in suffix:
                block.pre_ffn_norm = norm.name
            elif any(kw in suffix for kw in ("attention.output.LayerNorm",)):
                block.pre_attn_norm = norm.name
            elif any(kw in suffix for kw in ("output.LayerNorm",)):
                block.pre_ffn_norm = norm.name

        # For models with only post_attn_norm (no separate pre_ffn_norm),
        # use post_attn_norm as pre_ffn_norm (e.g. Qwen3, Gemma1, LLaMA)
        if block.post_attn_norm and not block.pre_ffn_norm:
            block.pre_ffn_norm = block.post_attn_norm
            block.post_attn_norm = ""  # It's not a separate post-attn norm

        # If we couldn't assign by name, assign by position
        if not block.pre_attn_norm and not block.pre_ffn_norm and len(norms) >= 2:
            block.pre_attn_norm = norms[0].name
            block.pre_ffn_norm = norms[1].name
        elif not block.pre_attn_norm and len(norms) >= 1:
            block.pre_attn_norm = norms[0].name

        # Assign residuals: first = attention residual, last = FFN residual
        if len(residuals) >= 2:
            block.attn_residual = residuals[0].name
            block.ffn_residual = residuals[-1].name
            if len(residuals) >= 3 and block.cross_attention:
                block.cross_attn_residual = residuals[1].name
        elif len(residuals) == 1:
            block.attn_residual = residuals[0].name

        # Detect norm type (pre-norm vs post-norm)
        # Pre-norm: norm comes BEFORE attention (LLaMA, Qwen3)
        # Post-norm: norm comes AFTER attention (original Transformer, BERT)
        if block.pre_attn_norm and block.attention and block.attention.q_proj:
            norm_idx = self._idx_by_name.get(block.pre_attn_norm, 0)
            q_idx = self._idx_by_name.get(block.attention.q_proj, 0)
            if norm_idx < q_idx:
                block.norm_type = "pre_norm"
            else:
                block.norm_type = "post_norm"

    # =========================================================================
    # Final Norm Detection
    # =========================================================================

    def _detect_final_norm(self, structure):
        """Detect the final normalization before LM head."""
        # Look for norm layer after last block but before LM head
        for layer in reversed(self.layers):
            if layer.layer_type in (LAYER_RMS_NORM, LAYER_LAYER_NORM):
                # Check it's not inside a block
                if layer.hf_module_name and not any(
                    layer.hf_module_name.startswith(scope)
                    for scope in self._find_block_scopes()
                    if scope + "." in layer.hf_module_name
                ):
                    # Final model norm (e.g. model.norm, encoder.final_layer_norm)
                    structure.final_norm = layer.name
                    break

    # =========================================================================
    # Architecture Type Inference
    # =========================================================================

    def _infer_arch_type(self, structure):
        """Infer architecture type from detected patterns."""
        has_cross_attn = any(b.cross_attention for b in structure.blocks)

        # Check config model_type first for known architectures
        if self.config:
            model_type = getattr(self.config, "model_type", "")
            if model_type in ("bert", "roberta", "distilbert", "albert",
                              "electra", "camembert", "xlm-roberta"):
                structure.arch_type = "encoder_only"
                return
            elif model_type in ("t5", "mt5", "bart", "mbart",
                                "pegasus", "marian"):
                structure.arch_type = "encoder_decoder"
                return

            # Check if it's an embedding model (base model without LM head)
            # These use decoder-like architecture but serve as encoders
            architectures = getattr(self.config, "architectures", []) or []
            is_base_model = any(
                arch.endswith("Model") and not arch.endswith(("ForCausalLM",
                    "ForConditionalGeneration", "ForSeq2SeqLM"))
                for arch in architectures
            )
            if is_base_model and not structure.lm_head:
                structure.arch_type = "embedding"
                return

        if has_cross_attn:
            structure.arch_type = "encoder_decoder"
        elif structure.lm_head or structure.tie_word_embeddings:
            structure.arch_type = "decoder_only"
        else:
            # Fallback: check if block scopes suggest encoder/decoder
            block_scopes = self._find_block_scopes()
            has_encoder = any("encoder" in s for s in block_scopes)
            has_decoder = any("decoder" in s for s in block_scopes)
            if has_encoder and has_decoder:
                structure.arch_type = "encoder_decoder"
            elif has_encoder:
                structure.arch_type = "encoder_only"
            else:
                structure.arch_type = "decoder_only"


# =============================================================================
# Convenience function
# =============================================================================

def detect_patterns(layers, model_config=None):
    """Detect structural patterns from a flat layer list.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        model_config: HuggingFace model config (optional).

    Returns:
        ModelStructure with all detected patterns.
    """
    detector = PatternDetector(layers, model_config)
    return detector.detect()

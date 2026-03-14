"""
INI configuration emitter for NNTrainer TorchFX converter.

Generates .ini configuration files that can be loaded directly by NNTrainer
via model->load("model.ini", MODEL_FORMAT_INI).

INI format matches NNTrainer's standard configuration:
  [Model]        - Network type, batch size
  [LayerName]    - Type, input_layers, layer-specific properties

Phase 4.2 of the TorchFX converter pipeline (DESIGN.md).
"""

from pattern_detector import ModelStructure, TransformerBlockPattern
from nntrainer_layers import NNTrainerLayerDef


# =============================================================================
# INI Emitter
# =============================================================================

class IniEmitter:
    """Generates NNTrainer INI configuration from converter output.

    Two modes:
      1. Flat mode: Emit every layer from the flat layer list (verbose, exact).
      2. Structured mode: Use ModelStructure to emit a clean, readable config
         with block structure and proper naming.
    """

    def __init__(self, layers, structure, batch_size=1):
        self.layers = layers
        self.structure = structure
        self.batch_size = batch_size
        self._by_name = {l.name: l for l in layers}

    def emit(self, mode="structured"):
        """Generate INI configuration string.

        Args:
            mode: "structured" uses pattern-detected structure for clean output.
                  "flat" emits every layer from the flat list verbatim.
                  When structured mode is requested but no transformer blocks
                  are detected, falls back to flat mode automatically.

        Returns:
            str: Complete INI file content.
        """
        if mode == "flat":
            return self._emit_flat()
        # Fall back to flat mode for non-transformer models
        if not self.structure.blocks:
            return self._emit_flat()
        return self._emit_structured()

    # =========================================================================
    # Structured Mode (pattern-based)
    # =========================================================================

    def _emit_structured(self):
        s = self.structure
        sections = []

        # File header comment
        sections.append(f"# Auto-generated NNTrainer configuration")
        sections.append(f"# Model: {s.model_type} ({s.arch_type})")
        sections.append("")

        # [Model] section
        sections.append("[Model]")
        sections.append("Type = NeuralNetwork")
        sections.append(f"batch_size = {self.batch_size}")
        sections.append("")

        # Input layer
        sections.append("[input0]")
        sections.append("Type = input")
        sections.append(f"Input_Shape = 1:1:1")
        sections.append("")

        # Embedding
        if s.embedding:
            # Only use tie_word_embeddings if there's an LM head to tie to
            emb_type = ("tie_word_embeddings"
                        if s.tie_word_embeddings and s.lm_head
                        else "embedding_layer")
            sections.append("[embedding0]")
            sections.append(f"Type = {emb_type}")
            sections.append("input_layers = input0")
            sections.append(f"in_dim = {s.vocab_size}")
            sections.append(f"out_dim = {s.hidden_size}")
            sections.append("")

        # Transformer blocks
        first_input = "embedding0" if s.embedding else "input0"
        norm_type = self._norm_type(s)

        if s.arch_type == "encoder_decoder":
            # Encoder blocks
            enc_blocks = s.encoder_blocks
            enc_b0 = enc_blocks[0] if enc_blocks else None
            for i in range(s.num_encoder_layers):
                input_name = (first_input if i == 0
                              else f"enc_layer{i-1}_block_output")
                sections.extend(
                    self._emit_block_ini(i, input_name, s,
                                         block=enc_b0, prefix=f"enc_layer{i}"))
                sections.append("")

            # Encoder final norm
            enc_last = f"enc_layer{s.num_encoder_layers - 1}_block_output"
            sections.append("[encoder_output_norm]")
            sections.append(f"Type = {norm_type}")
            sections.append(f"input_layers = {enc_last}")
            sections.append(f"epsilon = {s.norm_eps}")
            if norm_type == "rms_norm":
                sections.append("packed = false")
            sections.append("")

            # Decoder blocks (with cross-attention)
            dec_blocks = s.decoder_blocks
            dec_b0 = dec_blocks[0] if dec_blocks else None
            for i in range(s.num_decoder_layers):
                input_name = (first_input if i == 0
                              else f"dec_layer{i-1}_block_output")
                sections.extend(
                    self._emit_block_ini(
                        i, input_name, s,
                        block=dec_b0, prefix=f"dec_layer{i}",
                        encoder_output="encoder_output_norm"))
                sections.append("")

            # Decoder final norm
            if s.final_norm:
                dec_last = (f"dec_layer{s.num_decoder_layers - 1}"
                            f"_block_output")
                sections.append("[decoder_output_norm]")
                sections.append(f"Type = {norm_type}")
                sections.append(f"input_layers = {dec_last}")
                sections.append(f"epsilon = {s.norm_eps}")
                if norm_type == "rms_norm":
                    sections.append("packed = false")
                sections.append("")

            # LM head
            if s.lm_head:
                lm_type = ("tie_word_embeddings" if s.tie_word_embeddings
                            else "fully_connected")
                sections.append("[lm_head]")
                sections.append(f"Type = {lm_type}")
                sections.append("input_layers = decoder_output_norm")
                sections.append(f"unit = {s.vocab_size}")
                sections.append("disable_bias = true")
                if s.tie_word_embeddings:
                    sections.append("shared_from = embedding0")
                sections.append("")
        else:
            # Single-stack (decoder_only / encoder_only)
            for i in range(s.num_layers):
                input_name = (first_input if i == 0
                              else f"layer{i-1}_decoder_output")
                sections.extend(
                    self._emit_block_ini(i, input_name, s))
                sections.append("")

            # Final norm
            if s.final_norm:
                last_block = f"layer{s.num_layers - 1}_decoder_output"
                sections.append("[output_norm]")
                sections.append(f"Type = {norm_type}")
                sections.append(f"input_layers = {last_block}")
                sections.append(f"epsilon = {s.norm_eps}")
                if norm_type == "rms_norm":
                    sections.append("packed = false")
                sections.append("")

            # LM head
            if s.lm_head:
                lm_type = ("tie_word_embeddings" if s.tie_word_embeddings
                            else "fully_connected")
                sections.append("[lm_head]")
                sections.append(f"Type = {lm_type}")
                sections.append("input_layers = output_norm")
                sections.append(f"unit = {s.vocab_size}")
                sections.append("disable_bias = true")
                if s.tie_word_embeddings:
                    sections.append("shared_from = embedding0")
                sections.append("")

        return "\n".join(sections)

    def _emit_block_ini(self, layer_id, input_name, s,
                        block=None, prefix=None, encoder_output=None):
        """Emit a single transformer block as INI sections.

        Args:
            layer_id: Block index.
            input_name: Name of the input layer.
            s: ModelStructure.
            block: TransformerBlockPattern to use (default: s.blocks[0]).
            prefix: Layer name prefix (default: "layer{layer_id}").
            encoder_output: Encoder output for cross-attention (decoder only).
        """
        lines = []
        if prefix is None:
            prefix = f"layer{layer_id}"
        b0 = block if block is not None else (s.blocks[0] if s.blocks else None)
        norm_type = self._norm_type(s)

        role = f" [{b0.block_role}]" if b0 and b0.block_role else ""
        lines.append(f"# --- Block {layer_id}{role} ---")

        # Pre-attention norm
        if b0 and b0.pre_attn_norm:
            norm_name = f"{prefix}_attention_norm"
            lines.append(f"[{norm_name}]")
            lines.append(f"Type = {norm_type}")
            lines.append(f"input_layers = {input_name}")
            lines.append(f"epsilon = {s.norm_eps}")
            if norm_type == "rms_norm":
                lines.append("packed = false")
            lines.append("")
            attn_input = norm_name
        else:
            attn_input = input_name

        # Attention sub-layers
        if b0 and b0.attention:
            attn = b0.attention
            has_qk_norm = attn.has_qk_norm
            q_unit = s.head_dim * s.num_heads
            kv_unit = s.head_dim * s.num_kv_heads

            # Q projection
            lines.append(f"[{prefix}_wq]")
            lines.append("Type = fully_connected")
            lines.append(f"input_layers = {attn_input}")
            lines.append(f"unit = {q_unit}")
            lines.append("disable_bias = true")
            lines.append("")

            # K projection
            lines.append(f"[{prefix}_wk]")
            lines.append("Type = fully_connected")
            lines.append(f"input_layers = {attn_input}")
            lines.append(f"unit = {kv_unit}")
            lines.append("disable_bias = true")
            lines.append("")

            # V projection
            lines.append(f"[{prefix}_wv]")
            lines.append("Type = fully_connected")
            lines.append(f"input_layers = {attn_input}")
            lines.append(f"unit = {kv_unit}")
            lines.append("disable_bias = true")
            lines.append("")

            # Q/K norms
            q_input = f"{prefix}_wq"
            k_input = f"{prefix}_wk"
            if has_qk_norm:
                lines.append(f"[{prefix}_q_norm]")
                lines.append("Type = reshaped_rms_norm")
                lines.append(f"input_layers = {prefix}_wq")
                lines.append("packed = false")
                lines.append(f"epsilon = {s.norm_eps}")
                lines.append(f"feature_size = {s.head_dim}")
                lines.append("")

                lines.append(f"[{prefix}_k_norm]")
                lines.append("Type = reshaped_rms_norm")
                lines.append(f"input_layers = {prefix}_wk")
                lines.append("packed = false")
                lines.append(f"epsilon = {s.norm_eps}")
                lines.append(f"feature_size = {s.head_dim}")
                lines.append("")
                q_input = f"{prefix}_q_norm"
                k_input = f"{prefix}_k_norm"

            # MHA core
            lines.append(f"[{prefix}_attention]")
            lines.append("Type = mha_core")
            lines.append(f"input_layers = {q_input},{k_input},{prefix}_wv")
            lines.append(f"num_heads = {s.num_heads}")
            lines.append(f"num_heads_kv = {s.num_kv_heads}")
            if attn.has_rope and s.rope_theta:
                lines.append(f"rope_theta = {int(s.rope_theta)}")
            lines.append("")

            # O projection
            lines.append(f"[{prefix}_attention_out]")
            lines.append("Type = fully_connected")
            lines.append(f"input_layers = {prefix}_attention")
            lines.append(f"unit = {s.hidden_size}")
            lines.append("disable_bias = true")
            lines.append("")

        # Self-attention residual
        is_enc_dec = (b0 and b0.block_role in ("encoder", "decoder"))
        attn_add_name = "self_attn_add" if is_enc_dec else "decoder_add"
        block_out_name = "block_output" if is_enc_dec else "decoder_output"

        if b0 and b0.attn_residual:
            lines.append(f"[{prefix}_{attn_add_name}]")
            lines.append("Type = addition")
            lines.append(f"input_layers = {input_name},{prefix}_attention_out")
            lines.append("")
            last_residual = f"{prefix}_{attn_add_name}"
        else:
            last_residual = f"{prefix}_attention_out"

        # Cross-attention (decoder blocks in encoder-decoder models)
        if b0 and b0.cross_attention and encoder_output:
            # Cross-attention norm
            if b0.cross_attn_norm:
                lines.append(f"[{prefix}_cross_attn_norm]")
                lines.append(f"Type = {norm_type}")
                lines.append(f"input_layers = {last_residual}")
                lines.append(f"epsilon = {s.norm_eps}")
                if norm_type == "rms_norm":
                    lines.append("packed = false")
                lines.append("")
                cross_q = f"{prefix}_cross_attn_norm"
            else:
                cross_q = last_residual

            # Cross-attention Q/K/V (Q from decoder, K/V from encoder)
            q_unit = s.head_dim * s.num_heads
            kv_unit = s.head_dim * s.num_kv_heads
            lines.append(f"[{prefix}_cross_wq]")
            lines.append("Type = fully_connected")
            lines.append(f"input_layers = {cross_q}")
            lines.append(f"unit = {q_unit}")
            lines.append("disable_bias = true")
            lines.append("")

            lines.append(f"[{prefix}_cross_wk]")
            lines.append("Type = fully_connected")
            lines.append(f"input_layers = {encoder_output}")
            lines.append(f"unit = {kv_unit}")
            lines.append("disable_bias = true")
            lines.append("")

            lines.append(f"[{prefix}_cross_wv]")
            lines.append("Type = fully_connected")
            lines.append(f"input_layers = {encoder_output}")
            lines.append(f"unit = {kv_unit}")
            lines.append("disable_bias = true")
            lines.append("")

            lines.append(f"[{prefix}_cross_attention]")
            lines.append("Type = mha_core")
            lines.append(f"input_layers = {prefix}_cross_wq,"
                         f"{prefix}_cross_wk,{prefix}_cross_wv")
            lines.append(f"num_heads = {s.num_heads}")
            lines.append(f"num_heads_kv = {s.num_kv_heads}")
            lines.append("")

            lines.append(f"[{prefix}_cross_attention_out]")
            lines.append("Type = fully_connected")
            lines.append(f"input_layers = {prefix}_cross_attention")
            lines.append(f"unit = {s.hidden_size}")
            lines.append("disable_bias = true")
            lines.append("")

            # Cross-attention residual
            if b0.cross_attn_residual:
                lines.append(f"[{prefix}_cross_attn_add]")
                lines.append("Type = addition")
                lines.append(f"input_layers = {last_residual},"
                             f"{prefix}_cross_attention_out")
                lines.append("")
                last_residual = f"{prefix}_cross_attn_add"

        ffn_norm_input = last_residual

        # Pre-FFN norm
        if b0 and b0.pre_ffn_norm:
            lines.append(f"[{prefix}_ffn_norm]")
            lines.append(f"Type = {norm_type}")
            lines.append(f"input_layers = {ffn_norm_input}")
            lines.append(f"epsilon = {s.norm_eps}")
            if norm_type == "rms_norm":
                lines.append("packed = false")
            lines.append("")
            ffn_input = f"{prefix}_ffn_norm"
        else:
            ffn_input = ffn_norm_input

        # FFN sub-layers
        if b0 and b0.ffn:
            ffn = b0.ffn
            if ffn.ffn_type == "swiglu":
                lines.append(f"[{prefix}_ffn_up]")
                lines.append("Type = fully_connected")
                lines.append(f"input_layers = {ffn_input}")
                lines.append(f"unit = {s.intermediate_size}")
                lines.append("disable_bias = true")
                lines.append("")

                lines.append(f"[{prefix}_ffn_gate]")
                lines.append("Type = fully_connected")
                lines.append(f"input_layers = {ffn_input}")
                lines.append(f"unit = {s.intermediate_size}")
                lines.append("disable_bias = true")
                lines.append("")

                lines.append(f"[{prefix}_ffn_swiglu]")
                lines.append("Type = swiglu")
                lines.append(f"input_layers = {prefix}_ffn_up,"
                             f"{prefix}_ffn_gate")
                lines.append("")

                lines.append(f"[{prefix}_ffn_down]")
                lines.append("Type = fully_connected")
                lines.append(f"input_layers = {prefix}_ffn_swiglu")
                lines.append(f"unit = {s.hidden_size}")
                lines.append("disable_bias = true")
                lines.append("")
            else:
                act = "gelu" if ffn.ffn_type == "gelu_ffn" else "relu"
                lines.append(f"[{prefix}_ffn_fc1]")
                lines.append("Type = fully_connected")
                lines.append(f"input_layers = {ffn_input}")
                lines.append(f"unit = {s.intermediate_size}")
                lines.append("")

                lines.append(f"[{prefix}_ffn_act]")
                lines.append("Type = activation")
                lines.append(f"input_layers = {prefix}_ffn_fc1")
                lines.append(f"Activation = {act}")
                lines.append("")

                lines.append(f"[{prefix}_ffn_down]")
                lines.append("Type = fully_connected")
                lines.append(f"input_layers = {prefix}_ffn_act")
                lines.append(f"unit = {s.hidden_size}")
                lines.append("")

        # FFN residual
        if b0 and b0.ffn_residual:
            lines.append(f"[{prefix}_{block_out_name}]")
            lines.append("Type = addition")
            lines.append(f"input_layers = {last_residual},{prefix}_ffn_down")
            lines.append("")

        return lines

    def _norm_type(self, s):
        """Determine norm type string for INI."""
        if s.model_type in ("bert", "roberta", "distilbert", "albert"):
            return "layer_normalization"
        return "rms_norm"

    # =========================================================================
    # Flat Mode (verbatim layer list)
    # =========================================================================

    def _emit_flat(self):
        """Emit every layer from the flat list as-is."""
        sections = []
        sections.append("# Auto-generated NNTrainer configuration (flat mode)")
        sections.append("")

        sections.append("[Model]")
        sections.append("Type = NeuralNetwork")
        sections.append(f"batch_size = {self.batch_size}")
        sections.append("")

        for layer in self.layers:
            sections.append(f"[{layer.name}]")
            sections.append(f"Type = {layer.layer_type}")
            if layer.input_layers:
                sections.append(
                    f"input_layers = {','.join(layer.input_layers)}")
            for k, v in layer.properties.items():
                if isinstance(v, bool):
                    sections.append(f"{k} = {'true' if v else 'false'}")
                elif isinstance(v, (list, tuple)):
                    sections.append(
                        f"{k} = {','.join(str(x) for x in v)}")
                else:
                    sections.append(f"{k} = {v}")
            sections.append("")

        return "\n".join(sections)


# =============================================================================
# Convenience function
# =============================================================================

def emit_ini(layers, structure, batch_size=1, mode="structured"):
    """Generate NNTrainer INI configuration.

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        structure: ModelStructure from pattern detection.
        batch_size: Batch size for the model.
        mode: "structured" or "flat".

    Returns:
        str: Complete INI file content.
    """
    emitter = IniEmitter(layers, structure, batch_size)
    return emitter.emit(mode=mode)

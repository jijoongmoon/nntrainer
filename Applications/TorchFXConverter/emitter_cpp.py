"""
C++ code emitter for NNTrainer TorchFX converter.

Generates class-based C++ header (.h) and source (.cpp) files that construct
NNTrainer models, matching the style used in Applications/CausalLM/:
  - causal_lm.h / causal_lm.cpp (base class)
  - nntr_qwen3_causallm.h / nntr_qwen3_causallm.cpp (derived class)

The emitter produces:
  - Header: class declaration with virtual methods (constructModel,
            createAttention, createMlp, createTransformerBlock)
  - Source: method implementations using createLayer() / addLayer() API

Phase 4.1 of the TorchFX converter pipeline (DESIGN.md).
"""

from pattern_detector import ModelStructure, TransformerBlockPattern
from nntrainer_layers import NNTrainerLayerDef


# =============================================================================
# Helpers
# =============================================================================

def _q(s):
    """Quote a string as a C++ string literal."""
    return f'"{s}"'


def _cpp_layer(layer_type, props, indent=2):
    """Generate a createLayer() call as a list of C++ lines."""
    pad = "  " * indent
    lines = []
    lines.append(pad + 'layers.push_back(createLayer("' + layer_type + '", {')
    for i, p in enumerate(props):
        comma = "," if i < len(props) - 1 else ""
        lines.append(pad + "  " + p + comma)
    lines.append(pad + "}));")
    return lines


def _with_key(key, val):
    """Generate withKey() call as a C++ expression string."""
    return f'withKey("{key}", {val})'


# =============================================================================
# Model name helpers
# =============================================================================

def _class_name(model_type, arch_type):
    """Generate C++ class name from model type."""
    name_map = {
        "qwen3": "Qwen3CausalLM",
        "qwen2": "Qwen2CausalLM",
        "llama": "LlamaCausalLM",
        "mistral": "MistralCausalLM",
        "gemma": "GemmaCausalLM",
        "gemma2": "Gemma2CausalLM",
        "gemma3_text": "Gemma3CausalLM",
        "phi": "PhiCausalLM",
        "bert": "BertModel",
        "roberta": "RobertaModel",
        "t5": "T5Model",
        "mt5": "MT5Model",
    }
    # For embedding models, override the class name suffix
    if arch_type == "embedding":
        embed_names = {
            "gemma": "GemmaEmbeddingModel",
            "gemma2": "Gemma2EmbeddingModel",
            "gemma3_text": "Gemma3EmbeddingModel",
            "llama": "LlamaEmbeddingModel",
            "qwen3": "Qwen3EmbeddingModel",
            "qwen2": "Qwen2EmbeddingModel",
            "bert": "BertEmbeddingModel",
        }
        if model_type in embed_names:
            return embed_names[model_type]
        return model_type.capitalize() + "EmbeddingModel"

    if model_type in name_map:
        return name_map[model_type]
    # Fallback: capitalize model_type + arch suffix
    suffix = {"decoder_only": "CausalLM", "encoder_only": "Model",
              "encoder_decoder": "Model"}
    return model_type.capitalize() + suffix.get(arch_type, "Model")


def _file_basename(class_name):
    """Generate file basename from C++ class name (snake_case, no extension).

    e.g. "Qwen3CausalLM" -> "qwen3_causallm"
         "GemmaEmbeddingModel" -> "gemma_embedding_model"
         "MT5Model" -> "mt5_model"
    """
    import re
    # Insert underscore before uppercase letters that follow a lowercase/digit,
    # but keep consecutive uppercase together (e.g. "LM" stays as "lm")
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', class_name)
    # Insert underscore between consecutive uppercase and a following lowercase
    # e.g. "MT5Model" -> "MT5_Model"
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    return s.lower()


def _header_guard(class_name):
    """Generate header guard macro from class name."""
    basename = _file_basename(class_name).upper()
    return f"__{basename.upper()}_H__"


def _sanitize_model_name(model_name):
    """Sanitize a model name (e.g. HF model ID) into a file-safe snake_case basename.

    e.g. "KaLM-embedding-v2.5" -> "kalm_embedding_v2_5"
         "Qwen/Qwen3-0.6B"     -> "qwen3_0_6b"
    """
    import re
    # Take last component if it contains a slash (e.g. "org/model" -> "model")
    name = model_name.rsplit("/", 1)[-1]
    # Replace hyphens, dots, spaces with underscores
    name = re.sub(r'[-.\s]+', '_', name)
    # Collapse multiple underscores and strip leading/trailing
    name = re.sub(r'_+', '_', name).strip('_')
    return name.lower()


def get_output_filenames(model_type, arch_type, model_name=None):
    """Get output filenames for a given model.

    Args:
        model_type: HF config model_type (e.g. "qwen2").
        arch_type: Architecture type (e.g. "embedding", "decoder_only").
        model_name: Optional model name/ID to use for file naming instead of
                    model_type.  e.g. "KaLM-embedding-v2.5".

    Returns:
        dict with keys: "header", "source", "ini", "json"
    """
    if model_name:
        base = _sanitize_model_name(model_name)
    else:
        cname = _class_name(model_type, arch_type)
        base = _file_basename(cname)
    return {
        "header": f"{base}.h",
        "source": f"{base}.cpp",
        "ini": f"{base}.ini",
        "json": f"{base}.json",
    }


# =============================================================================
# C++ Emitter
# =============================================================================

class CppEmitter:
    """Generates class-based C++ header and source code from converter output."""

    def __init__(self, layers, structure, model_name=None):
        self.layers = layers
        self.structure = structure
        self._by_name = {l.name: l for l in layers}
        self._model_name = model_name

    def emit(self):
        """Generate combined C++ code (header + source).

        Returns a single string with header section followed by source section,
        separated by a clear delimiter comment.
        """
        return self.emit_header() + "\n" + self.emit_source()

    def _file_base(self):
        """Return the file basename, using model_name if provided."""
        if self._model_name:
            return _sanitize_model_name(self._model_name)
        cname = _class_name(self.structure.model_type, self.structure.arch_type)
        return _file_basename(cname)

    def emit_header(self):
        """Generate C++ header file (.h) content."""
        s = self.structure
        cname = _class_name(s.model_type, s.arch_type)
        base = self._file_base()
        guard = f"__{base.upper()}_H__"

        L = []
        L.append(f"// Auto-generated by TorchFX-to-NNTrainer converter")
        L.append(f"// Model: {s.model_type} ({s.arch_type})")
        L.append(f"")
        L.append(f"#ifndef {guard}")
        L.append(f"#define {guard}")
        L.append(f"")
        L.append(f"#include <layer.h>")
        L.append(f"#include <model.h>")
        L.append(f"")

        # Conditional includes for derived classes with custom layers
        has_qk_norm = (s.blocks and s.blocks[0].attention
                       and s.blocks[0].attention.has_qk_norm)
        if has_qk_norm:
            L.append(f"#include <reshaped_rms_norm.h>")
            L.append(f"")

        L.append(f"using LayerHandle = std::shared_ptr<ml::train::Layer>;")
        L.append(f"using ModelHandle = std::unique_ptr<ml::train::Model>;")
        L.append(f"")

        # Class declaration
        L.append(f"/**")
        L.append(f" * @brief {cname} Class")
        L.append(f" * @note Auto-generated from HuggingFace {s.model_type} model")
        L.append(f" */")
        L.append(f"class {cname} {{")
        L.append(f"")
        L.append(f"public:")
        L.append(f"  {cname}();")
        L.append(f"  virtual ~{cname}() = default;")
        L.append(f"")
        L.append(f"  /**")
        L.append(f"   * @brief Construct the model graph")
        L.append(f"   */")
        L.append(f"  virtual void constructModel();")
        L.append(f"")
        L.append(f"protected:")

        # createTransformerBlock(s) - encoder-decoder gets both
        if s.arch_type == "encoder_decoder":
            L.append(f"  /**")
            L.append(f"   * @brief Create a Transformer Encoder Block")
            L.append(f"   */")
            L.append(f"  virtual std::vector<LayerHandle>")
            L.append(f"  createEncoderBlock("
                     f"const int layer_id, std::string input_name);")
            L.append(f"")
            L.append(f"  /**")
            L.append(f"   * @brief Create a Transformer Decoder Block")
            L.append(f"   */")
            L.append(f"  virtual std::vector<LayerHandle>")
            L.append(f"  createDecoderBlock("
                     f"const int layer_id, std::string input_name,")
            L.append(f"                     "
                     f"std::string encoder_output);")
            L.append(f"")
        else:
            block_type = ("DecoderBlock" if s.arch_type == "decoder_only"
                          else "Block")
            L.append(f"  /**")
            L.append(f"   * @brief Create a Transformer {block_type}")
            L.append(f"   */")
            L.append(f"  virtual std::vector<LayerHandle>")
            L.append(f"  createTransformer{block_type}("
                     f"const int layer_id, std::string input_name);")
            L.append(f"")

        # createAttention
        L.append(f"  /**")
        L.append(f"   * @brief Create Attention layers")
        L.append(f"   */")
        L.append(f"  virtual std::vector<LayerHandle>")
        L.append(f"  createAttention(const int layer_id, int seq_len, "
                 f"int n_heads, int head_dim,")
        L.append(f"                  std::string query_name, "
                 f"std::string key_name,")
        L.append(f"                  std::string value_name);")
        L.append(f"")

        # createMlp
        L.append(f"  /**")
        L.append(f"   * @brief Create Feed Forward layers")
        L.append(f"   */")
        L.append(f"  virtual std::vector<LayerHandle>")
        L.append(f"  createMlp(const int layer_id, int dim, int hidden_dim,")
        L.append(f"            std::string input_name);")
        L.append(f"")

        # registerCustomLayers
        L.append(f"  /**")
        L.append(f"   * @brief Register custom layers")
        L.append(f"   */")
        L.append(f"  virtual void registerCustomLayers();")
        L.append(f"")

        # Member variables
        L.append(f"  ModelHandle model;")
        L.append(f"")
        L.append(f"  // Model constants")
        L.append(f"  unsigned int NUM_VOCAB = {s.vocab_size};")
        L.append(f"  int DIM = {s.hidden_size};")
        if s.arch_type == "encoder_decoder":
            L.append(f"  int NUM_ENCODER_LAYERS = {s.num_encoder_layers};")
            L.append(f"  int NUM_DECODER_LAYERS = {s.num_decoder_layers};")
        L.append(f"  int NUM_LAYERS = {s.num_layers};")
        L.append(f"  int NUM_HEADS = {s.num_heads};")
        L.append(f"  int NUM_KV_HEADS = {s.num_kv_heads};")
        L.append(f"  int HEAD_DIM = {s.head_dim};")
        L.append(f"  int INTERMEDIATE_SIZE = {s.intermediate_size};")
        L.append(f"  float NORM_EPS = {s.norm_eps or 1e-6}f;")
        if s.rope_theta:
            L.append(f"  unsigned int ROPE_THETA = {int(s.rope_theta)};")
        L.append(f"  bool TIE_WORD_EMBEDDINGS = "
                 f"{'true' if s.tie_word_embeddings else 'false'};")

        # GQA_SIZE
        if s.num_kv_heads and s.num_heads:
            gqa = s.num_heads // s.num_kv_heads
            L.append(f"  int GQA_SIZE = {gqa};")

        # Sliding window
        if s.blocks and s.blocks[0].attention:
            L.append(f"  unsigned int SLIDING_WINDOW = UINT_MAX;")

        # Runtime parameters (set externally)
        L.append(f"  unsigned int INIT_SEQ_LEN = 0;")
        L.append(f"  unsigned int NUM_TO_GENERATE = 0;")
        if s.blocks and s.blocks[0].attention and s.blocks[0].attention.has_rope:
            L.append(f"  unsigned int MAX_POSITION_EMBEDDINGS = "
                     f"{s.max_position_embeddings or 2048};")

        L.append(f"}};")
        L.append(f"")
        L.append(f"#endif // {guard}")
        L.append(f"")
        return "\n".join(L)

    def emit_source(self):
        """Generate C++ source file (.cpp) content."""
        s = self.structure
        cname = _class_name(s.model_type, s.arch_type)
        block_type = "DecoderBlock" if s.arch_type == "decoder_only" else "Block"

        header_file = self._file_base() + ".h"

        L = []
        L.append(f"// Auto-generated by TorchFX-to-NNTrainer converter")
        L.append(f"// Model: {s.model_type} ({s.arch_type})")
        L.append(f"")
        L.append(f'#include "{header_file}"')
        L.append(f"#include <model.h>")
        L.append(f"")
        L.append(f"using ml::train::createLayer;")
        L.append(f"")

        # Helper: withKey (same pattern as llm_util.hpp)
        L.append("template <typename T>")
        L.append('static std::string withKey(const std::string &key, T val) {')
        L.append('  return key + "=" + std::to_string(val);')
        L.append("}")
        L.append("")
        L.append("template <>")
        L.append('std::string withKey(const std::string &key, '
                 'std::string val) {')
        L.append('  return key + "=" + val;')
        L.append("}")
        L.append("")
        L.append("template <>")
        L.append('std::string withKey(const std::string &key, '
                 'const char *val) {')
        L.append('  return key + "=" + std::string(val);')
        L.append("}")
        L.append("")

        # constructModel
        if s.arch_type == "encoder_decoder":
            block_type = None  # handled separately
        else:
            block_type = ("DecoderBlock" if s.arch_type == "decoder_only"
                          else "Block")
        L.append(self._emit_construct_model(cname, block_type))

        # createTransformerBlock / encoder+decoder blocks
        if s.arch_type == "encoder_decoder":
            enc_blocks = s.encoder_blocks
            dec_blocks = s.decoder_blocks
            enc_b0 = enc_blocks[0] if enc_blocks else None
            dec_b0 = dec_blocks[0] if dec_blocks else None

            if enc_b0:
                L.append(self._emit_block_method(
                    cname, "EncoderBlock", enc_b0, is_encoder=True))
            if dec_b0:
                L.append(self._emit_block_method(
                    cname, "DecoderBlock", dec_b0, is_encoder=False))

            # Attention (use encoder block, decoder overrides with cross-attn)
            representative = enc_b0 or dec_b0
            if representative and representative.attention:
                L.append(self._emit_attention_method(cname, representative))

            # FFN
            if representative and representative.ffn:
                L.append(self._emit_ffn_method(cname, representative))
        elif s.blocks:
            b0 = s.blocks[0]
            L.append(self._emit_block_method(cname, block_type, b0))

            if b0.attention:
                L.append(self._emit_attention_method(cname, b0))
            if b0.ffn:
                L.append(self._emit_ffn_method(cname, b0))

        # registerCustomLayers
        L.append(self._emit_register_custom_layers(cname))

        return "\n".join(L)

    # =========================================================================
    # constructModel()
    # =========================================================================

    def _emit_construct_model(self, cname, block_type):
        s = self.structure
        L = []

        L.append(f"void {cname}::constructModel() {{")
        L.append(f"")
        L.append(f"  std::vector<LayerHandle> layers;")
        L.append(f"")
        L.append(f"  // Create model")
        L.append(f"  model = ml::train::createModel("
                 f"ml::train::ModelType::NEURAL_NET);")
        L.append(f"")

        # Input layer
        L.append(f"  // Input layer")
        L.extend(_cpp_layer("input", [
            'withKey("name", "input0")',
            'withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))',
        ]))
        L.append(f"")

        # Embedding
        if s.embedding:
            L.append(f"  // Embedding layer")
            if s.tie_word_embeddings:
                L.append(f'  const std::string embedding_type = '
                         f'TIE_WORD_EMBEDDINGS ? '
                         f'"tie_word_embeddings" : "embedding_layer";')
            else:
                L.append(f'  const std::string embedding_type = '
                         f'"embedding_layer";')
            L.extend(_cpp_layer("\" + embedding_type + \"", [
                'withKey("name", "embedding0")',
                'withKey("in_dim", NUM_VOCAB)',
                'withKey("out_dim", DIM)',
            ]))
            L.append(f"")

        # Transformer blocks loop
        first_in = '"embedding0"' if s.embedding else '"input0"'

        if s.arch_type == "encoder_decoder":
            # Encoder blocks
            L.append(f"  // Encoder blocks")
            L.append(f"  for (int i = 0; i < NUM_ENCODER_LAYERS; ++i) {{")
            L.append(f'    std::string enc_in = (i == 0) ? {first_in}')
            L.append(f'      : "enc_layer" + std::to_string(i - 1) '
                     f'+ "_block_output";')
            L.append(f"    auto block = createEncoderBlock(i, enc_in);")
            L.append(f"    layers.insert(layers.end(), "
                     f"block.begin(), block.end());")
            L.append(f"  }}")
            L.append(f"")

            # Encoder final norm
            norm_type = self._get_norm_type()
            L.append(f"  // Encoder final normalization")
            enc_norm_props = [
                'withKey("name", "encoder_output_norm")',
                'withKey("input_layers", "enc_layer" + '
                'std::to_string(NUM_ENCODER_LAYERS - 1) + "_block_output")',
                'withKey("epsilon", NORM_EPS)',
            ]
            if norm_type == "rms_norm":
                enc_norm_props.append('withKey("packed", "false")')
            L.extend(_cpp_layer(norm_type, enc_norm_props))
            L.append(f"")

            # Decoder blocks
            L.append(f"  // Decoder blocks")
            L.append(f"  for (int i = 0; i < NUM_DECODER_LAYERS; ++i) {{")
            L.append(f'    std::string dec_in = (i == 0) ? {first_in}')
            L.append(f'      : "dec_layer" + std::to_string(i - 1) '
                     f'+ "_block_output";')
            L.append(f'    auto block = createDecoderBlock(i, dec_in, '
                     f'"encoder_output_norm");')
            L.append(f"    layers.insert(layers.end(), "
                     f"block.begin(), block.end());")
            L.append(f"  }}")
            L.append(f"")

            # Decoder final norm
            if s.final_norm:
                L.append(f"  // Decoder final normalization")
                dec_norm_props = [
                    'withKey("name", "decoder_output_norm")',
                    'withKey("input_layers", "dec_layer" + '
                    'std::to_string(NUM_DECODER_LAYERS - 1) '
                    '+ "_block_output")',
                    'withKey("epsilon", NORM_EPS)',
                ]
                if norm_type == "rms_norm":
                    dec_norm_props.append('withKey("packed", "false")')
                L.extend(_cpp_layer(norm_type, dec_norm_props))
                L.append(f"")
        else:
            L.append(f"  // Transformer blocks")
            L.append(f"  for (int i = 0; i < NUM_LAYERS; ++i) {{")
            L.append(f'    std::string input_name = (i == 0) ? {first_in}')
            L.append(f'      : "layer" + std::to_string(i - 1) '
                     f'+ "_decoder_output";')
            L.append(f"    auto block = createTransformer{block_type}"
                     f"(i, input_name);")
            L.append(f"    layers.insert(layers.end(), "
                     f"block.begin(), block.end());")
            L.append(f"  }}")
            L.append(f"")

            # Final norm
            if s.final_norm:
                norm_type = self._get_norm_type()
                L.append(f"  // Final normalization")
                norm_props = [
                    'withKey("name", "output_norm")',
                    'withKey("input_layers", "layer" + '
                    'std::to_string(NUM_LAYERS - 1)'
                    ' + "_decoder_output")',
                    'withKey("epsilon", NORM_EPS)',
                ]
                if norm_type == "rms_norm":
                    norm_props.append('withKey("packed", "false")')
                L.extend(_cpp_layer(norm_type, norm_props))
                L.append(f"")

        # LM head
        if s.lm_head:
            L.append(f"  // LM head")
            if s.tie_word_embeddings:
                L.append(f'  const std::string lmhead_type = '
                         f'TIE_WORD_EMBEDDINGS ? '
                         f'"tie_word_embeddings" : "fully_connected";')
            else:
                L.append(f'  const std::string lmhead_type = '
                         f'"fully_connected";')
            lm_input = ("decoder_output_norm" if s.arch_type == "encoder_decoder"
                        else "output_norm")
            lm_props = [
                'withKey("name", "output_of_causallm")',
                'withKey("unit", NUM_VOCAB)',
                'withKey("disable_bias", "true")',
                f'withKey("input_layers", "{lm_input}")',
            ]
            if s.tie_word_embeddings:
                lm_props.append('withKey("shared_from", "embedding0")')
            L.extend(_cpp_layer("\" + lmhead_type + \"", lm_props))
            L.append(f"")

        # Add all layers
        L.append(f"  // Add layers to model")
        L.append(f"  for (auto &layer : layers) {{")
        L.append(f"    model->addLayer(layer);")
        L.append(f"  }}")
        L.append(f"}}")
        L.append(f"")
        return "\n".join(L)

    # =========================================================================
    # createTransformerBlock()
    # =========================================================================

    def _get_norm_type(self):
        """Return the norm layer type for this model."""
        return ("rms_norm" if self.structure.model_type not in
                ("bert", "roberta") else "layer_normalization")

    def _emit_block_method(self, cname, block_type, block,
                           is_encoder=None):
        s = self.structure
        norm_type = self._get_norm_type()
        L = []

        # Determine prefix for layer names: enc_layer / dec_layer / layer
        if is_encoder is True:
            prefix_expr = '"enc_layer" + std::to_string(layer_id)'
        elif is_encoder is False:
            prefix_expr = '"dec_layer" + std::to_string(layer_id)'
        else:
            prefix_expr = '"layer" + std::to_string(layer_id)'

        # Method signature
        L.append(f"std::vector<LayerHandle>")
        if block_type in ("EncoderBlock",):
            L.append(f"{cname}::create{block_type}("
                     f"const int layer_id,")
            L.append(f"  std::string input_name) {{")
        elif block_type == "DecoderBlock" and is_encoder is False:
            L.append(f"{cname}::create{block_type}("
                     f"const int layer_id,")
            L.append(f"  std::string input_name, "
                     f"std::string encoder_output) {{")
        else:
            L.append(f"{cname}::createTransformer{block_type}("
                     f"const int layer_id,")
            L.append(f"  std::string input_name) {{")

        L.append(f"")
        L.append(f"  std::vector<LayerHandle> layers;")
        L.append(f"  auto prefix = {prefix_expr};")
        L.append(f"")

        # Pre-attention norm
        if block.pre_attn_norm:
            L.append(f"  // Pre-attention normalization")
            norm_props = [
                'withKey("name", prefix + "_attention_norm")',
                'withKey("input_layers", input_name)',
                'withKey("epsilon", NORM_EPS)',
            ]
            if norm_type == "rms_norm":
                norm_props.append('withKey("packed", "false")')
            L.extend(_cpp_layer(norm_type, norm_props))
            L.append(f"")

        # Self-attention
        if block.attention:
            attn_in = ('prefix + "_attention_norm"'
                       if block.pre_attn_norm else "input_name")
            L.append(f"  auto att_layer =")
            L.append(f"    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, "
                     f"HEAD_DIM,")
            L.append(f"                    {attn_in},")
            L.append(f"                    {attn_in},")
            L.append(f"                    {attn_in});")
            L.append(f"")
            L.append(f"  layers.insert(layers.end(), att_layer.begin(), "
                     f"att_layer.end());")
            L.append(f"")

        # Self-attention residual
        if block.attn_residual:
            L.append(f"  // Self-attention residual connection")
            L.extend(_cpp_layer("addition", [
                'withKey("name", prefix + "_self_attn_add")',
                'withKey("input_layers", input_name + "," + '
                'prefix + "_attention_out")',
            ]))
            L.append(f"")
            last_residual = 'prefix + "_self_attn_add"'
        else:
            last_residual = 'prefix + "_attention_out"'

        # Cross-attention (decoder blocks in encoder-decoder models)
        if block.cross_attention and is_encoder is False:
            # Cross-attention norm
            if block.cross_attn_norm:
                L.append(f"  // Cross-attention normalization")
                cross_norm_props = [
                    'withKey("name", prefix + "_cross_attn_norm")',
                    f'withKey("input_layers", {last_residual})',
                    'withKey("epsilon", NORM_EPS)',
                ]
                if norm_type == "rms_norm":
                    cross_norm_props.append('withKey("packed", "false")')
                L.extend(_cpp_layer(norm_type, cross_norm_props))
                L.append(f"")
                cross_q = 'prefix + "_cross_attn_norm"'
            else:
                cross_q = last_residual

            # Cross-attention layers (Q from decoder, K/V from encoder)
            L.append(f"  auto cross_att =")
            L.append(f"    createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, "
                     f"HEAD_DIM,")
            L.append(f"                    {cross_q},")
            L.append(f"                    encoder_output,")
            L.append(f"                    encoder_output);")
            L.append(f"")
            L.append(f"  layers.insert(layers.end(), cross_att.begin(), "
                     f"cross_att.end());")
            L.append(f"")

            # Cross-attention residual
            if block.cross_attn_residual:
                L.append(f"  // Cross-attention residual")
                L.extend(_cpp_layer("addition", [
                    'withKey("name", prefix + "_cross_attn_add")',
                    f'withKey("input_layers", {last_residual} + "," + '
                    f'prefix + "_attention_out")',
                ]))
                L.append(f"")
                last_residual = 'prefix + "_cross_attn_add"'

        # Rename for decoder_only compatibility
        if is_encoder is None:
            # Single-stack model: use _decoder_add / _decoder_output naming
            attn_add_name = '"_decoder_add"'
            block_out_name = '"_decoder_output"'
        else:
            attn_add_name = '"_self_attn_add"'
            block_out_name = '"_block_output"'

        # Pre-FFN norm
        if block.pre_ffn_norm:
            L.append(f"  // Pre-FFN normalization")
            norm_props = [
                'withKey("name", prefix + "_ffn_norm")',
                f'withKey("input_layers", {last_residual})',
                'withKey("epsilon", NORM_EPS)',
            ]
            if norm_type == "rms_norm":
                norm_props.append('withKey("packed", "false")')
            L.extend(_cpp_layer(norm_type, norm_props))
            L.append(f"")
            ffn_in = 'prefix + "_ffn_norm"'
        else:
            ffn_in = last_residual

        # FFN
        if block.ffn:
            L.append(f"  auto ffn_layer = createMlp(layer_id, DIM, "
                     f"INTERMEDIATE_SIZE,")
            L.append(f"                             {ffn_in});")
            L.append(f"  layers.insert(layers.end(), ffn_layer.begin(), "
                     f"ffn_layer.end());")
            L.append(f"")

        # FFN residual
        if block.ffn_residual:
            L.append(f"  // FFN residual connection")
            L.extend(_cpp_layer("addition", [
                f'withKey("name", prefix + {block_out_name})',
                f'withKey("input_layers", {last_residual} + "," + '
                f'prefix + "_ffn_down")',
            ]))
            L.append(f"")

        L.append(f"  return layers;")
        L.append(f"}}")
        L.append(f"")
        return "\n".join(L)

    # =========================================================================
    # createAttention()
    # =========================================================================

    def _emit_attention_method(self, cname, block):
        attn = block.attention
        has_qk_norm = attn.has_qk_norm
        has_rope = attn.has_rope
        L = []

        L.append(f"std::vector<LayerHandle>")
        L.append(f"{cname}::createAttention(const int layer_id, int seq_len, "
                 f"int n_heads,")
        L.append(f"                         int head_dim, "
                 f"std::string query_name,")
        L.append(f"                         std::string key_name, "
                 f"std::string value_name) {{")
        L.append(f"")
        L.append(f"  std::vector<LayerHandle> layers;")
        L.append(f"")
        L.append(f'  auto Q = "layer" + std::to_string(layer_id) + "_wq";')
        L.append(f'  auto K = "layer" + std::to_string(layer_id) + "_wk";')
        L.append(f'  auto V = "layer" + std::to_string(layer_id) + "_wv";')
        if has_qk_norm:
            L.append(f'  auto Q_norm = "layer" + std::to_string(layer_id) '
                     f'+ "_q_norm";')
            L.append(f'  auto K_norm = "layer" + std::to_string(layer_id) '
                     f'+ "_k_norm";')
        L.append(f'  auto A = "layer" + std::to_string(layer_id) '
                 f'+ "_attention";')
        L.append(f'  auto O = "layer" + std::to_string(layer_id) '
                 f'+ "_attention_out";')
        L.append(f"")

        # V layer
        L.append(f"  // V layer")
        L.extend(_cpp_layer("fully_connected", [
            'withKey("name", V)',
            'withKey("unit", head_dim * n_heads / GQA_SIZE)',
            'withKey("disable_bias", "true")',
            'withKey("input_layers", value_name)',
        ]))

        # K layer
        L.append(f"")
        L.append(f"  // K layer")
        L.extend(_cpp_layer("fully_connected", [
            'withKey("name", K)',
            'withKey("unit", head_dim * n_heads / GQA_SIZE)',
            'withKey("disable_bias", "true")',
            'withKey("input_layers", key_name)',
        ]))

        # Q layer
        L.append(f"")
        L.append(f"  // Q layer")
        L.extend(_cpp_layer("fully_connected", [
            'withKey("name", Q)',
            'withKey("unit", head_dim * n_heads)',
            'withKey("disable_bias", "true")',
            'withKey("input_layers", query_name)',
        ]))

        # Q/K norms (Qwen3-style)
        if has_qk_norm:
            L.append(f"")
            L.append(f"  // K norm (reshaped RMS norm)")
            L.extend(_cpp_layer("reshaped_rms_norm", [
                'withKey("name", K_norm)',
                'withKey("input_layers", K)',
                'withKey("packed", "false")',
                'withKey("epsilon", NORM_EPS)',
                'withKey("feature_size", head_dim)',
            ]))

            L.append(f"")
            L.append(f"  // Q norm (reshaped RMS norm)")
            L.extend(_cpp_layer("reshaped_rms_norm", [
                'withKey("name", Q_norm)',
                'withKey("input_layers", Q)',
                'withKey("packed", "false")',
                'withKey("epsilon", NORM_EPS)',
                'withKey("feature_size", head_dim)',
            ]))

        # MHA core
        L.append(f"")
        L.append(f"  // Attention core layer")
        q_in = "Q_norm" if has_qk_norm else "Q"
        k_in = "K_norm" if has_qk_norm else "K"

        mha_props = [
            'withKey("name", A)',
            'withKey("num_heads", n_heads)',
            'withKey("num_heads_kv", n_heads / GQA_SIZE)',
            'withKey("max_timestep", std::to_string(INIT_SEQ_LEN + '
            'NUM_TO_GENERATE))',
        ]
        # Sliding window
        mha_props.append('withKey("sliding_window", SLIDING_WINDOW)')
        if has_rope:
            mha_props.append('withKey("rope_theta", ROPE_THETA)')
            mha_props.append(
                'withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS)')
        mha_props.append('withKey("max_new_tokens", NUM_TO_GENERATE)')
        mha_props.append(
            f'withKey("input_layers", {{{q_in}, {k_in}, V}})')
        L.extend(_cpp_layer("mha_core", mha_props))

        # O layer
        L.append(f"")
        L.append(f"  // O layer")
        L.extend(_cpp_layer("fully_connected", [
            'withKey("name", O)',
            'withKey("unit", DIM)',
            'withKey("disable_bias", "true")',
            'withKey("input_layers", A)',
        ]))

        L.append(f"")
        L.append(f"  return layers;")
        L.append(f"}}")
        L.append(f"")
        return "\n".join(L)

    # =========================================================================
    # createMlp()
    # =========================================================================

    def _emit_ffn_method(self, cname, block):
        ffn = block.ffn
        L = []

        L.append(f"std::vector<LayerHandle> {cname}::createMlp(")
        L.append(f"  const int layer_id, int dim, int hidden_dim,")
        L.append(f"  std::string input_name) {{")
        L.append(f"")
        L.append(f"  std::vector<LayerHandle> layers;")
        L.append(f"")

        if ffn.ffn_type == "swiglu":
            L.extend(_cpp_layer("fully_connected", [
                'withKey("name", "layer" + std::to_string(layer_id) '
                '+ "_ffn_up")',
                'withKey("unit", hidden_dim)',
                'withKey("disable_bias", "true")',
                'withKey("input_layers", input_name)',
            ]))

            L.append(f"")
            L.extend(_cpp_layer("fully_connected", [
                'withKey("name", "layer" + std::to_string(layer_id) '
                '+ "_ffn_gate")',
                'withKey("unit", hidden_dim)',
                'withKey("disable_bias", "true")',
                'withKey("input_layers", input_name)',
            ]))

            L.append(f"")
            L.extend(_cpp_layer("swiglu", [
                'withKey("name", "layer" + std::to_string(layer_id) '
                '+ "_ffn_swiglu")',
                'withKey("input_layers", "layer" + std::to_string(layer_id) '
                '+ "_ffn_up," + "layer" + std::to_string(layer_id) '
                '+ "_ffn_gate")',
            ]))

            L.append(f"")
            L.extend(_cpp_layer("fully_connected", [
                'withKey("name", "layer" + std::to_string(layer_id) '
                '+ "_ffn_down")',
                'withKey("unit", dim)',
                'withKey("disable_bias", "true")',
                'withKey("input_layers", "layer" + std::to_string(layer_id) '
                '+ "_ffn_swiglu")',
            ]))
        else:
            # GELU or ReLU FFN
            act = "gelu" if ffn.ffn_type == "gelu_ffn" else "relu"

            L.extend(_cpp_layer("fully_connected", [
                'withKey("name", "layer" + std::to_string(layer_id) '
                '+ "_ffn_fc1")',
                'withKey("unit", hidden_dim)',
                'withKey("input_layers", input_name)',
            ]))

            L.append(f"")
            L.extend(_cpp_layer("activation", [
                'withKey("name", "layer" + std::to_string(layer_id) '
                '+ "_ffn_act")',
                f'withKey("activation", "{act}")',
                'withKey("input_layers", "layer" + std::to_string(layer_id) '
                '+ "_ffn_fc1")',
            ]))

            L.append(f"")
            L.extend(_cpp_layer("fully_connected", [
                'withKey("name", "layer" + std::to_string(layer_id) '
                '+ "_ffn_down")',
                'withKey("unit", dim)',
                'withKey("input_layers", "layer" + std::to_string(layer_id) '
                '+ "_ffn_act")',
            ]))

        L.append(f"")
        L.append(f"  return layers;")
        L.append(f"}}")
        L.append(f"")
        return "\n".join(L)

    # =========================================================================
    # registerCustomLayers()
    # =========================================================================

    def _emit_register_custom_layers(self, cname):
        s = self.structure
        has_qk_norm = (s.blocks and s.blocks[0].attention
                       and s.blocks[0].attention.has_qk_norm)
        L = []

        L.append(f"void {cname}::registerCustomLayers() {{")
        L.append(f"  auto &ct_engine = nntrainer::Engine::Global();")
        L.append(f'  auto app_context =')
        L.append(f'    static_cast<nntrainer::AppContext *>('
                 f'ct_engine.getRegisteredContext("cpu"));')
        L.append(f"")
        L.append(f"  try {{")

        # Register standard custom layers
        custom_layers = ["EmbeddingLayer", "RMSNormLayer", "MHACore",
                         "SwiGLULayer", "TieWordEmbeddingLayer"]
        for cl in custom_layers:
            L.append(f"    app_context->registerFactory("
                     f"nntrainer::createLayer<{cl}>);")

        if has_qk_norm:
            L.append(f"    app_context->registerFactory("
                     f"nntrainer::createLayer<ReshapedRMSNormLayer>);")

        L.append(f"  }} catch (std::invalid_argument &e) {{")
        L.append(f'    std::cerr << "failed to register factory, reason: " '
                 f'<< e.what() << std::endl;')
        L.append(f"  }}")
        L.append(f"}}")
        L.append(f"")
        return "\n".join(L)


# =============================================================================
# Convenience functions
# =============================================================================

def emit_cpp(layers, structure, model_name=None):
    """Generate combined C++ code (header + source).

    Args:
        layers: List of NNTrainerLayerDef from converter pipeline.
        structure: ModelStructure from pattern detection.
        model_name: Optional model name for file naming.

    Returns:
        str: Combined header and source code.
    """
    emitter = CppEmitter(layers, structure, model_name=model_name)
    return emitter.emit()


def emit_cpp_header(layers, structure, model_name=None):
    """Generate C++ header file (.h) content."""
    return CppEmitter(layers, structure, model_name=model_name).emit_header()


def emit_cpp_source(layers, structure, model_name=None):
    """Generate C++ source file (.cpp) content."""
    return CppEmitter(layers, structure, model_name=model_name).emit_source()

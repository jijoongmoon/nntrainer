"""
NNTrainer layer type definitions and registry.

This module defines dataclasses representing NNTrainer layer definitions,
and provides the mapping between HuggingFace module types and NNTrainer
layer types.

NNTrainer Layer Types (relevant for HuggingFace model conversion):
  - input                  : Model input placeholder
  - fully_connected        : Dense/Linear layer (nn.Linear)
  - embedding_layer        : Token embedding (nn.Embedding)
  - tie_word_embeddings    : Shared embedding + LM head
  - rms_norm               : RMS normalization (HF *RMSNorm)
  - reshaped_rms_norm      : RMS norm with reshape (Qwen3 Q/K norm)
  - layer_normalization    : Layer normalization (nn.LayerNorm)
  - mha_core               : Multi-head attention core (SDPA + RoPE)
  - swiglu                 : SwiGLU activation (SiLU gate)
  - addition               : Element-wise addition (residual connections)
  - activation             : Activation function (relu, gelu, swish, etc.)
  - dropout                : Dropout (skipped in inference)
  - concat                 : Concatenation
  - reshape                : Reshape tensor
  - permute                : Permute dimensions
  - matmul                 : Matrix multiplication
  - batch_normalization    : Batch normalization
  - conv1d / conv2d        : Convolution layers
  - pooling2d              : Pooling layers
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NNTrainerLayerDef:
    """Definition of a single NNTrainer layer.

    This corresponds to a createLayer() call in C++ with a list of properties.
    """
    layer_type: str                           # e.g. "fully_connected", "rms_norm"
    name: str                                 # Layer name (used for connections)
    properties: dict = field(default_factory=dict)  # Key-value properties
    input_layers: list = field(default_factory=list) # Input layer name(s)

    # Source information (from HF model)
    hf_module_name: str = ""                  # e.g. "model.layers.0.self_attn.q_proj"
    hf_module_type: str = ""                  # e.g. "Linear", "Qwen3RMSNorm"

    # Weight information
    has_weight: bool = False
    has_bias: bool = False
    weight_hf_key: str = ""                   # HF state_dict key for weight
    bias_hf_key: str = ""                     # HF state_dict key for bias
    transpose_weight: bool = False            # Whether to transpose weight [out,in]->[in,out]
    shared_from: str = ""                     # For tied weights (e.g. tie_word_embeddings)

    def to_properties_list(self) -> list:
        """Convert to NNTrainer property string list for createLayer()."""
        props = [f"name={self.name}"]
        if self.input_layers:
            props.append(f"input_layers={','.join(self.input_layers)}")
        for k, v in self.properties.items():
            if isinstance(v, bool):
                props.append(f"{k}={'true' if v else 'false'}")
            elif isinstance(v, (list, tuple)):
                props.append(f"{k}={','.join(str(x) for x in v)}")
            else:
                props.append(f"{k}={v}")
        return props

    def to_cpp_call(self) -> str:
        """Generate C++ createLayer() call string."""
        props = self.to_properties_list()
        props_str = ", ".join(f'"{p}"' for p in props)
        return f'createLayer("{self.layer_type}", {{{props_str}}})'


# =============================================================================
# NNTrainer layer type constants
# =============================================================================

# Core layer types used in CausalLM
LAYER_INPUT = "input"
LAYER_FC = "fully_connected"
LAYER_EMBEDDING = "embedding_layer"
LAYER_TIE_WORD_EMBEDDINGS = "tie_word_embeddings"
LAYER_RMS_NORM = "rms_norm"
LAYER_RESHAPED_RMS_NORM = "reshaped_rms_norm"
LAYER_LAYER_NORM = "layer_normalization"
LAYER_MHA_CORE = "mha_core"
LAYER_SWIGLU = "swiglu"
LAYER_ADDITION = "addition"
LAYER_ACTIVATION = "activation"
LAYER_DROPOUT = "dropout"

# Shape manipulation
LAYER_CONCAT = "concat"
LAYER_RESHAPE = "reshape"
LAYER_PERMUTE = "permute"
LAYER_SPLIT = "split"

# Math operations (element-wise)
LAYER_MATMUL = "matmul"
LAYER_ADD = "add"
LAYER_MULTIPLY = "multiply"
LAYER_SUBTRACT = "subtract"
LAYER_DIVIDE = "divide"
LAYER_POW = "pow"
LAYER_SQRT = "sqrt"
LAYER_NEGATIVE = "negative"

# Trigonometric / math functions
LAYER_SIN = "sin"
LAYER_COS = "cos"
LAYER_TAN = "tan"

# Reduction operations
LAYER_REDUCE_MEAN = "reduce_mean"
LAYER_REDUCE_SUM = "reduce_sum"

# Indexing / selection
LAYER_GATHER = "gather"
LAYER_SLICE = "slice"

# Tensor manipulation
LAYER_FLATTEN = "flatten"
LAYER_TRANSPOSE = "transpose"
LAYER_IDENTITY = "identity"
LAYER_CAST = "cast"

# Convolution & pooling
LAYER_CONV1D = "conv1d"
LAYER_CONV2D = "conv2d"
LAYER_POOLING2D = "pooling2d"
LAYER_BATCH_NORM = "batch_normalization"

# Activation type strings (for LAYER_ACTIVATION)
ACT_RELU = "relu"
ACT_GELU = "gelu"
ACT_SWISH = "swish"  # SiLU = Swish
ACT_SIGMOID = "sigmoid"
ACT_TANH = "tanh"
ACT_SOFTMAX = "softmax"

# Intermediate/internal op types (used during mapping, collapsed by pattern_detector)
# These are not final NNTrainer layer types but help track graph structure
OP_RESHAPE = "reshape_op"
OP_TRANSPOSE = "transpose_op"
OP_PERMUTE = "permute_op"
OP_SDPA = "sdpa"
OP_NOOP = "noop"  # No-op (skipped in final output)

# Unsupported op marker (for ops requiring decomposition into supported primitives)
OP_UNSUPPORTED = "unsupported"

# =============================================================================
# NNTrainer Tensor & LazyTensor method mappings
# =============================================================================
# These define which operations can be expressed as Tensor methods or
# LazyTensor chains in generated C++ code. Used by the decomposer to choose
# the most efficient emission strategy.
#
# NNTrainer Tensor class supports these methods directly:
#   Arithmetic (in-place): add_i, subtract_i, multiply_i, divide_i
#   Element-wise math:     pow, sqrt, abs, neg, erf, inv_sqrt
#   Trigonometric:         sin, cos, tan
#   Reduction:             sum, average, sum_by_batch
#   Matrix:                dot (matmul)
#   Shape:                 transpose
#
# LazyTensor (Tensor::chain()) supports chaining these in-place ops:
#   add_i, subtract_i, multiply_i, divide_i, dot, transpose, sum, average

# Ops that can be chained via LazyTensor (in-place, no shape change)
LAZY_TENSOR_OPS = frozenset({
    LAYER_ADD, LAYER_ADDITION, LAYER_SUBTRACT,
    LAYER_MULTIPLY, LAYER_DIVIDE,
})

# Ops that are available as direct Tensor methods (not in LazyTensor but
# available on Tensor itself - can be used for single-op decomposition)
TENSOR_METHOD_OPS = frozenset({
    LAYER_POW, LAYER_SQRT, LAYER_NEGATIVE,
    LAYER_SIN, LAYER_COS, LAYER_TAN,
    LAYER_REDUCE_MEAN, LAYER_REDUCE_SUM,
    LAYER_MATMUL,  # dot()
})

# Direct Tensor methods for ops that would otherwise need decomposition.
# Maps PyTorch op name -> (NNTrainer Tensor method, is_in_place)
TENSOR_DIRECT_METHODS = {
    "rsqrt": ("inv_sqrt", False),    # Tensor::inv_sqrt(out) or inv_sqrt_i()
    "abs": ("abs", False),           # Tensor::abs(out)
    "neg": ("neg", False),           # Tensor::neg(out)
    "erf": ("erf", False),           # Tensor::erf(out)
    "pow": ("pow", False),           # Tensor::pow(exponent, out)
    "sqrt": ("sqrt", False),         # Tensor::sqrt(out)
    "sin": ("sin", False),           # Tensor::sin(out)
    "cos": ("cos", False),           # Tensor::cos(out)
    "tan": ("tan", False),           # Tensor::tan(out)
}

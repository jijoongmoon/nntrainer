# Torch.FX to NNTrainer Converter - Design Document

## 1. Overview

This converter takes a PyTorch model (primarily HuggingFace models), traces it using
a runtime-based Torch.FX tracer (not symbolic tracing), and converts the resulting
FX graph into an NNTrainer model definition.

The converter works in 3 phases:
1. **Trace**: Run the model with real input using `TorchFunctionMode`-based tracer
2. **Map**: Convert FX graph nodes to NNTrainer layer/op definitions
3. **Emit**: Generate NNTrainer model construction code (or JSON config)

## 2. NNTrainer Capabilities (Verified from Source Code)

### 2.1 Built-in Layers (Registered in `app_context.cpp`)

These layers are registered via `registerFactory()` in
`nntrainer/app_context.cpp:255-429` and are always available:

| Type String | Class | PyTorch Equivalent |
|---|---|---|
| `"input"` | InputLayer | - |
| `"weight"` | WeightLayer | - |
| `"fully_connected"` | FullyConnectedLayer | `nn.Linear` |
| `"batch_normalization"` | BatchNormalizationLayer | `nn.BatchNorm1d/2d/3d` |
| `"layer_normalization"` | LayerNormalizationLayer | `nn.LayerNorm` |
| `"conv1d"` | Conv1DLayer | `nn.Conv1d` |
| `"conv2d"` | Conv2DLayer | `nn.Conv2d` |
| `"conv2dtranspose"` | Conv2DTransposeLayer | `nn.ConvTranspose2d` |
| `"pooling2d"` | Pooling2DLayer | `nn.MaxPool2d` / `nn.AvgPool2d` / `nn.AdaptiveAvgPool2d` |
| `"flatten"` | FlattenLayer | `nn.Flatten` |
| `"reshape"` | ReshapeLayer | `Tensor.reshape` / `Tensor.view` |
| `"permute"` | PermuteLayer | `Tensor.permute` |
| `"activation"` | ActivationLayer | `nn.ReLU`, `nn.GELU`, etc. |
| `"addition"` | AdditionLayer | Residual add (multi-input) |
| `"concat"` | ConcatLayer | `torch.cat` |
| `"split"` | SplitLayer | `torch.split` |
| `"multiout"` | MultiOutLayer | - |
| `"embedding"` | EmbeddingLayer | `nn.Embedding` |
| `"rnn"` | RNNLayer | `nn.RNN` |
| `"rnncell"` | RNNCellLayer | `nn.RNNCell` |
| `"lstm"` | LSTMLayer | `nn.LSTM` |
| `"lstmcell"` | LSTMCellLayer | `nn.LSTMCell` |
| `"zoneout_lstmcell"` | ZoneoutLSTMCellLayer | - |
| `"gru"` | GRULayer | `nn.GRU` |
| `"grucell"` | GRUCellLayer | `nn.GRUCell` |
| `"dropout"` | DropOutLayer | `nn.Dropout` |
| `"attention"` | AttentionLayer | - |
| `"mol_attention"` | MoLAttentionLayer | - |
| `"multi_head_attention"` | MultiHeadAttentionLayer | `nn.MultiheadAttention` |
| `"positional_encoding"` | PositionalEncodingLayer | - |
| `"identity"` | IdentityLayer | `nn.Identity` |
| `"upsample2d"` | Upsample2dLayer | `nn.Upsample` |
| `"channel_shuffle"` | ChannelShuffle | - |
| `"time_dist"` | TimeDistLayer | - |
| `"centroid_knn"` | CentroidKNN | - |
| `"add"` | AddLayer | `torch.add` / `+` |
| `"subtract"` | SubtractLayer | `torch.sub` / `-` |
| `"multiply"` | MultiplyLayer | `torch.mul` / `*` |
| `"divide"` | DivideLayer | `torch.div` / `/` |
| `"pow"` | PowLayer | `torch.pow` |
| `"sqrt"` | SQRTLayer | `torch.sqrt` |
| `"sin"` | SineLayer | `torch.sin` |
| `"cos"` | CosineLayer | `torch.cos` |
| `"tan"` | TangentLayer | `torch.tan` |
| `"negative"` | NegativeLayer | `torch.neg` / `-x` |
| `"matmul"` | MatMulLayer | `torch.matmul` / `@` |
| `"cast"` | CastLayer | `Tensor.to(dtype)` |
| `"gather"` | GatherLayer | `torch.gather` / indexing |
| `"slice"` | SliceLayer | `Tensor[..., a:b]` |
| `"reduce_mean"` | ReduceMeanLayer | `torch.mean` |
| `"reduce_sum"` | ReduceSumLayer | `torch.sum` |
| `"preprocess_flip"` | PreprocessFlipLayer | - |
| `"preprocess_translate"` | PreprocessTranslateLayer | - |
| `"preprocess_l2norm"` | PreprocessL2NormLayer | `F.normalize` |
| `"mse"` | MSELossLayer | `nn.MSELoss` |
| `"cross_sigmoid"` | CrossEntropySigmoidLossLayer | `nn.BCEWithLogitsLoss` |
| `"cross_softmax"` | CrossEntropySoftmaxLossLayer | `nn.CrossEntropyLoss` |
| `"constant_derivative"` | ConstantDerivativeLossLayer | - |

**NOT registered in app_context.cpp** (exist as headers but require separate registration):
- `"depthwiseconv2d"` - DepthwiseConv2DLayer (`depthwise_conv2d_layer.h` exists)
- `"rmsnorm"` - RMSNormLayerCl (only in `cl_layers/rmsnorm_layer_cl.h`, OpenCL backend)
- `"swiglu"` - SwiGLULayerCl (only in `cl_layers/swiglu_cl.h`, OpenCL backend)

### 2.2 Pooling2D Modes (from `common_properties.h`)

| Mode | Description | PyTorch Equivalent |
|---|---|---|
| `max` | Max pooling | `nn.MaxPool2d` |
| `average` | Average pooling | `nn.AvgPool2d` |
| `global_max` | Global max pooling | `nn.AdaptiveMaxPool2d((1,1))` |
| `global_average` | Global average pooling | `nn.AdaptiveAvgPool2d((1,1))` |

### 2.3 Activation Types (from `acti_func.h` + `common_properties.h`)

All verified from the `setActiFunc()` switch statement:

| ActivationType | PyTorch Equivalent |
|---|---|
| `ACT_TANH` | `nn.Tanh` |
| `ACT_SIGMOID` | `nn.Sigmoid` |
| `ACT_RELU` | `nn.ReLU` |
| `ACT_LEAKY_RELU` | `nn.LeakyReLU` (slope=0.01 fixed) |
| `ACT_SWISH` | `nn.SiLU` |
| `ACT_GELU` | `nn.GELU` |
| `ACT_TANH_GELU` | `nn.GELU(approximate='tanh')` |
| `ACT_SIGMOID_GELU` | GELU via sigmoid approximation |
| `ACT_ELU` | `nn.ELU` |
| `ACT_SELU` | `nn.SELU` |
| `ACT_SOFTMAX` | `nn.Softmax` |
| `ACT_SOFTPLUS` | `nn.Softplus` |
| `ACT_MISH` | `nn.Mish` |
| `ACT_NONE` | Identity |

### 2.4 Tensor Operations (from `tensor.h`)

Operations available on the `Tensor` class (verified from source):

**Arithmetic**: `add`, `subtract`, `multiply`, `divide` (element-wise, with scalar and tensor variants, strided variants, in-place `_i` variants)

**Math**: `pow`, `sqrt`, `inv_sqrt`, `erf`, `neg`, `sin`, `cos`, `tan`, `abs`

**Reduction**: `sum` (with axis), `sum_by_batch`, `average` (with axis), `l2norm`, `maxValue`, `minValue`, `max_abs`

**Linear Algebra**: `dot` (GEMM), `dotBatched`, `dot_deriv_wrt_1`, `dot_deriv_wrt_2`

**Shape**: `reshape`, `transpose`, `split` (equal and custom sizes), `concat`, `cat` (static), `getBatchSlice`, `getSharedDataTensor`, `mergeAxis`, `fill`

**Indexing**: `argmax`, `topK`

**Normalization**: `normalization` (L2), `standardization` (z-score)

**Masking**: `dropout_mask`, `filter_mask`, `zoneout_mask`

**Backend-only** (in `cpu_backend.h`, NOT exposed in Tensor public API):
- `clamp` - exists but not in Tensor class
- `softmax`, `swiglu`, `tanh_gelu`, `tanh_gelu_mul` - backend optimized
- `ele_mul`, `ele_add`, `ele_sub`, `ele_div` - raw element-wise
- `sgemm`, `sgemv`, `saxpy`, `sdot` - BLAS operations
- `compute_rotary_embedding_value` - RoPE computation

**NOT available anywhere**:
- `exp`, `log`, `clamp` (not in Tensor public API)
- `where`, `masked_fill`
- `repeat`, `expand`, `broadcast`
- Comparison operations (`gt`, `lt`, `eq`, `ge`, `le`)

### 2.5 CausalLM Application Layers (from `Applications/CausalLM/`)

These are custom layers registered at runtime via `registerCustomLayers()`.
They are NOT in `app_context.cpp` - they must be registered by the application.

**Base Transformer registers** (in `transformer.cpp:registerCustomLayers()`):

| Type String | Class | Purpose |
|---|---|---|
| `"swiglu"` | SwiGLULayer | SwiGLU activation: `silu(gate) * up` |
| `"rms_norm"` | RMSNormLayer | RMS normalization with learnable gamma |
| `"mha_core"` | MHACoreLayer | Multi-head attention with KV-cache, RoPE, GQA, sliding window |
| `"tie_word_embeddings"` | TieWordEmbedding | Shared embedding/LM-head weights |
| `"embedding_layer"` | EmbeddingLayer | Token embedding with scale |

**CausalLM adds** (in `causal_lm.cpp:registerCustomLayers()`):

| Type String | Class | Purpose |
|---|---|---|
| `"lm_head"` | LmHeadLayer | Final projection to vocabulary |

**Qwen3 adds** (in `qwen3_causallm.cpp:registerCustomLayers()`):

| Type String | Class | Purpose |
|---|---|---|
| `"reshaped_rms_norm"` | ReshapedRMSNormLayer | RMSNorm with reshape for Q/K normalization |

**Gemma3 adds** (in `gemma3_causallm.cpp:registerCustomLayers()`):

| Type String | Class | Purpose |
|---|---|---|
| `"reshaped_rms_norm"` | ReshapedRMSNormLayer | Same as Qwen3 |

**MoE models add**:

| Type String | Class | Model |
|---|---|---|
| `"qwen_moe"` | MoELayer | Qwen3MoECausalLM |
| `"moe_slim"` | SlimMoELayer | Qwen3SlimMoECausalLM |
| `"moe_cached_slim"` | CachedSlimMoELayer | Qwen3CachedSlimMoECausalLM |
| `"gpt_oss_moe"` | GptOssMoELayer | GptOssForCausalLM |
| `"gpt_oss_moe_slim_cached"` | GptOssMoESlimCachedLayer | GptOssCachedSlimCausalLM |

**Additional CausalLM layers** (for Embedding models):

| Type String | Class | Purpose |
|---|---|---|
| `"embedding_normalize"` | EmbeddingNormalizeLayer | L2 normalization for embeddings |
| `"embedding_pooling"` | EmbeddingPoolingLayer | Pooling for sentence-transformers |
| `"qkv_layer"` | QKVLayer | Fused Q/K/V projection |

## 3. CausalLM Model Structures (Verified from Source Code)

### 3.1 Base Transformer / Llama (`transformer.cpp`)

```
Input
  └─ Embedding (embedding_layer or tie_word_embeddings)
     └─ [Decoder Block × NUM_LAYERS]
        ├─ RMSNorm (attention_norm)
        ├─ Attention:
        │   ├─ FC (V)
        │   ├─ FC (K)
        │   ├─ FC (Q)
        │   ├─ MHACore (attention)
        │   └─ FC (O)
        ├─ Addition (residual)
        ├─ RMSNorm (ffn_norm)
        ├─ MLP (SwiGLU):
        │   ├─ FC (ffn_up)
        │   ├─ FC (ffn_gate)
        │   ├─ SwiGLU
        │   └─ FC (ffn_down)
        └─ Addition (residual)
     └─ RMSNorm (output_norm)
     └─ LMHead (lm_head or tie_word_embeddings)
```

### 3.2 Qwen2 (`qwen2_causallm.cpp`)

Same as base Transformer except:
- Q, K, V layers have **bias enabled** (`disable_bias: "false"`)

### 3.3 Qwen3 (`qwen3_causallm.cpp`)

Same as base Transformer except:
- **Adds `reshaped_rms_norm`** after Q and K projections (Q_norm, K_norm)
- MHACore receives Q_norm and K_norm instead of raw Q and K

```
Attention (Qwen3 variant):
  ├─ FC (V)
  ├─ FC (K) → ReshapedRMSNorm (K_norm)
  ├─ FC (Q) → ReshapedRMSNorm (Q_norm)
  ├─ MHACore (Q_norm, K_norm, V)
  └─ FC (O)
```

### 3.4 Gemma3 (`gemma3_causallm.cpp`)

Significantly different structure:
- **GELU-based MLP** instead of SwiGLU
- **Extra post-normalization** layers (post_attention_norm, post_ffn_norm)
- Per-layer attention type support (sliding vs full window)
- `attn_logit_softcapping` parameter

```
[Decoder Block] (Gemma3 variant):
  ├─ RMSNorm (attention_norm)
  ├─ Attention:
  │   ├─ FC (Q) → ReshapedRMSNorm (Q_norm)
  │   ├─ FC (K) → ReshapedRMSNorm (K_norm)
  │   ├─ FC (V)
  │   ├─ MHACore (Q_norm, K_norm, V)
  │   └─ FC (O)
  ├─ RMSNorm (post_attention_norm)    ← EXTRA
  ├─ Addition (residual)
  ├─ RMSNorm (pre_ffn_norm)
  ├─ MLP (GELU variant):              ← DIFFERENT
  │   ├─ FC (gate)
  │   ├─ Activation (tanh_gelu)       ← NOT SwiGLU
  │   ├─ FC (up)
  │   ├─ Multiply (gate_gelu * up)    ← element-wise multiply
  │   └─ FC (down)
  ├─ RMSNorm (post_ffn_norm)          ← EXTRA
  └─ Addition (residual)
```

### 3.5 GptOss (`gptoss_causallm.cpp`)

- Q, K, V, O have **bias enabled**
- **Sink attention** (`use_sink: "true"`)
- **YARN rope scaling** (`rope_scaling_type: "yarn"`)
- MLP replaced with MoE layer (`gpt_oss_moe`)

### 3.6 Qwen3 MoE Variants

Same attention as Qwen3 (with reshaped_rms_norm on Q/K).
MLP replaced with MoE:
- `qwen_moe` (standard, `moe_activation: "swish"`)
- `moe_slim` (FSU optimized)
- `moe_cached_slim` (KV cache optimized)

### 3.7 Embedding Models (Qwen2Embedding, Qwen3Embedding, EmbeddingGemma)

Transformer structure without LM head. Adds:
- `embedding_normalize` (L2 normalization)
- `embedding_pooling` (sentence-transformer pooling)

## 4. Conversion Priority Strategy

### Priority 1: CausalLM Application-Level Patterns (Coarse-Grained)

When a subgraph pattern matches a known CausalLM building block, convert it as a
single NNTrainer custom layer. This preserves optimizations (KV-cache, RoPE, etc.).

| HuggingFace Pattern | NNTrainer CausalLM Layer(s) | Type String(s) |
|---|---|---|
| `model.embed_tokens` (nn.Embedding) | EmbeddingLayer / TieWordEmbedding | `"embedding_layer"` / `"tie_word_embeddings"` |
| Q/K/V Linear + Attention + O Linear | FC + MHACoreLayer + FC | `"fully_connected"` + `"mha_core"` |
| Q/K/V + QKNorm + Attention + O (Qwen3/Gemma3) | FC + ReshapedRMSNorm + MHACore + FC | `"fully_connected"` + `"reshaped_rms_norm"` + `"mha_core"` |
| SwiGLU FFN (gate+up→silu*gate→down) | FC + SwiGLULayer + FC | `"fully_connected"` + `"swiglu"` |
| GeGLU FFN (gate→gelu*up→down) (Gemma3) | FC + Activation + Multiply + FC | `"fully_connected"` + `"activation"` + `"multiply"` |
| RMSNorm (weight * x / rms(x)) | RMSNormLayer | `"rms_norm"` |
| Reshaped RMSNorm | ReshapedRMSNormLayer | `"reshaped_rms_norm"` |
| LM Head (final linear projection) | LmHeadLayer / TieWordEmbedding | `"lm_head"` / `"tie_word_embeddings"` |
| QKV fused projection | QKVLayer | `"qkv_layer"` |
| MoE block (router + experts) | MoE variant | `"qwen_moe"` / `"gpt_oss_moe"` etc. |
| Embedding + L2 Normalize | EmbeddingNormalizeLayer | `"embedding_normalize"` |
| Embedding + Pooling | EmbeddingPoolingLayer | `"embedding_pooling"` |

### Priority 2: NNTrainer Built-in Layers (Medium-Grained)

For modules that don't match CausalLM blocks but correspond to NNTrainer's
registered layer types:

| PyTorch nn.Module | NNTrainer Layer | Type String |
|---|---|---|
| `nn.Linear` | FullyConnectedLayer | `"fully_connected"` |
| `nn.Conv1d` | Conv1DLayer | `"conv1d"` |
| `nn.Conv2d` | Conv2DLayer | `"conv2d"` |
| `nn.ConvTranspose2d` | Conv2DTransposeLayer | `"conv2dtranspose"` |
| `nn.BatchNorm1d/2d/3d` | BatchNormalizationLayer | `"batch_normalization"` |
| `nn.LayerNorm` | LayerNormalizationLayer | `"layer_normalization"` |
| `nn.Embedding` | EmbeddingLayer (built-in) | `"embedding"` |
| `nn.ReLU` | ActivationLayer | `"activation"` with `activation=relu` |
| `nn.LeakyReLU` | ActivationLayer | `"activation"` with `activation=leaky_relu` (slope=0.01 fixed) |
| `nn.GELU` | ActivationLayer | `"activation"` with `activation=gelu` |
| `nn.GELU(approximate='tanh')` | ActivationLayer | `"activation"` with `activation=tanh_gelu` |
| `nn.SiLU` | ActivationLayer | `"activation"` with `activation=swish` |
| `nn.Sigmoid` | ActivationLayer | `"activation"` with `activation=sigmoid` |
| `nn.Tanh` | ActivationLayer | `"activation"` with `activation=tanh` |
| `nn.Softmax` | ActivationLayer | `"activation"` with `activation=softmax` |
| `nn.Softplus` | ActivationLayer | `"activation"` with `activation=softplus` |
| `nn.ELU` | ActivationLayer | `"activation"` with `activation=elu` |
| `nn.SELU` | ActivationLayer | `"activation"` with `activation=selu` |
| `nn.Mish` | ActivationLayer | `"activation"` with `activation=mish` |
| `nn.Dropout` | DropOutLayer | `"dropout"` |
| `nn.LSTM` | LSTMLayer | `"lstm"` |
| `nn.GRU` | GRULayer | `"gru"` |
| `nn.RNN` | RNNLayer | `"rnn"` |
| `nn.MultiheadAttention` | MultiHeadAttentionLayer | `"multi_head_attention"` |
| `nn.MaxPool2d` | Pooling2DLayer | `"pooling2d"` with `pooling=max` |
| `nn.AvgPool2d` | Pooling2DLayer | `"pooling2d"` with `pooling=average` |
| `nn.AdaptiveMaxPool2d((1,1))` | Pooling2DLayer | `"pooling2d"` with `pooling=global_max` |
| `nn.AdaptiveAvgPool2d((1,1))` | Pooling2DLayer | `"pooling2d"` with `pooling=global_average` |
| `nn.Identity` | IdentityLayer | `"identity"` |
| `nn.Upsample` | Upsample2dLayer | `"upsample2d"` |
| `nn.Flatten` | FlattenLayer | `"flatten"` |
| Residual Add (multi-input) | AdditionLayer | `"addition"` |

### Priority 3: NNTrainer Tensor/Op Layers (Fine-Grained)

For tensor-level operations:

| PyTorch Operation | NNTrainer Layer | Type String |
|---|---|---|
| `torch.add` / `Tensor + Tensor` | AddLayer | `"add"` |
| `torch.sub` / `Tensor - Tensor` | SubtractLayer | `"subtract"` |
| `torch.mul` / `Tensor * Tensor` | MultiplyLayer | `"multiply"` |
| `torch.div` / `Tensor / Tensor` | DivideLayer | `"divide"` |
| `torch.matmul` / `Tensor @ Tensor` | MatMulLayer | `"matmul"` |
| `torch.pow` | PowLayer | `"pow"` |
| `torch.sqrt` | SQRTLayer | `"sqrt"` |
| `torch.sin` | SineLayer | `"sin"` |
| `torch.cos` | CosineLayer | `"cos"` |
| `torch.tan` | TangentLayer | `"tan"` |
| `torch.neg` / `-x` | NegativeLayer | `"negative"` |
| `torch.mean` / `Tensor.mean` | ReduceMeanLayer | `"reduce_mean"` |
| `torch.sum` / `Tensor.sum` | ReduceSumLayer | `"reduce_sum"` |
| `torch.cat` | ConcatLayer | `"concat"` |
| `torch.split` | SplitLayer | `"split"` |
| `torch.gather` / indexing | GatherLayer | `"gather"` |
| `Tensor[..., a:b]` / slicing | SliceLayer | `"slice"` |
| `Tensor.to(dtype)` | CastLayer | `"cast"` |
| `Tensor.reshape` / `Tensor.view` | ReshapeLayer | `"reshape"` |
| `Tensor.permute` | PermuteLayer | `"permute"` |

### Priority 4: Unsupported Operations (Fallback)

Operations that have **no NNTrainer equivalent** at any level:

| PyTorch Operation | Status | Workaround |
|---|---|---|
| `torch.exp` | NOT available | Requires custom layer |
| `torch.log` | NOT available | Requires custom layer |
| `torch.clamp` / `torch.clip` | Backend only (not in Tensor API) | Requires custom layer or backend call |
| `torch.where` | NOT available | Requires custom layer |
| `torch.masked_fill` | NOT available | Requires custom layer |
| `Tensor.repeat` / `Tensor.expand` | NOT available | Requires custom layer |
| Comparison ops (`gt`, `lt`, `eq`, `ge`, `le`) | NOT available | Requires custom layer |
| `nn.GroupNorm` | NOT available | Requires custom layer |
| `nn.ReLU6` | NOT available | Use `relu` + custom clamp |
| `nn.Hardswish` / `nn.Hardsigmoid` | NOT available | Requires custom layer |
| `nn.PReLU` | NOT available (LeakyReLU has fixed slope=0.01) | Use `leaky_relu` approximation |
| `nn.Conv3d` / `nn.ConvTranspose1d/3d` | NOT available | Requires custom layer |
| `nn.AdaptiveAvgPool1d` / `nn.AvgPool1d` / `nn.MaxPool1d` | NOT available | Reshape + 2D pooling workaround |
| `nn.Dropout2d` / `nn.Dropout3d` | NOT available | Use standard `dropout` |

These will be flagged in the conversion report with suggested workarounds.

## 5. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TorchFXConverter                      │
│                                                          │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐ │
│  │  Tracer   │──▶│ PatternMatcher│──▶│  NNTrainer       │ │
│  │ (runtime) │   │ (subgraph    │   │  ModelEmitter    │ │
│  │           │   │  detection)  │   │  (code/config    │ │
│  │           │   │              │   │   generation)    │ │
│  └──────────┘   └──────────────┘   └──────────────────┘ │
│       │               │                    │             │
│       ▼               ▼                    ▼             │
│  FX Graph        Mapped Nodes      NNTrainer Model      │
│  (nodes with     (with nntrainer   (C++ code or         │
│   metadata)       layer mapping)    JSON config)        │
└─────────────────────────────────────────────────────────┘
```

### 5.1 Component Details

#### A. Tracer (Existing Code - Enhanced)

The provided `Tracer` class using `TorchFunctionMode` + module hooks.

**Enhancements needed:**
- Update `LEAF_MODULES` to include NNTrainer-mappable modules (see Section 6)
- Add metadata collection for weight shapes, dtypes, and parameter values
- Track module hierarchy for pattern matching

#### B. PatternMatcher

Detects subgraph patterns in the FX graph that correspond to CausalLM building blocks.

**Key patterns to detect:**

1. **Transformer Decoder Block Pattern (Llama/Qwen2 style):**
   ```
   RMSNorm -> Q_proj, K_proj, V_proj -> MHACore -> O_proj -> Add (residual)
   -> RMSNorm -> gate_proj, up_proj -> SwiGLU -> down_proj -> Add (residual)
   ```

2. **Transformer Decoder Block Pattern (Qwen3 style):**
   ```
   RMSNorm -> Q_proj -> Q_norm, K_proj -> K_norm, V_proj -> MHACore -> O_proj -> Add
   -> RMSNorm -> gate_proj, up_proj -> SwiGLU -> down_proj -> Add
   ```

3. **Transformer Decoder Block Pattern (Gemma3 style):**
   ```
   RMSNorm -> Q -> Q_norm, K -> K_norm, V -> MHACore -> O -> RMSNorm(post) -> Add
   -> RMSNorm -> gate -> GELU, up -> Multiply -> down -> RMSNorm(post) -> Add
   ```

4. **RMSNorm Pattern:**
   ```
   variance = mean(x^2, dim=-1, keepdim=True)
   x_norm = x * rsqrt(variance + eps)
   output = weight * x_norm
   ```

5. **SwiGLU Pattern:**
   ```
   gate = silu(gate_proj(x))
   up = up_proj(x)
   output = gate * up
   ```

6. **GeGLU Pattern (Gemma3):**
   ```
   gate = gelu(gate_proj(x))
   up = up_proj(x)
   output = gate * up
   ```

7. **Rotary Embedding Pattern:**
   ```
   cos/sin computation -> complex multiply -> apply to Q/K
   ```

#### C. NNTrainer ModelEmitter

Generates NNTrainer model construction code:

- **JSON Config Mode**: Produces `nntr_config.json` compatible with CausalLM app
- **C++ Code Mode**: Generates `createLayer()` calls matching `transformer.cpp` style
- **Weight Converter**: Generates `weight_converter.py` script for PyTorch→NNTrainer
  weight format conversion

## 6. Updated LEAF_MODULES

The `LEAF_MODULES` in the Tracer should be redefined to match NNTrainer's conversion
granularity:

```python
# Level 1: NNTrainer built-in layer equivalents
NNTRAINER_LAYER_MODULES = (
    nn.Linear,           # -> fully_connected
    nn.Conv1d,           # -> conv1d
    nn.Conv2d,           # -> conv2d
    nn.ConvTranspose2d,  # -> conv2dtranspose
    nn.BatchNorm1d,      # -> batch_normalization
    nn.BatchNorm2d,      # -> batch_normalization
    nn.BatchNorm3d,      # -> batch_normalization
    nn.LayerNorm,        # -> layer_normalization
    nn.Embedding,        # -> embedding
    nn.LSTM,             # -> lstm
    nn.GRU,              # -> gru
    nn.RNN,              # -> rnn
    nn.MultiheadAttention,  # -> multi_head_attention
    nn.Dropout,          # -> dropout
    nn.MaxPool2d,        # -> pooling2d(max)
    nn.AvgPool2d,        # -> pooling2d(average)
    nn.AdaptiveAvgPool2d, # -> pooling2d(global_average)
    nn.AdaptiveMaxPool2d, # -> pooling2d(global_max)
    nn.Upsample,         # -> upsample2d
    nn.Identity,         # -> identity
)

# Level 2: Activation modules (mapped to nntrainer activation layer)
NNTRAINER_ACTIVATION_MODULES = (
    nn.ReLU,       # -> activation(relu)
    nn.LeakyReLU,  # -> activation(leaky_relu) [slope=0.01 fixed]
    nn.GELU,       # -> activation(gelu) or activation(tanh_gelu)
    nn.SiLU,       # -> activation(swish)
    nn.Sigmoid,    # -> activation(sigmoid)
    nn.Tanh,       # -> activation(tanh)
    nn.Softmax,    # -> activation(softmax)
    nn.Softplus,   # -> activation(softplus)
    nn.ELU,        # -> activation(elu)
    nn.SELU,       # -> activation(selu)
    nn.Mish,       # -> activation(mish)
)

# HuggingFace-specific modules to treat as leaf (populated dynamically)
HUGGINGFACE_LEAF_MODULES = ()
# Populated based on model architecture detection:
#   - LlamaRMSNorm / Qwen2RMSNorm / Qwen3RMSNorm / Gemma3RMSNorm -> rms_norm
#   - LlamaMLP / Qwen2MLP -> swiglu pattern
#   - Gemma3MLP -> geglu pattern
#   - LlamaAttention / Qwen2Attention / etc. -> mha_core pattern
#   etc.

# Combined LEAF_MODULES for the Tracer
LEAF_MODULES = (
    NNTRAINER_LAYER_MODULES +
    NNTRAINER_ACTIVATION_MODULES +
    tuple(HUGGINGFACE_LEAF_MODULES)
)
```

## 7. HuggingFace Model Architecture Detection

The converter auto-detects the HuggingFace model architecture and configures
pattern matching accordingly:

```python
# Supported HuggingFace architectures and their module mappings
HF_ARCHITECTURE_MAP = {
    "LlamaForCausalLM": {
        "rms_norm": ["LlamaRMSNorm"],
        "attention": ["LlamaAttention", "LlamaSdpaAttention"],
        "mlp": ["LlamaMLP"],           # SwiGLU pattern
        "mlp_type": "swiglu",
        "decoder_layer": ["LlamaDecoderLayer"],
        "q_norm": False,               # No Q/K normalization
        "bias_in_qkv": False,
    },
    "Qwen2ForCausalLM": {
        "rms_norm": ["Qwen2RMSNorm"],
        "attention": ["Qwen2Attention", "Qwen2SdpaAttention"],
        "mlp": ["Qwen2MLP"],           # SwiGLU pattern
        "mlp_type": "swiglu",
        "decoder_layer": ["Qwen2DecoderLayer"],
        "q_norm": False,
        "bias_in_qkv": True,           # Qwen2 uses bias in Q/K/V
    },
    "Qwen3ForCausalLM": {
        "rms_norm": ["Qwen3RMSNorm"],
        "attention": ["Qwen3Attention"],
        "mlp": ["Qwen3MLP"],           # SwiGLU pattern
        "mlp_type": "swiglu",
        "decoder_layer": ["Qwen3DecoderLayer"],
        "q_norm": True,                # Uses reshaped_rms_norm on Q/K
        "bias_in_qkv": False,
    },
    "Qwen3MoeForCausalLM": {
        "rms_norm": ["Qwen3MoeRMSNorm"],
        "attention": ["Qwen3MoeAttention"],
        "mlp": ["Qwen3MoeSparseMoeBlock"],
        "mlp_type": "moe",             # MoE pattern
        "moe_layer_type": "qwen_moe",
        "moe_activation": "swish",
        "decoder_layer": ["Qwen3MoeDecoderLayer"],
        "q_norm": True,
        "bias_in_qkv": False,
    },
    "Gemma3ForCausalLM": {
        "rms_norm": ["Gemma3RMSNorm"],
        "attention": ["Gemma3Attention"],
        "mlp": ["Gemma3MLP"],
        "mlp_type": "geglu",           # GELU + Multiply pattern (NOT SwiGLU)
        "decoder_layer": ["Gemma3DecoderLayer"],
        "q_norm": True,
        "bias_in_qkv": False,
        "post_attention_norm": True,   # Extra post-attention RMSNorm
        "post_ffn_norm": True,         # Extra post-FFN RMSNorm
        "attn_logit_softcapping": True,
    },
    "GptOssForCausalLM": {
        "rms_norm": ["GptOssRMSNorm"],
        "attention": ["GptOssAttention"],
        "mlp": ["GptOssMoEBlock"],
        "mlp_type": "moe",
        "moe_layer_type": "gpt_oss_moe",
        "decoder_layer": ["GptOssDecoderLayer"],
        "q_norm": False,
        "bias_in_qkv": True,
        "use_sink": True,              # Sink attention
        "rope_scaling_type": "yarn",   # YARN rope scaling
    },
}
```

## 8. Conversion Pipeline

```python
class TorchFXToNNTrainerConverter:
    def __init__(self, model, sample_input):
        self.model = model
        self.sample_input = sample_input
        self.arch = self._detect_architecture()

    def convert(self):
        # Phase 1: Trace
        leaf_modules = self._build_leaf_modules()
        tracer = Tracer(self.model, leaf_modules=leaf_modules)
        with tracer:
            output = self.model(**self.sample_input)
        graph = tracer.graph

        # Phase 2: Pattern Match & Map
        mapped_graph = self._map_nodes(graph)

        # Phase 3: Emit
        nntrainer_config = self._emit_config(mapped_graph)
        weight_converter = self._emit_weight_converter(mapped_graph)

        return nntrainer_config, weight_converter

    def _detect_architecture(self):
        """Detect HuggingFace model architecture from model class name."""
        ...

    def _build_leaf_modules(self):
        """Build LEAF_MODULES tuple based on detected architecture."""
        ...

    def _map_nodes(self, graph):
        """Map FX graph nodes to NNTrainer layer definitions."""
        for node in graph.nodes:
            if node.op == 'call_module':
                self._map_module_node(node)
            elif node.op == 'call_function':
                self._map_function_node(node)
            elif node.op == 'call_method':
                self._map_method_node(node)
        ...

    def _map_module_node(self, node):
        """Map a call_module node using Priority 1 -> 2 -> 3 -> 4."""
        ...

    def _map_function_node(self, node):
        """Map torch.* function calls to NNTrainer ops."""
        ...

    def _map_method_node(self, node):
        """Map tensor method calls (reshape, view, etc.) to NNTrainer ops."""
        ...
```

## 9. Weight Conversion Strategy

NNTrainer uses a flat binary weight format. The converter generates a Python script
that:

1. Loads PyTorch model state_dict
2. Reorders weights to match NNTrainer's layer ordering
3. Handles dtype conversion (FP32 -> FP16/INT8 as specified)
4. Handles transposition differences (NNTrainer FC weights may differ from PyTorch)
5. Saves in NNTrainer binary format

This follows the existing pattern in `Applications/CausalLM/res/*/weight_converter.py`.

## 10. Output Artifacts

The converter produces:
1. **`nntr_config.json`** - NNTrainer model configuration
2. **`weight_converter.py`** - Weight conversion script
3. **`conversion_report.txt`** - Mapping report showing:
   - Successfully mapped layers with NNTrainer type strings
   - Unsupported operations (with warnings and workarounds)
   - Shape validation results
   - Model architecture detected

## 11. File Structure

```
Applications/TorchFXConverter/
├── DESIGN.md                    # This document
├── converter.py                 # Main converter class
├── tracer.py                    # Enhanced Tracer (from provided code)
├── pattern_matcher.py           # Subgraph pattern detection
├── node_mapper.py               # FX node -> NNTrainer layer mapping
├── model_emitter.py             # NNTrainer config/code generation
├── weight_converter_gen.py      # Weight converter script generator
├── hf_architectures.py          # HuggingFace architecture definitions
└── tests/
    ├── test_converter_linear.py
    ├── test_converter_llama.py
    └── test_converter_qwen.py
```

## 12. Key Design Decisions

### 12.1 Why Runtime Tracing (not Symbolic)?

HuggingFace models contain extensive conditionals (cache handling, attention mask
computation, sliding window logic, etc.) that make symbolic tracing impractical.
Runtime tracing with real inputs captures the exact execution path for a given
configuration.

### 12.2 Why Coarse-Grained Mapping First?

CausalLM layers like `MHACoreLayer` include critical optimizations:
- KV-cache management for incremental inference
- RoPE (Rotary Position Embedding) integration
- Sliding window attention
- GQA (Grouped Query Attention) support

Mapping at the fine-grained (tensor op) level would lose these optimizations.

### 12.3 Architecture-Specific vs Generic?

We support both:
- **Architecture-specific path**: For known HuggingFace architectures (Llama, Qwen2/3,
  Gemma3, GptOss), use predefined patterns for reliable conversion
- **Generic path**: For unknown architectures, fall back to layer-by-layer mapping
  using Priority 2-4 rules

### 12.4 Layer Registration Note

CausalLM custom layers (swiglu, rms_norm, mha_core, etc.) are NOT part of
NNTrainer's core `app_context.cpp` registration. They must be registered by the
application at runtime via `registerCustomLayers()`. The converter must ensure
the generated code or config includes these registrations.

# CausalLM Inference with NNTrainer

- This application provides examples to run causal LLM models using NNTrainer.
- This example only provides *inference* mode, not *training* mode yet.

## Supported Models

- Llama (LlamaForCausalLM)
- Qwen3 (1.7b/4b/7b/14b) (Qwen3ForCausalLM)
- Qwen3 MoE (30b-A3b) (Qwen3MoeForCausalLM)
- GPT-OSS-20b (GptOssForCausalLM)
- You can try your own model with custom layers!
- Feel free to contribute!

## How to Run

- Download and copy the model files from HuggingFace to `res/{model}` directory.
- The folder should contain
    - config.json
    - generation_config.json
    - tokenizer.json
    - tokenizer_config.json
    - vocab.json
    - nntr_config.json
    - NNTrainer weight binary file (matches with the name in nntr_config.json)
    - which are usually included in HF model deployment.
- Compile the Application
- If you test CausalLM on your PC, build with `-Denable-transformer=true`
- Run the model with the following command

```
$ cd build/Applications/CausalLM
$ ./nntr_causallm {your model config folder}
```

e.g.,

```
$ ./nntr_causallm /tmp/nntrainer/Applications/CausalLM/res/qwen3-4b/
```

### Recommended Configuration

- PC test
```
$ meson build -Denable-fp16=true -Dggml-thread-backend=omp -Denable-transformer=true -Domp-num-threads=4
$ export OMP_THREAD_LIMIT=16 && export OMP_WAIT_POLICY=active && export OMP_PROC_BIND=true && export OMP_PLACES=cores && export OMP_NUM_THREADS=4
```

- Android test
```
$ ./tools/package_android.sh -Domp-num-threads=4 -Dggml-thread-backend=omp
```

## Model Implementations

| Factory Key | Source File | Description |
|:---:|:---:|:---|
| LlamaForCausalLM | causal_lm | Base Llama decoder-only transformer model |
| Qwen3ForCausalLM | qwen3_causallm | Basic implementation of Qwen3 model |
| Qwen3MoeForCausalLM | qwen3_moe_causallm | Basic implementation of Qwen3 MoE model |
| Qwen3SlimMoeForCausalLM | qwen3_slim_moe_causallm | FSU-scheme-activated Qwen3 MoE model (not recommended) |
| NNTRQwen3ForCausalLM | nntr_qwen3_causallm | Q/K/V parallelized Qwen3 model (not recommended) |
| NNTRQwen3MoECausalLM | nntr_qwen3_moe_causallm | Q/K/V parallelized Qwen3 MoE model (not recommended) |
| Qwen3CachedSlimMoeForCausalLM | qwen3_cached_slim_moe_causallm | MoE-specific FSU-based Qwen3 MoE model with KV-cache |
| GptOssForCausalLM | gptoss_causallm | Basic implementation of GPT-OSS model |
| GptOssCachedSlimCausalLM | gptoss_cached_slim_causallm | MoE-specific FSU-based GPT-OSS model with KV-cache |

## Architecture

All models follow a decoder-only transformer structure:

```
[Input] -> [Embedding] -> [Decoder Block] x N -> [RMSNorm] -> [LMHead] -> [Output]
```

### Custom Layers (in `layers/` directory)

| Layer | Description |
|:---:|:---|
| EmbeddingLayer | Token embedding with incremental forwarding support |
| MHACore | Multi-head attention with RoPE, GQA, sliding window, KV-cache |
| QKVLayer | Parallel Q/K/V projections |
| RMSNorm | Root Mean Square Layer Normalization |
| ReshapedRMSNorm | RMSNorm with reshaping for Q/K projections |
| SwiGLU | Swish-Gated Linear Unit activation |
| TieWordEmbedding | Weight tying between embedding and LM head |
| QwenMoELayer | Qwen3 Mixture of Experts (based on Llama-MoE) |
| QwenMoELayerFSU | FSU-scheme MoE for reduced memory |
| QwenMoELayerCached | FSU-based MoE with expert caching and prediction scoring |
| GptOssMoELayer | GPT-OSS Mixture of Experts with SiLU activation |
| GptOssMoELayerCached | FSU-based GPT-OSS MoE with expert caching |

### Key Features

- **Grouped-Query Attention (GQA)**: Configurable Q and KV head counts
- **Rotary Positional Embeddings (RoPE)**: With configurable theta and YARN scaling support
- **KV-Cache**: For efficient autoregressive inference
- **Sliding Window Attention**: For limited attention context
- **Mixture of Experts (MoE)**: Top-k expert routing with optional caching
- **FSU (Free Sweep Update)**: Memory swap optimization for large models
- **Quantization**: Support for Q4_0, FP16, and mixed precision (e.g., Q4_0-FP32)
- **Vocabulary Selection**: Optional vocabulary pruning for efficiency

## Pre-configured Models (in `res/` directory)

| Model | Directory | Tensor Type |
|:---:|:---:|:---:|
| Qwen3-4B | res/qwen3-4b/ | FP32-FP32 |
| Qwen3-30B-A3B (MoE) | res/qwen3-30b-a3b/ | Q4_0-FP32 |
| Qwen3-30B-A3B Cached | res/qwen3-30b-a3b-slim-cached/ | Q4_0-FP32 |
| GPT-OSS-20B | res/gpt-oss-20b/ | FP32-FP32 |

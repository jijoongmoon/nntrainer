# Qwen2.5-Omni (Thinker text model) for nntrainer CausalLM

Runs the **Thinker text decoder** of
[Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)
(text in / text out) on the nntrainer CausalLM application.

The Thinker's text model is a Qwen2.5-style decoder (Q/K/V bias, GQA,
RMSNorm, SwiGLU, untied lm_head), so it reuses the Qwen2 transformer graph
(`models/qwen25_omni/`). For pure-text inputs Qwen's M-RoPE is numerically
identical to standard 1-D RoPE, so no RoPE changes are needed.

Not included (future work): audio/vision encoders, the Talker and the
Token2Wav speech decoder.

## 1. Convert the HF checkpoint to an nntrainer FP32 .bin

```bash
pip install torch numpy safetensors huggingface_hub

python weight_converter.py \
    --model_path Qwen/Qwen2.5-Omni-3B \
    --output_dir ./qwen2.5-omni-3b
```

`--model_path` accepts a local checkpoint directory or a HuggingFace repo id
(downloads config/tokenizer/safetensors only). The script streams the thinker
weights tensor-by-tensor from the safetensors shards, so it does not need a
transformers version that knows Qwen2.5-Omni, and skips the audio/vision/
talker towers entirely.

The output directory then contains everything the app needs:

```
qwen2.5-omni-3b/
├── nntr_qwen2.5_omni_3b_fp32.bin   # FP32 weights (~12.7 GiB)
├── config.json                     # HF config (kept as-is; app flattens it)
├── generation_config.json          # eos/bos token ids for the runtime
├── nntr_config.json                # FP32 runtime config
├── tokenizer.json
└── tokenizer_config.json
```

## 2. Quantize to Q4_0

Use the `nntr_quantize` tool built alongside the application:

```bash
# FC layers -> Q4_0, embedding/lm_head stay FP32
nntr_quantize ./qwen2.5-omni-3b --fc_dtype Q4_0

# or additionally squeeze the embedding to Q6_K (~2.0 GiB total).
# Omni-3B's lm_head is untied and LmHeadLayer::save supports Q4_0 but not
# Q6_K, so the lm_head dtype must be pinned explicitly:
nntr_quantize ./qwen2.5-omni-3b --fc_dtype Q4_0 --embd_dtype Q6_K --lmhead_dtype Q4_0
```

When quantizing in-place, the new config is written as
`nntr_config_quantized.json` — rename it to `nntr_config.json` (keep a copy
of the FP32 one if needed):

```bash
mv ./qwen2.5-omni-3b/nntr_config_quantized.json ./qwen2.5-omni-3b/nntr_config.json
```

## 3. Run

```bash
nntr_causallm ./qwen2.5-omni-3b "Give me a short introduction to large language model."
```

Without a prompt argument, the `sample_input` from `nntr_config.json` is used.

## Audio encoder (Thinker audio tower)

The Whisper-style audio tower runs as a standalone encoder
(`Qwen25OmniAudioEncoder`, `models/qwen25_omni/qwen25_omni_audio_encoder.*`).
Because Omni's audio attention is strictly windowed (no attention across
200-mel-frame windows, positions restart per window), the graph is compiled
per-chunk and run repeatedly — mathematically identical to the HF reference.

```bash
# convert (writes encoder + head .bin, config.json, nntr_config.json)
python audio_encoder_converter.py \
    --model_path Qwen/Qwen2.5-Omni-3B --output_dir ./qwen2.5-omni-3b-audio

# encode a mel feature file ([int32 n_mels][int32 n_frames][fp32 data],
# n_frames even) into 25 embeddings/sec ([int32 n][int32 2048][fp32 data]):
nntr_causallm ./qwen2.5-omni-3b-audio mel_input.bin   # -> mel_input.bin.embd

# verify against HF (transformers >= 4.52):
python test_audio_encoder.py --audio_model_dir ./qwen2.5-omni-3b-audio \
    --binary <build>/Applications/CausalLM/nntr_causallm
```

Verified vs HF fp32: identical token counts, per-token cosine > 0.99996
(residual diff comes from mha_core's internal fp16 KV cache).

### Q4_0 audio encoder

The 32 encoder blocks' FC weights (q/k/v/out/fc1/fc2) quantize to Q4_0;
conv, layernorm, sinusoid pos-embed and the head proj stay FP32. The
encoder's FC layers carry `weight_dtype=Q4_0` while `model_tensor_type`
stays FP32, so only those weights are 4-bit.

```bash
python audio_encoder_converter.py --model_path Qwen/Qwen2.5-Omni-3B \
    --output_dir ./qwen2.5-omni-3b-audio-q4 --fc-dtype q4_0 --target x86
```

Encoder bin: 2423 MiB -> 360 MiB. End-to-end speech chat with the Q4_0
encoder + Q4_0/Q6_K decoder produces the same answer as the all-FP32
pipeline (the decoder LLM absorbs the encoder's 4-bit noise).

## Audio chat (speech in / text out)

`Qwen25OmniAudioChat` (`models/qwen25_omni/qwen25_omni_audio_causallm.*`)
combines the audio encoder and the text decoder: an `embedding_injection`
layer replaces each `<|AUDIO|>` placeholder embedding with the encoder
output at the same position (the HF masked_scatter equivalent). The decoder
loads the SAME .bin as the text-only model.

Model directory (e.g. `models/qwen2.5-omni-3b-chat/`):
- `config.json` — HF Omni config with `architectures: ["Qwen25OmniAudioChat"]`
- `nntr_config.json` — text-model config plus
  `"audio_encoder_path": "<dir of the converted audio encoder>"`
- decoder `.bin` (symlink to the text model's is fine)
- do NOT place `tokenizer_config.json` here (the audio prompt builds its own
  chat template)

```bash
# 16 kHz 16-bit PCM wav goes straight in (mel computed in C++,
# models/qwen25_omni/whisper_mel.cpp — a WhisperFeatureExtractor port):
nntr_causallm ./qwen2.5-omni-3b-chat \
    "audio:question.wav What do you hear in this audio? Answer briefly."

# or precompute mel features in Python and pass the .mel file instead:
python wav_to_mel.py question.wav                     # -> question.mel
nntr_causallm ./qwen2.5-omni-3b-chat "audio:question.mel <question>"
```

Verified vs HF fp32 greedy decoding: identical generated text. The
standalone encoder also accepts .wav directly and drops the computed
features next to it as "<file>.wav.mel" for inspection.

## Vision encoder (Thinker visual tower, standalone)

`Qwen25OmniVisionEncoder` (`models/qwen25_omni/qwen25_omni_vision_encoder.*`)
runs the Qwen2.5-VL-style image tower: patch_embed (Conv3d as linear) -> 32
pre-norm blocks (RMSNorm, separate q/k/v with bias, 2D-RoPE via the custom
`vision_rope` layer, full bidirectional mha_core, proj, SwiGLU) -> patch
merger (RMSNorm + 2x2 spatial merge + MLP) -> 2048-d embeddings.

Windowed attention is supported for any image size: the custom
`vision_attention` layer masks each token to its window (block-diagonal)
without reordering patches — token a attends b iff they share a window, with
window ids derived from the grid at finalize. Full-attention layers
(`fullatt_block_indexes` = [7,15,23,31]) attend across the whole image. This
replaces mha_core in the tower (no KV cache, exact FP32). 2D-RoPE stays in
the original patch order. The graph is still compiled for a fixed grid
(grid_h/grid_w), so one converted dir serves one image size; convert per
size (e.g. --grid_h 16 --grid_w 16 for 224x224).

```bash
python vision_encoder_converter.py --model_path Qwen/Qwen2.5-Omni-3B \
    --output_dir ./qwen2.5-omni-3b-vision --grid_h 8 --grid_w 8

# encode a patch feature file ([int32 gh][int32 gw][fp32 patches[gh*gw][1176]]):
nntr_causallm ./qwen2.5-omni-3b-vision patches.bin     # -> patches.bin.embd

python test_vision_encoder.py --vision_model_dir ./qwen2.5-omni-3b-vision \
    --binary <build>/Applications/CausalLM/nntr_causallm
```

Video (multi-frame) is handled by the same encoder: pass `--grid_t N` (N
temporal patches) at convert time. Attention never crosses frames (HF builds
cu_seqlens per frame even for the full-attention layers), so `vision_attention`
puts each frame in its own window group (full layers → per frame, windowed →
per spatial window of a frame), and the 2D-RoPE spatial positions repeat per
frame. The merger still groups 4 consecutive (2x2 spatial) patches.

Verified vs HF fp32 (per-token cosine = 1.000000, max abs diff ~1e-4):
112x112 image (1 window), 224x224 image (4 windows + full layers), and a
2-frame 8x8 video (grid_t=2, per-frame windowing). The temporal position
scaling for the decoder (video t_index) is the remaining piece for video
chat; the encoder embeddings themselves match.

## Image chat (image + text in / text out)

`Qwen25OmniVisionChat` (`models/qwen25_omni/qwen25_omni_vision_causallm.*`)
wires the vision encoder into the thinker decoder. Two pieces beyond the
audio path:

- **M-RoPE** — each attention layer applies the host-computed rotary cos/sin
  (built from the 3D t/h/w position ids per HF `get_rope_index`) to q and k
  via the custom `mrope_apply` layer, with `mha_core` at `rope_theta=0`
  (so the core attention/KV-cache is untouched for every other model). For
  pure text the 3 axes are equal, so M-RoPE reduces to 1-D RoPE.
- **embedding_injection** replaces `<|IMAGE|>` placeholders with vision
  embeddings, exactly like the audio path.

Model dir (e.g. `models/qwen2.5-omni-3b-vchat/`): `config.json` with
`architectures: ["Qwen25OmniVisionChat"]`, `nntr_config.json` with
`"vision_encoder_path": "<converted vision encoder dir>"`, the decoder `.bin`
(symlink to the text model), and NO `tokenizer_config.json` (the image prompt
builds its own chat template).

Video uses the same path with a `video:` prefix and a `<|VIDEO|>` (151656)
placeholder; the decoder's M-RoPE applies the temporal t_index
(`frame * video_second_per_grid * position_id_per_seconds`, set in
nntr_config) to video tokens. The vision_encoder_path must point to a video
(grid_t>1) encoder; the embedding_injection layer matches both image and
video tokens. The prompt delimiters are the real tokens `<|vision_bos|>` /
`<|vision_eos|>` (151652/153) — NOT "<|vision_start|>", which tokenizes to
junk.

```bash
# patches.bin: [int32 gh][int32 gw][fp32 (t*gh*gw) x 1176] from the HF processor
nntr_causallm ./qwen2.5-omni-3b-vchat \
    "image:patches.bin What colors do you see? Answer briefly."
nntr_causallm ./qwen2.5-omni-3b-vchat-video \
    "video:vpatches.bin What colors do you see? Answer briefly."

python test_vision_chat.py --vchat_dir ./qwen2.5-omni-3b-vchat \
    --binary <build>/Applications/CausalLM/nntr_causallm
python test_video_chat.py --vchat_dir ./qwen2.5-omni-3b-vchat-video \
    --binary <build>/Applications/CausalLM/nntr_causallm
```

Verified vs HF fp32 greedy: image (112/224px) and a 2-frame video both
produce "red, green, blue" identically to the HF thinker.

Verified vs HF fp32 greedy: on a confident image (RGB stripes, "What colors
do you see?") both produce "red, green, blue", at 112x112 (1 window) and
224x224 (4 windows) alike. On high-entropy inputs (e.g. pure noise) the first
token can differ by a near-tie, as with any fp16-KV / approx-kernel decode.
**Scope: single image** (one image per prompt; the decoder M-RoPE handles any
grid, the encoder is compiled per image size).

## Weight layout notes (for maintainers)

The .bin layout follows nntrainer's weight load order, which is the
symbolic graph's DFS-from-output order (`Model::compile` in
`api/ccapi/src/tensor_api_graph.cpp`) — **not** layer creation order:

```
embed_tokens (vocab, hidden)                      # no transpose
per layer:
  input_layernorm
  q_proj.weight^T, q_proj.bias
  k_proj.weight^T, k_proj.bias
  v_proj.weight^T, v_proj.bias
  o_proj.weight^T
  post_attention_layernorm
  gate_proj.weight^T                              # gate loads first!
  up_proj.weight^T
  down_proj.weight^T
final norm
lm_head.weight^T                                  # untied for Omni-3B
```

FP32 FC weights are stored transposed, i.e. as (in_features, out_features).
`createMlp()` creates the `ffn_up` FC first, but the graph wires
`swiglu({gate, up})` and the DFS visits the gate branch first, so
`gate_proj` must be written **before** `up_proj` (verified against HF
greedy decoding with a synthetic checkpoint; same order as the qwen2
converter since commit de8f981cf).

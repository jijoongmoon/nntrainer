# Qwen2.5-Omni Token2Wav (Phase 2): codec ids → 24 kHz waveform

## Context

Phase 1 (Talker → codec token ids) is DONE & HF-verified (committed `5ee53d971`). Phase 2 turns those codec ids into a 24 kHz waveform — the last piece of Qwen2.5-Omni speech output. It has two subsystems:
1. **DiT** (flow-matching): codec ids + speaker conditioning → 80-dim mel, via a host-driven RK4 ODE sampler (36 DiT forward calls).
2. **BigVGAN** (vocoder): mel → waveform, via ConvTranspose1d upsampling + dilated convs + snake activation.

Token2Wav is the largest single subsystem. **Decision (user-approved): build BigVGAN first** (deterministic, no RNG/ODE — isolates the genuinely-new conv ops and de-risks the conv-weight-layout unknown), then DiT, then end-to-end. **ECAPA speaker encoder is precomputed in Python (MVP)** — the speaker `cond` is constant per speaker; the in-C++ ECAPA port is deferred.

All facts below were ground-truthed against the checkpoint (809 token2wav tensors, **no weight_norm**, snakebeta α/β present) and HF `modeling_qwen2_5_omni.py` (transformers 4.53), correcting reader mis-reads (codec vocab **8194** not 8193; RoPE on **head 0 only**; `norm_out` is AdaLN-final not Linear).

## Confirmed architecture (HF, ground truth)

**Top** `Qwen2_5OmniToken2WavModel.forward(code, conditioning[1,192], reference_mel[1,400,80], num_steps=10, guidance_scale=0.5, sway=-1.0)` (modeling.py:3624): `mel = dit.sample(...)` → `wav = bigvgan(mel)`.

**BigVGAN** (modeling.py:3308-3379; 449 tensors): `process_mel`(host elementwise: exp→amp→dB→normalize) → `conv_pre` Conv1d(80→1536,k7,p3) → 6 upsample stages i: `ConvTranspose1d`(strides[5,3,2,2,2,2], kernels[11,7,4,4,4,4], pad=(k−stride)//2, ch 1536→768→384→192→96→48→24) then **mean of 3 AMPBlocks** (dilated Conv1d dilations[1,3,5] + snakebeta) → `activation_post`(snakebeta+anti-alias) → `conv_post` Conv1d(24→1,k7,p3,**no bias**) → clamp[-1,1]. 1440× upsample → 24 kHz. **snakebeta**: `x + (1/(exp(β)+1e-9))·sin²(exp(α)·x)`, per-channel α,β (log-domain, init 0). **TorchActivation1d**: Kaiser-sinc ×2 upsample → snakebeta → ×2 downsample (anti-alias).

**DiT** (modeling.py:3443-3590; 360 tensors): hidden 1024, 22 blocks, 16 heads, head_dim 64, ff_mult 2, codec embed [8194,512], repeats 2. Per block: AdaLN-Zero(time→6 gates) → attention(q/k/v/o 1024², **RoPE head 0 only**, block-diff mask) → gated residual → AdaLN(scale/shift) → MLP(1024→2048→1024, tanh-GELU) → gated residual. Final AdaLN-Final + proj_out(1024→80). Sampler: noise `randn[1,30000,80][:,:T]`; 10 sway time points; **RK4** (9 intervals × 4 stages = 36 DiT calls); CFG `guided+(guided−null)·0.5`.

## Sub-phase 2A — BigVGAN (mel → waveform) — DO FIRST

**New nntrainer layers** (app `Applications/CausalLM/layers/`, registered like `mrope_apply`):
- `conv1d_transpose` — thin wrapper delegating to core `Conv2DTransposeLayer` (`nntrainer/layers/conv2d_transpose_layer.h`) with H=1, exactly as `Conv1DLayer` wraps `Conv2DLayer` (`nntrainer/layers/conv1d_layer.h:107`). **Spike first** (see Risks): confirm PyTorch ConvTranspose1d weight `[in,out,k]` → the layout Conv2DTranspose expects, and that `pad=(k−stride)//2` with no `output_padding` gives HF's output length.
- `snake_beta` — custom layer, 2 per-channel weights (α,β), `out = x + sin²(exp(α)·x)/(exp(β)+1e-9)`; mirror `mrope_apply.{h,cpp}` structure (finalize + incremental_forwarding + 2 weights).
- `antialiased_snake` — wraps snake_beta with fixed Kaiser-sinc ×2 up/down (depthwise, replicate-pad). Filters computed at finalize (`kaiser_sinc_filter1d`, modeling.py:3094-3137). **Highest-risk piece** — prototype against HF on one AMPBlock.
- **Reuse**: core `Conv1DLayer` ("conv1d") with dilation for conv_pre/conv_post/resblock convs.

**Reuse**: `Conv1DLayer` (dilation support), the **two-graph / non-incremental run** idiom from `qwen25_omni_audio_encoder.cpp:215-244` (compile once, `incremental_inference(BATCH,{mel},{},T,0,T,false)`).

**Files:**
- `res/qwen2.5-omni/token2wav_converter.py` (new) — `bigvgan.bin` (conv kernels **NOT transposed**, plain `[out,in,k]`; snakebeta α/β as `[C]`; no weight_norm folding — assert none present) + later DiT. Reuse `ShardedSafetensors`/`save_fc` from `weight_converter.py`; conv-no-transpose precedent `audio_encoder_converter.py:141`.
- `models/qwen25_omni/qwen25_omni_bigvgan.{h,cpp}` (new) — build the conv graph, host `process_mel`, non-incremental run, write 24 kHz WAV.
- `res/qwen2.5-omni/dump_token2wav_refs.py` (new) — HF reference dumper.
- edit `models/qwen25_omni/meson.build`, `layers/meson.build`, register layers.

**Verify (Stage C):** HF dumps `mel` + per-stage tensors (post-conv_pre, each ups, each AMPBlock, conv_post, final wav). nntrainer feeds the dumped mel; compare layer-by-layer (conv_pre→catches layout; one AMPBlock→catches snake/anti-alias; full wav max-abs err < ~1e-3, FP32). Listen test.

## Sub-phase 2B — DiT (codec + speaker → mel)

**New custom layers:** `dit_rope` (head-0-only, `[...,d/2,2]` pair-rotate, head_dim 64), `dit_adaln_zero` + `dit_adaln_final` (LayerNorm no-affine + Linear(1024→6144/2048) from SiLU(time) → modulate), `dit_attention` (FC q/k/v/o + dit_rope on head 0 + host-precomputed block-diff additive mask; full attention, no KV cache — fixed seq). **Reuse** FC, tanh-GELU (`acti_func.h` ACT_TANH_GELU), LayerNorm.

**Host-side (in a `qwen25_omni_token2wav_dit.{h,cpp}` driver):**
- codec embed [8194,512] lookup + `repeat_interleave(2)` (like talker `codecEmbed`); `drop_code` uncond = row 0.
- timestep embedding (closed-form sinus 256 + 2 FCs) per RK4 stage.
- speaker `cond[192]` + ECAPA `condition[128]` **precomputed in Python** → `spk_cond_<name>.bin`; host concatenates input-embed parts in HF order [mel(80), condition(128), code(512), speaker(192)] → proj FC(912→1024).
- **RK4 ODE loop** (9 intervals × 4 stages) + **CFG** (2 host passes: conditioned + uncond, combine `g+(g−n)·0.5`) calling the compiled per-step DiT graph. Mirror the audio-encoder host loop (`qwen25_omni_audio_encoder.cpp:274-309`). Do NOT put RK4/CFG in the graph.

**Verify (Stage A):** feed HF-dumped `initial_state` + per-(t) inputs to the per-step graph; compare velocity (cond/uncond/combined) < ~1e-3 — surfaces head-0-RoPE + block-mask bugs. **(Stage B):** host RK4 over fixed noise; compare per-interval + final mel < ~1e-3.

## Sub-phase 2C — end-to-end

`qwen25_omni_token2wav.{h,cpp}` orchestrates Talker(`talkerDecode`→codec ids) → DiT(host RK4)→mel → BigVGAN→WAV. **Verify (Stage D):** HF full `Token2WavModel.forward` wav with matched noise → bit-near; free-RNG run → perceptually clean listen test.

## Riskiest unknowns — spike before/within 2A
1. **ConvTranspose1d weight layout** (PyTorch `[in,out,k]` vs nntrainer Conv2DTranspose) + `(k−stride)//2` padding, no `output_padding` — micro-test a hand-set kernel vs HF before the converter.
2. **Anti-aliased snake** Kaiser-sinc replicate-pad slicing parity (modeling.py:3146-3160) — prototype on one AMPBlock.
3. **(2B)** head-0-only RoPE + block-diff mask in `dit_attention`.
4. Verification noise: inject HF's `initial_state` rather than RNG-matching.

## Notes / reuse anchors
- No weight_norm in checkpoint → converter writes plain conv weights (assert if `weight_g` ever appears).
- Codec vocab **8194** (config 8193 +1). RoPE **head 0 only** (HF quirk, must replicate). conv_post no bias. BigVGAN resblocks **averaged** (÷3), not summed.
- Host loop + non-incremental graph precedents: `qwen25_omni_audio_encoder.cpp`. Custom-layer + converter precedents: `mrope_apply.{h,cpp}`, `audio_encoder_converter.py`. Talker host codec lookup: `qwen25_omni_talker_causallm.h:99`.
- Full HF spec dumped in this session's understanding workflow (synthesis archived in the task transcript).

## This session: implement Sub-phase 2A (BigVGAN) through Stage C verification.

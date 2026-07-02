# Qwen2.5-Omni Token2Wav — Phase 2A (BigVGAN) — CONFIRMED & CORRECTED PLAN

Status: ready for approval before coding. All facts below are ground-truthed against nntrainer core code, HF `modeling_qwen2_5_omni.py` (transformers 4.53), and the restored ground-truth dump in `/tmp/omni_t2w_dump`. This supersedes the BigVGAN (2A) portion of `purring-wondering-barto.md`.

---

## 1. Corrections to the approved plan

Each item is a place where the approved plan (`purring-wondering-barto.md`) or its "known facts" were wrong or stale. Consolidated and deduped across all 5 verification dimensions.

### C1 — Total upsample is **240×**, not 1440× (STALE ERROR, appears 4×)
- **Plan claim** (line 17, line 63 context): "1440× upsample → 24 kHz".
- **Reality**: `upsample_rates = [5,3,2,2,2,2]`, product = **240**. `mel_T=128 × 240 = 30720 = wav_len`.
- **Evidence**: `/tmp/omni_t2w_dump/meta.json` `upsample_rates=[5,3,2,2,2,2]`; `mel.npy[1,80,128]`; `wav.npy[30720]`; `ups5.npy[1,24,30720]`; `128*240=30720` exactly. Independently confirmed by all 5 dimensions.

### C2 — `Conv2DTransposeLayer` has a real WIDTH-axis bug — must be fixed before any upsample stage works (BLOCKING, new finding the plan did not know)
- **Plan claim** (line 24, line 56): just wrap `Conv2DTransposeLayer` with H=1 like `Conv1DLayer` wraps `Conv2DLayer` and lengths/values will be correct after confirming weight layout + `(k−stride)//2` padding.
- **Reality**: `conv2d_transpose_layer.cpp` computes the OUTPUT WIDTH using `eff_k_height` instead of `eff_k_width` in **three** places — `finalize` (line 274), `im2col_transpose` (line 154), `col2im_transpose` (line 74). For a Conv1D-transpose mapping (`kernel '1,k'` ⇒ k_height=1, k_width=k) this makes `out_w = (T−1)*s + 1 − 2p`, i.e. SHORT by `(k−1)`. The layer is internally self-consistent (im2col/col2im share the bug) so it does NOT crash — it silently produces the WRONG length and wrong values. The height-axis workaround (`kernel 'k,1'`) does NOT help (width becomes `k`, same bug).
- **Effect, as-written vs expected**: ups outputs `[630,1914,3837,7677,15357,30717]` vs expected `[640,1920,3840,7680,15360,30720]`. After fixing `eff_k_height→eff_k_width`, python sim reproduces `[640,1920,3840,7680,15360,30720]` exactly.
- **Evidence**: `conv2d_transpose_layer.cpp:73-74,153-154,273-274`; dump `ups{0..5}.npy` lengths.

### C3 — ConvTranspose1d weight needs a `(0,1)` transpose (plan was AMBIGUOUS; now CONFIRMED)
- **Plan claim** (line 24, line 32): conv kernels "NOT transposed, plain `[out,in,k]`" — written as a blanket rule, but the ups are ConvTranspose1d.
- **Reality**: plain `Conv1d` (conv_pre/conv_post + all resblock convs) is `[out,in,k]` and needs only an unsqueeze to `[out,in,1,k]` — NO transpose. But `ConvTranspose1d` (the 6 ups) weight is `[in,out,k]`; nntrainer `Conv2DTransposeLayer` requests `kernel_dim = [filter_size, in_channel, kh, kw] = [out,in,kh,kw]` (`conv2d_transpose_layer.cpp:241-242`), so the converter MUST transpose dims `(0,1)` then unsqueeze a `kh=1` axis.
- **Evidence**: safetensors `ups.0.0.weight=[1536,768,11]` (in=1536=conv_pre out ch, out=768=stage0 ch); `conv2d_transpose_layer.cpp:242`.

### C4 — AMPBlock internal structure undercounted (3 sub-blocks, 6 convs, 6 snakes — not 1 pair)
- **Plan claim** (line 17, line 28): AMPBlock = "dilated Conv1d dilations[1,3,5] + snakebeta" implying a single act→conv→act→conv pair.
- **Reality**: an AMPBlock runs the act→conv→act→conv→add pattern **three** times (chained, per kernel-index). It has `convs1.{0,1,2}` (dilations 1/3/5, all WITH bias), `convs2.{0,1,2}` (all dilation 1, all WITH bias), and `activations.{0..5}` (6 SnakeBeta-in-TorchActivation1d). `acts1 = activations[::2] = [0,2,4]`, `acts2 = activations[1::2] = [1,3,5]`. Per iter k: `residual=x; x=acts1[k](x); x=convs1[k](x); x=acts2[k](x); x=convs2[k](x); x=residual+x`.
- **Evidence**: HF `modeling_qwen2_5_omni.py:3290-3300`; checkpoint `resblocks.N.activations.{0..5}` + `convs1.{0,1,2}` + `convs2.{0,1,2}`.

### C5 — Resblocks operate at POST-upsample channels (1536 is consumed by ups0 immediately)
- **Plan claim** (line 17): channel chain "1536→768→…→24" read as if resblocks run at those listed channels including 1536.
- **Reality**: 1536 = conv_pre out = upsample_initial_channel = INPUT channel of ups0 only. After each ConvTranspose the channel halves. AMPBlocks at stage i run at `1536 // 2^(i+1)` = **768/384/192/96/48/24** — never 1536. conv_pre's 1536-ch output is consumed by ups0 before any resblock.
- **Evidence**: header `resblocks.0.convs1.0.weight=[768,768,3]` (stage0 @768); `resblocks.15-17 @24`; HF `:3337`.

### C6 — ups are ConvTranspose1d (deconv), and nntrainer has NO conv1d_transpose layer
- **Plan claim** (line 17, "upsample stages i: ConvTranspose1d" — correct here, but line 32's blanket "conv kernels NOT transposed" contradicts it).
- **Reality**: ups are `ConvTranspose1d` with layout `[in,out,k]` + bias. nntrainer ships only `conv2d_transpose_layer.{cpp,h}` — no conv1d_transpose. We must add a thin wrapper (see §3) AND fix C2.
- **Evidence**: HF `:3322`; header `ups.0.0.weight=[1536,768,11]`; `ls nntrainer/layers` shows no conv1d_transpose.

### C7 — `requestWeight` canonical signature (memory note was wrong)
- **Plan/memory claim**: `requestWeight(dim, ZEROS, NONE, 0, 0, name, /*trainable*/false)`.
- **Reality**: signature is `requestWeight(const TensorDim&, Initializer, WeightRegularizer, float reg_const, float decay, const std::string& name, bool trainable=true, bool is_virtual=false, unsigned int out_axis=3)` (`layer_context.h:212-216`). The two floats are `reg_const, decay` and the canonical values are `1.0f, 0.0f` (NOT `0,0`); there is NO trailing `,0,0` after `name`. Correct call (from `centroid_knn.cpp:64-66`):
  `context.requestWeight(dim, nntrainer::Initializer::ZEROS, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "alpha", false);`
- **Evidence**: `layer_context.h:212-216`; `centroid_knn.cpp:64-66`.

### C8 — snake_beta is NEVER applied bare — it is wrapped in TorchActivation1d anti-alias
- **Plan claim** (line 25): `snake_beta` listed as the elementwise activation; line 26 has `antialiased_snake` separately, but the wiring relationship was not explicit.
- **Reality**: SnakeBeta is always the INNER op of `TorchActivation1d` = `UpSample1d(×2) → SnakeBeta → DownSample1d(×2)`. Both resblock activations (HF `:3284`) and `activation_post` (`:3343-3345`) use the wrapper. The snake_beta layer is correct as pure-elementwise, but it operates on the **2× upsampled** signal, NOT the raw feature map. Therefore `snake_beta` CANNOT be validated against `activation_post.npy` (that dump is post-downsample).
- **Evidence**: HF `:3206-3211` (`upsample(h); act(h); downsample(h)`), `:3284`, `:3343-3345`.

### C9 — SnakeBeta `exp()` is UNCONDITIONAL; trained α/β are non-zero log-domain values
- **Plan claim** (line 17, line 63): "log-domain, init 0".
- **Reality**: init-at-construction is 0, but the TRAINED checkpoint values are non-zero (`activation_post.act.alpha[24]` range `[-1.85, 0.23]`, `beta[24]` range `[-1.94, 0.74]`). HF hard-codes `torch.exp()` on α and β inline (`:3085-3086`) — there is NO `alpha_logscale` flag in this impl. nntrainer must ALWAYS apply `exp` at runtime; do NOT pre-exp in the converter. `no_div_by_zero = 1e-9` is added to `exp(beta)` BEFORE the reciprocal (`:3075`), NOT to alpha. NO clamp anywhere in SnakeBeta.
- **Evidence**: HF `:3072-3091`; safetensors `activation_post.act.alpha/beta = F32`; loaded alpha first5 `[-0.574,-1.852,0.173,-0.777,0.127]`.

### C10 — conv_post has no bias; alpha/beta/all convs are F32 (not bf16); 449 bigvgan tensors
- **Plan claim** (line 63): "conv_post no bias" — CORRECT. (Restated for completeness.)
- **Reality additions**: `conv_post.bias` key is genuinely ABSENT from the checkpoint (do NOT emit zero bias — a spurious tail write corrupts nothing after it but is wrong). ALL 449 bigvgan tensors are F32 in a single shard `model-00003-of-00003.safetensors`. So the snake_beta layer can assume FP32 weights+input (matches MRoPEApplyLayer's FP32-only assumption).
- **Evidence**: header `conv_post.weight=[1,24,7]`, no `conv_post.bias`; all bigvgan tensors F32; HF `:3346-3348` `bias=False`.

### C11 — Kaiser-window `beta` ≠ 0.3 (it is derived = 4.6638); up and down filters are IDENTICAL for ratio=2
- **Plan claim** (line 17/26): cutoff/half_width "≈ .3" was ambiguous.
- **Reality**: `cutoff=0.5/ratio=0.25`, `half_width=0.6/ratio=0.3` (both correct), but the Kaiser-WINDOW `beta` is a DERIVED value: `attenuation = 2.285*(6−1)*pi*1.2 + 7.95 = 51.0212 (>50) → beta = 0.1102*(51.0212−8.7) = 4.6638`. For ratio=2 (the only ratio used) the UP and DOWN filters are the SAME 12-tap symmetric kernel (`np.allclose=True`). Filters are NOT in the checkpoint (`register_buffer(..., persistent=False)`, HF `:3151,:3181`; zero `filter` keys in the index) — they must be baked at finalize.
- **Evidence**: HF `:3094-3137,:3150,:3168-3169`; computed taps below; `np.allclose(up,down)=True`.

---

## 2. Confirmed BigVGAN architecture (verified end-to-end, exact dump shapes)

All shapes from `/tmp/omni_t2w_dump`. HF graph `modeling_qwen2_5_omni.py:3364-3379`. 449 tensors, all F32, prefix `token2wav.code2wav_bigvgan_model.`.

```
mel [1,80,128]  (DiT/CFM output, natural-log domain; mel.min=-11.62)
  │
  ▼  process_mel_spectrogram   (HOST, pure elementwise, fixed scalars — §3 step 0)
processed_mel [1,80,128]       (range [-1.0, 0.638]; 6.76% saturate to -1.0)
  │
  ▼  conv_pre = Conv1d(80→1536, k7, s1, pad3, +bias)     weight [1536,80,7]  bias [1536]
conv_pre_out [1,1536,128]
  │
  ▼  STAGE 0:  ups0 = ConvTranspose1d(1536→768, k11, s5, pad3)  +bias
  │            weight [1536,768,11]→nntr[768,1536,1,11]; bias[768]
  │            ups0 [1,768,640]      (T: 128→640 = ×5)
  │            mean_3( AMPBlock@768 k3, AMPBlock@768 k7, AMPBlock@768 k11 ) → ÷3
  │            stage0 [1,768,640]    (shape preserved; ≠ ups0)
  ▼
  ▼  STAGE 1:  ups1 = ConvTranspose1d(768→384, k7, s3, pad2)   +bias
  │            weight [768,384,7]→nntr[384,768,1,7]; bias[384]
  │            ups1 [1,384,1920]     (640→1920 = ×3)
  │            mean_3(AMPBlock@384 k3/k7/k11) → stage1 [1,384,1920]
  ▼
  ▼  STAGE 2:  ups2 ConvTranspose1d(384→192,k4,s2,pad1) → ups2 [1,192,3840]   (×2)
  │            mean_3(AMPBlock@192) → stage2 [1,192,3840]
  ▼
  ▼  STAGE 3:  ups3 ConvTranspose1d(192→96,k4,s2,pad1)  → ups3 [1,96,7680]    (×2)
  │            mean_3(AMPBlock@96)  → stage3 [1,96,7680]
  ▼
  ▼  STAGE 4:  ups4 ConvTranspose1d(96→48,k4,s2,pad1)   → ups4 [1,48,15360]   (×2)
  │            mean_3(AMPBlock@48)  → stage4 [1,48,15360]
  ▼
  ▼  STAGE 5:  ups5 ConvTranspose1d(48→24,k4,s2,pad1)   → ups5 [1,24,30720]   (×2)
  │            mean_3(AMPBlock@24)  → stage5 [1,24,30720]
  ▼
  ▼  activation_post = TorchActivation1d(SnakeBeta(24))   alpha[24], beta[24]
activation_post_out [1,24,30720]   (T preserved: up×2 → snake → down×2)
  │
  ▼  conv_post = Conv1d(24→1, k7, s1, pad3, NO bias)   weight [1,24,7]
  │
  ▼  clamp(-1, 1).squeeze()        (HOST)
wav [30720]   (range [-0.285, 0.375]); 30720 = 128 × 240
```

**AMPBlock (resblock j; ch = 1536//2^(stage+1); kernel = [3,7,11][j%3]):**
```
for k in 0..2:    # k indexes (convs1[k] dilation [1,3,5][k], convs2[k] dilation 1)
    residual = x
    x = acts[2k]  (x)          # TorchActivation1d(SnakeBeta(ch))  -> §3 antialiased_snake
    x = convs1[k](x)           # Conv1d(ch→ch, kernel, dilation=[1,3,5][k], pad=(kernel*d−d)/2, +bias)
    x = acts[2k+1](x)          # TorchActivation1d(SnakeBeta(ch))
    x = convs2[k](x)           # Conv1d(ch→ch, kernel, dilation=1, pad, +bias)
    x = residual + x
return x
```
- 18 resblocks = 6 stages × 3 kernels; resblock j: stage=j//3, kernel=[3,7,11][j%3].
- Stage output = `sum(3 AMPBlocks(ups_out)) / 3` (HF `:3370-3374`) — the 3 blocks are PARALLEL (each consumes the SAME ups_out), summed then divided by 3.
- Evidence: dump `conv_pre.npy[1,1536,128]`; `ups{i}.npy`/`stage{i}.npy` shapes above; `activation_post.npy[1,24,30720]`; `wav.npy[30720]`; header `resblocks.0.convs1.0.weight[768,768,3]`, `resblocks.17.convs1.0.weight[24,24,11]`.

---

## 3. Implementation order for this session (file-by-file, lowest → highest risk)

> **PREREQUISITE FIX (BLOCKING, do before any ups can work):** `nntrainer/layers/conv2d_transpose_layer.cpp` — change the WIDTH formula to use `eff_k_width` in all three spots: `finalize` line 274, `im2col_transpose` line 154, `col2im_transpose` line 74: `width = (in_width − 1) * mstride[1] + eff_k_width;` (currently `eff_k_height`). Rebuild. Then run existing `test/unittest` conv2d_transpose tests for regression (square-kernel `kh==kw` cases unaffected; only non-square kernels change behavior). This is required for C2.

### Step 0 — host `process_mel_spectrogram` (no layer; in the bigvgan model driver)
Pure elementwise, fixed scalars. Run on mel `[80,T]` fp32 before feeding conv_pre. Literal HF-faithful form (recommend for first impl):
```cpp
const double LN10 = std::log(10.0);                          // 2.302585092994046
const float  MIN_LEVEL = (float)std::exp(-115.0/20.0*LN10);  // 1.7782794e-06
// per element x:
float amp = std::exp(x);
float cl  = amp < MIN_LEVEL ? MIN_LEVEL : amp;
float db  = 20.0f*std::log10(cl) - 20.0f;                    // amplitude_to_db(...,-115) - 20
float n   = 2.0f*((db + 115.0f)/115.0f) - 1.0f;              // normalize(max=1,min_db=-115)
float out = n < -1.0f ? -1.0f : (n > 1.0f ? 1.0f : n);
```
Verified vs `processed_mel.npy` to fp32 max-abs err **2.384e-7** (≪1e-4). Folded fast form (`a=0.151058950227218, b=0.6521739130434783`, mel clamp at `-13.239864284715765`) matches to 1.6e-7 — defer to profiling. Evidence: HF `:3350-3362`; `/tmp/omni_t2w_dump/mel.npy`→`processed_mel.npy`.

### Step 1 — `snake_beta` (LOWEST risk; do FIRST) — full spec in §4
Files: `Applications/CausalLM/layers/snake_beta.{h,cpp}`. Pure elementwise per-channel; mirrors `mrope_apply.{h,cpp}`. Two trainable=false weights `alpha[C]`, `beta[C]` (F32). Formula `y = x + (1/(exp(beta)+1e-9)) * sin(x*exp(alpha))^2`. **Do NOT validate against `activation_post.npy`** (that is post-downsample — C8); validate with a hand-fed `[1,C,T]` synthetic vs a one-line numpy reference. Cheap sanity: `x=0 → y=0` everywhere.

### Step 2 — `conv1d_transpose` (after the C2 layer fix)
Files: `Applications/CausalLM/layers/conv1d_transpose.{h,cpp}`. Copy the `conv1d_layer.cpp:45-75` wrapper pattern but delegate to `Conv2DTransposeLayer`.
- finalize: assert input height==1; set conv2d-transpose props `filters=out_ch`, `kernel_size='1,'+k`, `stride='1,'+s`, `dilation='1,1'`, `padding='0,0,'+p+','+p` where `p=(k−s)//2`. NO `output_padding` (HF uses none; `(k−s)` is even for all stages: 11−5=6, 7−3=4, 4−2=2 → p=3,2,1).
- I/O: input `[B,in_ch,1,T]` → output `[B,out_ch,1,T_out]`, `T_out=(T−1)*s + k − 2p` (correct AFTER the eff_k_width fix).
- forwarding internal path: `Conv2DTransposeLayer` reshapes filter to `[filter_size, in*kh*kw]` and does `filter_kernel.dot(result, out, false, true)` (`conv2d_transpose_layer.cpp:352-372`); im2col column order is `(in,kh,kw)` row-major (`:99-101,:181-183`) which matches `[out,in,1,k]` row-major flatten — no extra reorder.
- Bias: `[1,out,1,1]` broadcast over `[B,out,1,T]`.
- **Per-stage params:**
  | stage | filters | kernel | stride | padding | dilation |
  |---|---|---|---|---|---|
  | 0 | 768 | 1,11 | 1,5 | 0,0,3,3 | 1,1 |
  | 1 | 384 | 1,7  | 1,3 | 0,0,2,2 | 1,1 |
  | 2 | 192 | 1,4  | 1,2 | 0,0,1,1 | 1,1 |
  | 3 | 96  | 1,4  | 1,2 | 0,0,1,1 | 1,1 |
  | 4 | 48  | 1,4  | 1,2 | 0,0,1,1 | 1,1 |
  | 5 | 24  | 1,4  | 1,2 | 0,0,1,1 | 1,1 |
- **Converter transpose (C3):** PyTorch convT weight `W[in,out,k]` → nntrainer `[out,in,1,k]` via `np.ascontiguousarray(W.transpose(1,0,2))[:,:,None,:]`. Bias loads as-is into `[1,out,1,1]`.
- **Micro-spike FIRST:** feed `conv_pre.npy` through ups0 (transposed weight + bias) and compare to `ups0.npy` (atol ~1e-4) — confirms BOTH the length fix and transpose direction. If garbage/transposed, flip the converter `(0,1)` axis.

### Step 3 — `antialiased_snake` (HIGHEST risk — prototype on one AMPBlock vs HF first)
Files: `Applications/CausalLM/layers/antialiased_snake.{h,cpp}`. Wraps `snake_beta` core in fixed Kaiser-sinc ×2 up/down (depthwise, replicate-pad). One 12-tap symmetric filter (same for up and down):
```
taps = [0.00202896, 0.00938947, -0.02554346, -0.05765738, 0.12857258, 0.4432098,
        0.4432098, 0.12857258, -0.05765738, -0.02554346, 0.00938947, 0.00202896]  (fp32, sum=1.0)
```
Per-instance ops (`x[B,C,T]`, α[C], β[C], ratio=2, k=12, depthwise groups=C, no bias):
1. **UPSAMPLE (T→2T):** replicate-pad `(5,5)` → conv_transpose1d(stride=2, groups=C) → **multiply ALL outputs by ratio=2** → slice `[15 : len−15]` → `[B,C,2T]`. (HF `:3144-3162`)
2. **SNAKEBETA** on the 2T signal: `a=exp(α)`, `b=exp(β)`; `h = up + (1/(b+1e-9))*sin(up*a)^2`.
3. **DOWNSAMPLE (2T→T):** replicate-pad `(5,6)` → conv1d(stride=2, groups=C) → NO ratio multiply, NO slice → `[B,C,T]`. (HF `:3176-3186`)
- Bake the filter at finalize as depthwise convT weight `[C,1,12]` and depthwise conv weight `[C,1,12]` (broadcast single tap vector across C). Numpy reference matches PyTorch to **8.25e-7** on random `[1,4,37]`.
- **Validate FIRST against the `stage5 → activation_post` dump pair** (`stage5.npy[1,24,30720]` → `activation_post.npy[1,24,30720]`): that single pair exercises the full up→snake→down for the `activation_post` instance end-to-end. Do this before wiring resblocks.
- Evidence: HF `:3094-3211`; dump pair above.

### Step 4 — converter `token2wav_bigvgan_converter.py`
File: `Applications/CausalLM/res/qwen2.5-omni/token2wav_bigvgan_converter.py`. Reuse `ShardedSafetensors` + `resolve_model_dir` from `weight_converter.py` (import like `audio_encoder_converter.py:42`). `prefix='token2wav.code2wav_bigvgan_model.'`. Emit helpers: `save_conv` (`[out,in,k]→[out,in,1,k]`, NO transpose), `save_convT` (`[in,out,k]→[out,in,1,k]`, transpose `(0,1)`), `save_snake(name)` (α then β, `[C]`, raw log-domain, NO transpose, NO pre-exp). All FP32.

**DFS-from-output write order (mirror HF forward; align to nntrainer load order per the gate-before-up precedent `weight_converter.py:27-32`):**
1. `conv_pre.weight`, `conv_pre.bias`
2. for stage i in 0..5:
   - `ups.{i}.0.weight` (transpose), `ups.{i}.0.bias`
   - for block b in 0..2 (resblock = i*3+b), in AMPBlock internal order: per iter k in 0..2 → `activations.{2k}.act.alpha`,`.beta`; `convs1.{k}.weight`,`.bias`; `activations.{2k+1}.act.alpha`,`.beta`; `convs2.{k}.weight`,`.bias`
3. `activation_post.act.alpha`, `.beta`
4. `conv_post.weight` (NO bias — do NOT emit)

> **Branch-order caveat:** the 3 parallel AMPBlocks per stage are summed→÷3; the exact DFS visit order through the addition node must match the nntrainer graph wiring. **Build the graph, dump the actual weight-request order, align the converter** (same spike that bit the MLP gate/up fix). Do not assume block0,1,2.

### Step 5 — `qwen25_omni_bigvgan.{h,cpp}` model + meson + registration
Files: `Applications/CausalLM/models/qwen25_omni/qwen25_omni_bigvgan.{h,cpp}`; edit `models/qwen25_omni/meson.build` and `layers/meson.build`; register `snake_beta`, `conv1d_transpose`, `antialiased_snake` where `mrope_apply` is registered.
- Input `[BATCH,80,1,128]` (channel-major `[B,C,1,T]`, audio_encoder idiom). Host `process_mel` (step 0) before conv_pre.
- conv_pre: core `conv1d` filters=1536,k=7,pad=3,+bias. Per stage: `conv1d_transpose` → 3 AMPBlock subgraphs → addition → constant ×(1/3). AMPBlock chains `antialiased_snake → conv1d(dilation) → antialiased_snake → conv1d → addition(residual)` ×3. activation_post `antialiased_snake`. conv_post `conv1d` filters=1,k=7,pad=3,**disable_bias=true**. Final clamp host-side, write 24 kHz WAV.
- Run idiom: compile once, non-incremental (`incremental_inference(BATCH,{mel},{},T,0,T,false)`) like `qwen25_omni_audio_encoder.cpp:215-244`.
- conv1d dilation maps to conv2d `'1,d'`; verify Padding1D matches HF `_get_padding=(k*d−d)/2` for dilations 1/3/5, kernels 3/7/11.

### Step 6 — Stage C verify
File: extend the HF dumper / reuse `/tmp/omni_t2w_dump`. Feed dumped `mel`, compare layer-by-layer: `conv_pre` (catches plain-conv layout) → `ups0` (catches convT layout + length fix) → one AMPBlock (catches snake/anti-alias) → `activation_post` → full `wav` max-abs err < ~1e-3 (FP32). Listen test.

---

## 4. `snake_beta` — full code-ready spec (implemented first)

Namespace `causallm`, mirror `mrope_apply.{h,cpp}`. FP32 weights + input (C10).

### Header (`snake_beta.h`)
```cpp
class SnakeBetaLayer final : public nntrainer::Layer {
public:
  SnakeBetaLayer() : Layer() {}
  ~SnakeBetaLayer() {}
  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to, bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override {}
  bool supportBackwarding() const override { return false; }
  void exportTo(nntrainer::Exporter &, const ml::train::ExportMethods &) const override {}
  const std::string getType() const override { return SnakeBetaLayer::type; }
  void setProperty(const std::vector<std::string> &values) override {}
  inline static const std::string type = "snake_beta";
private:
  std::array<unsigned int, 2> wt_idx; // 0 = alpha, 1 = beta
};
```

### finalize() — exact `requestWeight` (C7-verified against `layer_context.h:212` + `centroid_knn.cpp:64-66`)
```cpp
const auto &in = context.getInputDimensions()[0];
unsigned int C = /* channel axis of `in` — see Open Risk R1 */;
nntrainer::TensorDim wdim({1, 1, 1, C}); // 1-D length-C; axis must match converter emit shape [C]
wt_idx[0] = context.requestWeight(wdim, nntrainer::Initializer::ZEROS,
              nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "alpha", false);
wt_idx[1] = context.requestWeight(wdim, nntrainer::Initializer::ZEROS,
              nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "beta",  false);
context.setOutputDimensions({in});
```
(`ZEROS` is only a placeholder; the converter overwrites with trained log-domain values — C9.)

### forwarding() — math (precompute per channel, broadcast over T)
```cpp
Tensor &x = context.getInput(0);  Tensor &out = context.getOutput(0);
const float *ap = context.getWeight(wt_idx[0]).getData<float>(); // [C]
const float *bp = context.getWeight(wt_idx[1]).getData<float>(); // [C]
for (b in batch)
  for (c in [0,C)) {
    float a  = expf(ap[c]);
    float bb = 1.0f / (expf(bp[c]) + 1e-9f);     // 1e-9 added to exp(beta), NOT alpha
    for (t in [0,T)) {
      float v = x(b,c,t);
      float s = sinf(v * a);
      out(b,c,t) = v + bb * s * s;               // y = x + (1/(exp(b)+1e-9)) * sin(x*exp(a))^2
    }
  }
```
Verified isolated: ch0 α_raw=−0.5742(exp 0.5631), β_raw=−0.8828(exp 0.4136): x=0→0, x=0.6667→0.9917, x=2.0→3.9706. Fixed point `x=0→y=0` is a cheap runtime assertion.

### incremental_forwarding()
Snake is positionwise/stateless → thin wrapper that runs the same elementwise math over the whole tensor (BigVGAN is one-shot decode, not autoregressive). Full forwarding is sufficient.

### Registration boilerplate (mirror)
Find where the `"mrope_apply"` type string is registered for the CausalLM app (the `createLayer`/`registerCustomLayer` factory list) and add a parallel entry for `"snake_beta"`. Add `snake_beta.{cpp,h}` to `Applications/CausalLM/layers/meson.build`. The same hook accepts multiple custom layers (no single-custom-layer assumption — verify).

### Converter weight name/order (C9)
Per `TorchActivation1d.act` instance emit TWO weights **in order [alpha, beta]**, each `[C]` F32, names matching the requested `"alpha"`/`"beta"`. Source HF keys:
- `token2wav.code2wav_bigvgan_model.resblocks.{i}.activations.{j}.act.{alpha,beta}` (i∈0..17, j∈0..5)
- `token2wav.code2wav_bigvgan_model.activation_post.act.{alpha,beta}`
- 218 α/β tensors total (18×6×2 + 2). NO transpose (1-D). Load RAW log-domain — do NOT pre-exp (exp is runtime, C9).

---

## 5. Open runtime risks (consolidated, priority order)

**P0 — `Conv2DTransposeLayer` width-bug fix + regression (C2, BLOCKING).** Apply `eff_k_height→eff_k_width` at `conv2d_transpose_layer.cpp:74,154,274`, rebuild, run `test/unittest` conv2d_transpose tests (square-kernel cases unaffected; non-square change). Without this every ups stage is wrong length AND value.

**P0 — ups0 micro-spike (convT layout + transpose direction).** Feed `conv_pre.npy` → ups0 (transposed weight `[768,1536,1,11]` + bias) → compare `ups0.npy` (atol ~1e-4). Validates the length fix AND the converter `(0,1)` transpose. If garbage/transposed → flip transpose axis. Also confirm the `(in,kh,kw)` im2col column order matches the `[out,in,1,k]` row-major flatten (value mismatch with correct length ⇒ kh/kw or in-order issue), and that bias `[1,out,1,1]` broadcasts over the H=1 output.

**P0 — antialiased_snake on the `stage5→activation_post` dump pair.** Most error-prone op (ratio=2 scalar multiply + asymmetric `[15:-15]` slice + replicate pad). Validate the full up→snake→down end-to-end against that pair FIRST, before wiring resblocks.

**P1 — `[B,C,T]` 4D axis mapping (snake_beta + antialias).** Pin whether the conv producer emits `[B,C,1,T]` (channel=dim1) or `[B,1,C,T]` (channel=height) so finalize requests the `[C]` weight on the matching broadcast axis and forwarding indexes channel correctly. Micro-spike: dump the conv/convT output dim feeding a snake instance.

**P1 — replicate (edge) padding primitive.** Confirm nntrainer Conv1d/ConvTranspose1d input pad is replicate/edge, NOT zero (up uses (5,5), down (5,6)). A zero-pad fallback silently corrupts edges — compare edge samples in the spike.

**P1 — depthwise (groups=C) support.** Verify nntrainer conv1d/convT1d honor `groups=channels` with a `[C,1,12]` weight. If not native, implement the Kaiser FIR as a per-channel scalar op in the custom layer.

**P1 — AMPBlock 3-branch DFS load order + ÷3 (C4).** Build graph, dump actual weight-request order, align converter (gate/up precedent `weight_converter.py:27-32`). Confirm `addition + constant ×(1/3)` reproduces mean WITHOUT introducing a loadable weight that shifts DFS order. nntrainer has no built-in mean-of-N layer.

**P2 — conv1d dilation parity.** Verify conv1d dilation→conv2d `'1,d'` reproduces PyTorch dilated conv, and Padding1D matches HF `_get_padding=(k*d−d)/2` for dilations 1/3/5 × kernels 3/7/11.

**P2 — conv_post no-bias path.** Ensure conv1d `disable_bias` path exists and the converter emits NO bias bytes (C10).

**P2 — snake α/β index mapping.** Confirm `wt_idx[0]=alpha, wt_idx[1]=beta` matches the converter emit order (DFS-from-output load per MEMORY). A swap still runs but produces wrong audio — assert via known-input micro-spike.

**P3 — fp16/bf16 drift (deferred; dump is fp32).** snake `exp()`/`sin()`, antialias, and process_mel are verified exact in fp32. If any stage runs fp16, micro-check numeric drift before trusting — process_mel scale (~0.151) and mel range (~[−11.6,−0.1]) are within fp16, but near the −1.0 clamp boundary a few elements may shift.

**P3 — upstream mel units (cross-phase prerequisite).** process_mel's `exp()` assumes natural-log mel (mel.min=−11.6). When DiT (2B) is wired, confirm it emits mel in the SAME log scale; a different normalization breaks the exp assumption.

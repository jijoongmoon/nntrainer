# Qwen2.5-Omni Token2Wav DiT (Phase 2B) — CONFIRMED Plan

DiT produces the mel `[1,80,128]` that the already-validated Phase 2A BigVGAN consumes. The DiT runs as a **host-driven RK4 loop** over a **single compiled, non-incremental nntrainer graph** (one ODE-function evaluation per call). All facts below are cross-verified against HF source (`modeling_qwen2_5_omni.py` / `configuration_qwen2_5_omni.py`), the checkpoint (`model-00003-of-00003.safetensors`, prefix `token2wav.code2wav_dit_model.`, 360 F32 tensors), and the restored dump (`/tmp/omni_t2w_dump/`).

---

## 1. Corrections to the prior plan (deduped, with evidence)

| # | Prior-plan claim | Reality | Evidence |
|---|---|---|---|
| **C1** | "ECAPA condition[128] **precomputed**; cond[192] is the speaker vector run through an encoder" | **Roles are crossed.** The 128-d slot is `spk_encoder(ref_mel)` (ECAPA-TDNN) computed **in-graph every call**; the 192-d slot is the **precomputed raw** `cond.npy[1,192]` (no encoder, just broadcast). There is a confusing **arg-name swap** between `sample()` and `forward()`: `sample`'s `conditioning_vector(==cond192)` is passed as `forward`'s `speaker_embedding`; `sample`'s `reference_mel(==ref_mel)` is passed as `forward`'s `condition_vector`. | `modeling:2787` `spk_encoder=ECAPA_TimeDelayNet`; `modeling:2807` `condition_vector=self.spk_encoder(condition_vector)`; `sample` `:3557-3558`; dump `cond.npy[1,192]`, `ref_mel.npy[1,400,80]`; ckpt `input_embed.spk_encoder.fc.weight[128,1536,1]`, `blocks.0.conv.weight[256,80,5]` |
| **C2** | "block-diff **additive** mask (single uniform mask)" | Mask is a **BOOLEAN allow-mask** (`True`=keep) that **varies per layer**: `mask = (block_diff >= -look_backward) & (block_diff <= look_ahead)`. block_size=24; `look_ahead_layers=[10]`, `look_backward_layers=[0,20]`. So only **3 layers** widen by one block (L0,L20 backward; L10 ahead); the **other 19 are strictly block-diagonal** (`block_diff==0`, intra-24-frame-block only). nntrainer must convert to additive (0 keep / −inf drop) for a softmax-add kernel. | `modeling:3039-3040` mask formula; `_create_block_diff:3476-3484`; `configuration:853-855` defaults |
| **C3** | "AdaLN-Zero(time→**6 gates**)" | The 6 chunks are **`[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`** (2 shifts, 2 scales, 2 gates) — **shift-before-scale**. Only `gate_*` are multiplicative residual gates. The **same** per-block `attn_norm.linear[6144,1024]` produces both the attn-norm modulation **and** the mlp-norm modulation; the MLP pre-norm is a **separate no-affine LayerNorm** (`ff_norm`) consuming `scale_mlp/shift_mlp`. **Two no-affine LayerNorms per block.** | `modeling:2842` chunk-6; `modeling:3026` `ff_norm=nn.LayerNorm(...,elementwise_affine=False,eps=1e-6)`; `:3046` |
| **C4** | "AdaLN-Final + proj_out(1024→80)" | Correct, **but** `norm_out` chunk order is **`[scale, shift]` (SCALE first)** — opposite to per-block's shift-first. `norm_out.linear[2048,1024]` (=dim×2, NOT ×6). Easy to swap silently. | `modeling:2861` `scale,shift=chunk(emb,2)` vs `:2842` shift-first; ckpt `norm_out.linear.weight[2048,1024]`, `proj_out.weight[80,1024]` |
| **C5** | "RoPE on head 0 only — reuse `mrope_apply`" | Head-0-only **confirmed**, but the rotate is **ADJACENT-PAIR** (`rotate_half_codec`: `reshape(...,-1,2)`, `(x0,x1)->(-x1,x0)`) with **INTERLEAVED** cos/sin `[f0,f0,f1,f1,...]` (`stack((freqs,freqs)).reshape`). This is **NOT** the half-split rotate used by `mrope_apply` (`mrope_apply.cpp:65` `i<half? -x[i+half]:x[i-half]`). **`mrope_apply` cannot be reused** — a new `dit_rope` is required. | `modeling:2965` head-0 only; `:2910-2916` adjacent-pair; `:2469-2470` interleaved; ckpt `rotary_embed.inv_freq[32]` |
| **C6** | "CFG = **two** forward passes per stage" | It is **ONE batched forward of batch=2** per ODE call (`torch.cat([h,h],dim=0)` inside `DiTInputEmbedding`), chunked into guided/null. So **36 ODE calls = 36 batched DiT forwards** (= 72 single-row equivalents). For nntrainer you may run **two batch-1 forwards** instead (numerically identical) — **recommended** given the Talker batched-prefill MHA bug. | `modeling:2800` cat; `:3574-3575` chunk+combine |
| **C7** | "config carries hidden/depth/heads/mask params (read from `config.json`)" | `config.json` `dit_config` stores **aliases** `dim/depth/heads` that `Qwen2_5OmniDiTConfig.__init__` **does not accept** (swallowed as `**kwargs`, ignored). Real values come from **class defaults**. `block_size/look_ahead_layers/look_backward_layers` and `rope_theta` are **absent from JSON** — **hard-code** `24 / [10] / [0,20] / 10000`. | `config.json` has `dim/depth/heads` only; `configuration:843-867` defaults; no `attribute_map` in this config |
| **C8** | "ff_mult 2" / "uncond branch unspecified" | ff_mult=2 → inner **2048** confirmed (NOT the nn.GELU-default 4). MLP = `[Linear(1024→2048), GELU(tanh), Dropout(0.1), Linear(2048→1024)]`, ckpt keys `ff.ff.0` / `ff.ff.3` (index 2 Dropout = no-op at inference, no weight). **The CFG null branch needs a SECOND code embed**: `drop_code=True` zeros the ids → all map to **codec_embed row 0** (NOT a zero vector), then repeat_interleave. Host must build **both** conditional and unconditional code embeddings. | `modeling:2871` `inner=int(dim*mult)`; ckpt `ff.ff.0[2048,1024]`,`ff.ff.3[1024,2048]`; `:2821-2822` `drop_code: code=zeros_like(code)`; `:3503` |

**Items the prior plan got RIGHT** (stated to remove residual uncertainty): input-embed concat **order** `[mel80, cond128, code512, speaker192]→proj(912→1024)`; codec table `[8194,512]`, repeats=2; `proj_out(1024→80)`; sampler 36 DiT calls; CFG combine `guided+(guided−null)*0.5 = 1.5·guided − 0.5·null`; noise `[1,30000,80]` sliced to `[:, :128]`.

---

## 2. Confirmed DiT architecture

### Dimensions (final, verified)
`hidden=1024`, `depth=22`, `heads=16`, `head_dim=64` (16×64=1024=inner_dim ⇒ q/k/v are **square** [1024,1024]), `ff_mult=2 → ff_inner=2048`, `mel_dim=80`, `emb_dim(codec)=512`, `enc_dim(ECAPA out)=128`, `enc_emb_dim(speaker)=192`, `codec_vocab=8194 (=num_embeds 8193 +1)`, `repeats=2`, `time_freq=256`, `block_size=24`, `seq=128 = num_codes(64)×repeats(2)`.

### Tensor inventory (360 total, all F32)
- **Non-block (52):** `input_embed.proj` (W[1024,912]+b[1024]); **40 ECAPA `input_embed.spk_encoder.*`** (separate sub-spec, §4); `text_embed.codec_embed.weight[8194,512]`; `time_embed.time_mlp.0` (W[1024,256]+b) & `.2` (W[1024,1024]+b); `rotary_embed.inv_freq[32]`; `norm_out.linear` (W[2048,1024]+b); `proj_out` (W[80,1024]+b).
- **Per block (14 × 22 = 308):** `attn_norm.linear` W[6144,1024]+b[6144]; `attn.to_q/to_k/to_v` W[1024,1024]+b **each** (bias present); `attn.to_out.0` W[1024,1024]+b; `ff.ff.0` W[2048,1024]+b; `ff.ff.3` W[1024,2048]+b. (`ff_norm` and the AdaLN-internal LN have **no weights** — no-affine.)

### Per-block op sequence (`modeling:3029-3050`)
```
emb = attn_norm.linear( SiLU(time_emb) )            # [B,6144]
shift_msa,scale_msa,gate_msa, shift_mlp,scale_mlp,gate_mlp = chunk(emb, 6)   # SHIFT first
n   = LN_noaffine(h) * (1+scale_msa) + shift_msa     # eps=1e-6, no gamma/beta
a   = attn(n)                                        # see below
h   = h + gate_msa * a                               # gated residual
n2  = ff_norm(h) * (1+scale_mlp) + shift_mlp          # separate no-affine LN, reuses scale/shift_mlp
f   = ff(n2)  = Linear(1024→2048) → tanh-GELU → Linear(2048→1024)
h   = h + gate_mlp * f
```

### Attention (per block)
`q,k,v = to_q/to_k/to_v(n)` → reshape `[B,seq,16,64]` → **apply ADJACENT-PAIR RoPE to HEAD 0 ONLY** (heads 1..15 untouched, `modeling:2965`) → scaled-dot-product (scale `1/√64`), non-causal, with the **boolean block mask** (same for all 16 heads) → reshape → `to_out.0`.
- RoPE freqs: `inv_freq = 1/(10000^(arange(0,64,2)/64))` [32]; `freqs=stack((freqs,freqs),-1).reshape→[seq,64]` (interleaved-duplicate `[f0,f0,f1,f1,…]`). Rotate: `(x0,x1)→(-x1,x0)` per adjacent pair (`modeling:2910-2916`).
- **Mask** (`modeling:3039-3040`, C2): `block_id[p]=p//24`; keep iff `(block_id[j]-block_id[i]) ∈ [-look_backward, look_ahead]`. Per layer: **L10** ahead=1, **L0 & L20** backward=1, **all other 19** ahead=back=0 (strict block-diagonal). For seq=128, blocks sizes `[24,24,24,24,24,8]`.

### Time embedding (`modeling:3004-3015`)
`SinusPositionEmbedding(256, scale=1000)` → `Linear(256→1024)` → SiLU → `Linear(1024→1024)` → `[B,1024]`. Sinusoid: `half=128; emb=exp(arange(128)·−ln(10000)/127); out=cat(sin(1000·t·emb), cos(1000·t·emb))`. This single vector feeds **every** AdaLN-Zero block **and** AdaLN-Final.

### Final
`norm_out` (AdaLN-Final, **scale-first**): `LN_noaffine(h)·(1+scale)+shift` → `proj_out(1024→80)` → velocity `[B,seq,80]`. Permute to `[1,80,128]` **only at the very end of `sample()`** (`modeling:3589`). Dump `dit_mel.npy[1,80,128]` (min −11.616, max −0.095, mean −3.989, std 2.376) is the BigVGAN input and the Stage-B target.

---

## 3. Confirmed sampler (host pseudocode)

**Integrator = classic RK4 (3/8-rule), 9 intervals × 4 evals = 36 DiT forwards.** NOT Euler/midpoint.

- **Noise:** `randn([1,30000,80])` → slice `[:, :maximum_duration]`, `maximum_duration=num_codes·repeats=128` (`modeling:3544-3546`). Solver works in `[1,128,80]`. **For Stage-B bit-match, LOAD `initial_state_full.npy[:, :128]` — do NOT call C++ RNG.**
- **Time schedule** (`modeling:3578-3583`): `t = linspace(0,1,10)`; `t += −1.0·(cos(π/2·t) − 1 + t)` ⇒ `[0, .01519, .06031, .13397, .23396, .35721, .5, .65798, .82635, 1.0]` (9 intervals, denser near 0; endpoints stay exactly 0 and 1).
- **CFG:** `guidance_scale=0.5` ⇒ `v = guided + (guided−null)·0.5 = 1.5·guided − 0.5·null` (`modeling:3574-3575`).

```cpp
const float c13 = 1.0f/3.0f, c23 = 2.0f/3.0f, gscale = 0.5f;
float y[128*80] = initial_state;          // [T,80], T=128
for (int i = 0; i < 9; ++i) {
    float ts = t[i], te = t[i+1], dt = te - ts;
    auto k1 = ode(ts,              y);
    auto k2 = ode(ts + dt*c13,     y + dt*c13*k1);
    auto k3 = ode(ts + dt*c23,     y + dt*(k2 - c13*k1));
    auto k4 = ode(te,              y + dt*(k1 - k2 + k3));
    y = y + (k1 + 3.0f*(k2+k3) + k4) * (dt/8.0f);   // delta; modeling:3395,3425
}
mel = permute(y, [80,128]);                // feed to Qwen25OmniBigVGAN::vocode

ode(ts, y):                                 // one ODE eval
    v_cond = DiT_forward(y, ts, ecapa128_pos, code_embed,        speaker192);
    v_null = DiT_forward(y, ts, ecapa128_neg, code_embed_uncond, zeros192);
    return 1.5f*v_cond - 0.5f*v_null;        // elementwise [T,80]
```
(HF stores a trajectory + linear-interp, but the integration grid == output grid, so `solution[-1]==y`; only the last `y` is needed.)

---

## 4. Conditioning & host precompute

All conditioning except `(y, t)` is **constant across all 36 calls** → compute **once** before the RK4 loop.

1. **Codec embed (host gather + repeat_interleave):** gather 64 rows from `codec_embed[8194,512]` by `codes` (mirror Talker host lookup, `qwen25_omni_talker_causallm.h:99-101`), then `repeat_interleave(repeats=2, dim=1)` → `[128,512]` where each id duplicates **adjacently** `[c0,c0,c1,c1,…]` (`modeling:2825`, **not** tiling). Build **`code_embed_uncond`** the same way from **row 0** (drop_code zeros ids; `modeling:2821-2822`).
2. **Speaker 192 (precomputed, raw):** `cond.npy[1,192]` → broadcast to `[128,192]`. This is `forward`'s `speaker_embedding`, concatenated **raw** (no encoder), placed **last**.
3. **ECAPA 128 (in-HF-graph; host-precompute for nntrainer):** `spk_encoder(ref_mel[1,400,80])→[1,128]` broadcast to `[128,128]`. **Two versions needed for CFG**: `ecapa128_pos = ECAPA(ref_mel)` and **`ecapa128_neg = ECAPA(zeros_like(ref_mel))`** (the null row zeros ref_mel **before** the encoder, so it is **not** a zero vector — `modeling:2802,2807`). Verify `ECAPA(zeros)` numerically. ECAPA is its own sub-network (TDNN/Res2Net/SE/AttentiveStatsPool/Conv1d-fc, reflect-pad, masked-softmax) — for bring-up **precompute in Python and inject**; port last (§5, risk R3).
4. **Input-embed concat (graph FC):** `cat([y_mel(80) | ecapa128 | code_embed(512) | speaker192], dim=-1) = [.,912]` → `proj(912→1024)`. **Column layout follows the FORWARD order** (`modeling:2808`), i.e. `[mel 0:80][ecapa 80:208][code 208:720][speaker 720:912]` — NOT the constructor sum-order at `:2784`.

---

## 5. Implementation order (lowest → highest risk)

> Pattern templates (all confirmed present): converter `res/qwen2.5-omni/token2wav_bigvgan_converter.py`, FC transpose convention `weight_converter.py:158-164` / `audio_encoder_converter.py:133` (`w.T → [in,out]`, bias raw, embedding raw); model class `models/qwen25_omni/qwen25_omni_bigvgan.cpp`; host-loop/non-incremental idiom `models/qwen25_omni/qwen25_omni_audio_encoder.cpp`; layer skeletons `Applications/CausalLM/layers/{vision_attention,mrope_apply,snake_beta}.*`; dump spike `res/qwen2.5-omni/dump_token2wav_refs.py`, `spike_bigvgan{,_prep}.{cpp,py}`.

### Step 1 — Converter `res/qwen2.5-omni/token2wav_dit_converter.py` (low risk)
PREFIX `token2wav.code2wav_dit_model.`; all F32 (no dtype switch). FC weights transposed `w.T=[in,out]`, biases raw, `codec_embed[8194,512]` **raw** (no transpose). **Emit in DFS-from-output order = input-side first, output last** (the exact class of bug in `weight-bin-load-order-dfs` memory; lock converter order and graph wiring together):
1. ECAPA `spk_encoder.*` (Conv1d sub-tree in forward order; `[out,in,k]→[out,in,1,k]`, +bias) — see token2wav_bigvgan_converter conv handling.
2. `text_embed.codec_embed.weight[8194,512]` raw.
3. `input_embed.proj` (FC, →[912,1024]+b).
4. `time_embed.time_mlp.0` (FC →[256,1024]+b), `.2` (FC →[1024,1024]+b). (`.1`=SiLU, no weight.)
5. `rotary_embed.inv_freq[32]` — **optional** (recompute on host preferred).
6. blocks 0..21, each: `attn_norm.linear`(→[1024,6144]+b), `to_q,to_k,to_v`(→[1024,1024]+b each), `to_out.0`(+b), `ff.ff.0`(→[1024,2048]+b), `ff.ff.3`(→[2048,1024]+b). (Order MUST match graph DFS load order.)
7. `norm_out.linear`(→[1024,2048]+b), `proj_out`(→[1024,80]+b).

`assert tensors_written == 360`.

### Step 2 — `layers/dit_rope.{h,cpp}` (low-med risk)
Inputs `0=x[B,1,seq,1024]`, `1=cos[seq,64]`, `2=sin[seq,64]` (host-filled, interleaved). **No weights.** Apply **adjacent-pair** rotation `(x0,x1)→(-x1,x0)` to the **first 64 channels (head 0) only**; copy heads 1..15 through. Apply once to q-input, once to k-input (two instances or one layer reused). **Do NOT reuse `mrope_apply`/`vision_rope`** (half-split convention, C5). Index cos/sin per-batch (or share row 0) so batch=2 works.

### Step 3 — `layers/dit_attention.{h,cpp}` (med risk — fork `vision_attention.cpp`)
Inputs `0=q,1=k,2=v` `[B,1,128,1024]` (q/k already `dit_rope`'d). Non-causal FP32 SDPA, scale `1/√64`, 16 heads, **no weights**. Props `block_size(=24)`, `look_ahead(0|1)`, `look_backward(0|1)`; at `finalize` build `block_id[p]=p//24` and the additive mask (`0`/`−inf`). `supportBackwarding=false`. **Do NOT reuse `mha_core`** (mask commented out at `mha_core.h:372`, RoPE on all heads, causal-only). Each block instantiates with its own look props (L0,L10,L20 special). **Verify batch=2 correctness or run batch=1 ×2** (Talker batched-prefill MHA bug, risk R7).

### Step 4 — `layers/dit_adaln_zero.{h,cpp}` (med risk)
Inputs `0=hidden[B,1,128,1024]`, `1=time_emb[B,1,1,1024]`. Weight `linear[1024,6144]+b`. Internally: `emb=linear(SiLU(time))`; chunk-6 `[shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp]`. **Compute no-affine LayerNorm internally** (eps=1e-6) — **cannot use `layer_normalization` (forces gamma=ones/beta=zeros affine)**. Output the attn-pre-norm `LN·(1+scale_msa)+shift_msa`, and expose `gate_msa, shift_mlp, scale_mlp, gate_mlp` (and the `ff_norm` pre-norm `LN·(1+scale_mlp)+shift_mlp`) for the block's residual gating (`h+gate_msa·attn`, `h+gate_mlp·ff`). Fold the two no-affine LNs and the gating here to avoid spurious LN weights and load-order drift.

### Step 5 — `layers/dit_adaln_final.{h,cpp}` (low-med risk)
Inputs `0=hidden,1=time_emb`. Weight `linear[1024,2048]+b`. `emb=linear(SiLU(time))`; chunk-2 **`[scale,shift]` (scale FIRST, C4)**; out `LN_noaffine·(1+scale)+shift`.

### Step 6 — `models/qwen25_omni/qwen25_omni_dit.{h,cpp}` (med-high risk; mirror `qwen25_omni_bigvgan.cpp`)
`registerCustomLayers()` for the 4 new layers (REUSE: `fully_connected` for all Linears; activation `"tanh_gelu"`→`ACT_TANH_GELU` for MLP — confirmed `common_properties.h:987-988`, `acti_func.h:87`; activation `"swish"` for SiLU; addition/scale for residual gates).
**Per-step graph (compiled once, INFERENCE):** assemble raw concat `[.,912]` on host → graph `proj(912→1024)` → 22 blocks `{dit_adaln_zero → Wq/Wk/Wv → dit_rope(q),dit_rope(k) → dit_attention(per-layer mask) → Wo → +gate_msa·attn → ff_norm-mod → FC/tanh_gelu/FC → +gate_mlp·ff}` → `dit_adaln_final` → `proj_out` → velocity `[B,128,80]`. Side inputs refreshed per call: `time_emb` (host computes sinusoid+`time_mlp`, or only sinusoid + in-graph MLP); `cos/sin` constant. **Recommend two batch-1 forwards** (cond, null) combined on host over batch=2.
**Host driver:** precompute §4 conditioning once; run §3 RK4 loop calling the graph; permute final `y→[1,80,128]`; hand to `Qwen25OmniBigVGAN::vocode`. Add to `models/qwen25_omni/meson.build`; add layer `shared_library` entries to `Applications/CausalLM/layers/meson.build` mirroring the `snake_beta`/`antialiased_snake` blocks.

### Step 7 — Verification spikes (mirror BigVGAN HF-dump spikes)
- **Stage A (per-step velocity):** **extend `dump_token2wav_refs.py`** to hook `ode_function` and dump, for one fixed `(x=initial_state[:, :128], t=sway[1])`: `v` (post-CFG `[1,128,80]`), the **pre-CFG guided/null pair**, `ecapa128_pos/neg`, `code_embed`, and `time_emb`. The current dump has **only** final `dit_mel` — per-step tensors do **not** exist yet (risk R8). Inject identical `(x,t)` into the nntrainer per-step graph; match `v` to ~1e-4. This also lets you cross-check the **interleaved RoPE** layout and the per-layer block mask block-by-block.
- **Stage B (full RK4):** drive the host loop from `initial_state_full.npy[:, :128]` (HF's exact noise, **no C++ RNG**); compare final mel to `dit_mel.npy[1,80,128]`. Then end-to-end through BigVGAN.

---

## 6. Open runtime risks (priority order)

1. **R1 — Per-layer block mask is active and non-uniform (C2).** seq=128 spans 6 blocks (`[24×5, 8]`); 19 layers strictly block-diagonal, L0/L20 +prev, L10 +next. Boolean→additive (0/−inf). A wrong mask **degrades quality subtly, does not crash** — verify against a Stage-A HF mask/attention dump. (`modeling:3039-3040`, `configuration:853-855`.)
2. **R2 — Interleaved adjacent-pair RoPE on head 0 only (C5).** Differs from every existing nntrainer RoPE (half-split). `dit_rope` must match `[f0,f0,f1,f1,…]` + `(x0,x1)→(-x1,x0)` exactly; verify numerically in Stage A (no current intermediate dump). (`modeling:2469-2470,2910-2916,2965`.)
3. **R3 — ECAPA port.** TDNN/Res2Net/SE/AttentiveStatsPool with **reflect-pad `same` Conv1d** and **masked softmax** (`modeling:2487-2494,2620-2651,2749-2755`); ~40 weights. **Bring-up: precompute `ecapa128_pos`/`ecapa128_neg` in Python and inject.** Port to host C++/small graph last; verify `ECAPA(zeros)` separately (null row uses it, NOT a zero vector).
4. **R4 — CFG null branch must use `code_embed_uncond`=row-0 (C8) and `ecapa128_neg`, `speaker=0` (C1/C8).** Easy to wrongly zero the code vector; HF embeds **id 0**. (`modeling:2821-2822,3503`.)
5. **R5 — No-affine LayerNorm.** `layer_normalization` forces affine (gamma=ones/beta=zeros). Fold the 3 no-affine LNs/block-level into `dit_adaln_zero`/`dit_adaln_final` (eps=1e-6) — do not emit gamma/beta weights.
6. **R6 — AdaLN chunk orders (C3/C4):** per-block **shift-first** `[shift_msa,scale_msa,gate_msa,shift_mlp,scale_mlp,gate_mlp]`; final **scale-first** `[scale,shift]`. Mismatching is silent.
7. **R7 — CFG batch=2 vs the Talker batched-prefill MHA bug.** That bug made batched multi-row attention mismatch HF (last row wrong). **Run the two CFG rows as separate batch-1 forwards** unless `dit_attention` batch=2 is explicitly verified in Stage A.
8. **R8 — Dump only has final mel.** Block-by-block validation requires the **new Stage-A hooks** (R7-class risk if skipped). Production sampling is non-deterministic; only Stage-B with the **injected** `initial_state_full.npy` can bit-match.
9. **R9 — Activation key correctness:** MLP must use `"tanh_gelu"` (`ACT_TANH_GELU`, `acti_func.h:87`/`common_properties.h:988`), **not** `"gelu"` (erf). SiLU = `"swish"`.
10. **R10 — Side-input refresh per call:** `time_emb` changes every ODE eval. Use `inference` (like BigVGAN) and confirm the compiled graph accepts refreshed `input0`+`time_emb`+`cos/sin` without recompiling (BigVGAN uses `inference`; audio encoder uses `incremental_inference` — pick `inference`).
11. **R11 — Converter/graph DFS load-order coupling.** Safetensors storage order is **alphabetical**, NOT data-flow; lock converter emit order ↔ graph wiring together (`weight-bin-load-order-dfs` memory). `assert 360`. FC transpose `[out,in]→[in,out]`; `codec_embed` raw.

This is a well-scoped engineering task; Sonnet should be sufficient for executing the implementation steps above, escalating to deeper review only for the Stage-A numeric mismatch debugging (RoPE/mask) if it arises.

# HANDOFF — Qwen2.5-Omni Speech Output (nntrainer)

> **IN-REPO COPY** (committed 2026-07-02 for cross-machine continuation). On the original
> dev machine the live copies are `~/.claude/plans/*.md` and `~/.claude/projects/.../memory/*.md`;
> on any other machine use the files in THIS directory: confirmed plans `dit-2B-confirmed.md`,
> `bigvgan-2A-confirmed.md`, `purring-wondering-barto.md`, and memory snapshots under `memories/`.
> Machine-specific paths below (HF snapshot `$SNAP`, `HF_HOME`) must be adapted locally.

> **START HERE in a new session.** Written 2026-07-02. Branch: **`qwen25-omni-multimodal`**.
> Repo: `/home/jijoongmoon/WorkSpace1/nntrainer-p`. Related memories:
> [[omni-talker-speech-output]] (deep detail), [[omni-audio-architecture]] (multimodal-in),
> [[weight-bin-load-order-dfs]], [[omni-talker-batched-prefill-mha-bug]], [[plan-first-preference]].
> Confirmed plans: `~/.claude/plans/dit-2B-confirmed.md`, `bigvgan-2A-confirmed.md`, `purring-wondering-barto.md`.

---

## 0. TL;DR — the single next action

Building Qwen2.5-Omni **speech output** in nntrainer. Pipeline:
`input → Thinker(text+hidden) → Talker(codec ids) → DiT(mel) → BigVGAN(24kHz wav)`.

**Thinker, Talker, BigVGAN, DiT are ALL DONE & HF-verified** (2026-07-18, machine
nntrainer-Galaxy-Book6-Ultra, branch qwen25-omni-multimodal + gauss GPU merge):
Stage A velocity max 1.3e-5, Stage B dit_mel max 2.9e-5 / identical stats,
file-chained Talker→DiT→BigVGAN wav matches HF at max 2 int16 LSB (99.8% ≤1).

**Phase 2C core is ALSO DONE (2026-07-18, same session):**
- `Qwen25OmniToken2Wav` (models/qwen25_omni/qwen25_omni_token2wav.*): in-process
  DiT → BigVGAN chain, arch "Qwen25OmniToken2Wav", model dir = union of both
  converters' outputs. 64-code output byte-identical to the file-chained path.
- ECAPA-TDNN ported to host C++ (`ecapa_tdnn.{h,cpp}`, weights `ecapa.bin`
  emitted by the DiT converter; spec sheet extracted+verified from HF 4.57.6,
  numpy ref matched 6.6e-7). Wrapper computes pos=ECAPA(ref_mel.bin) and
  neg=ECAPA(zeros) itself; falls back to injected ecapa_pos/neg.bin.
- Variable length: `Qwen25OmniDiT::ensure_seq` / `Qwen25OmniBigVGAN::
  ensure_frames` recompile the graph and reload weights on a length change
  (weights are length-agnostic). Full 127-code talker reply verified:
  60960-sample wav matches HF at max 2 int16 LSB (99.93% ≤1).

**DiT-on-CUDA status (2026-07-18 evening):** opt-in via `NNTR_ENGINE=cuda`
(default stays pure CPU). The 156 block FCs carry engine=cuda tags → cuBLAS
FP32 SGEMM; custom dit_* layers + activations stay host (UVM-coherent).
CORRECT in sync mode (Stage A 1.4e-5 / Stage B 2.5e-5; the model auto-sets
NNTR_CUDA_HOST_MAPPED=1, DEV_ACT=0, M2B=0) but only CPU-parity speed:
- pinned pool puts the 1.25 GB of WEIGHTS on PCIe every forward (GPU util
  ~75% memory-bound); managed pool instead ping-pongs activation pages with
  the host layers. The missing fast config is weights=managed +
  activations=pinned — needs a small split in the cuda allocator policy
  (nntrainer/cuda_mem_allocator.cpp use_host_mapped is pool-global today).
- NNTR_CUDA_ASYNC=1 gives wrong output on this mixed cpu/cuda graph even
  with drain_if_async() at every host consumer (dit_* layers + dit_act) —
  ordering bug somewhere in the async submission path; gauss only validated
  async on the all-managed quantized decode path. Keep ASYNC=0.
- CPU got faster anyway: OpenMP over dit_attention heads → 64-code Stage B
  31.1 s → 25.8 s. COORDINATE core (gauss) changes with the e1 PR effort
  before touching the allocator/async paths.

**ONE-SHOT PROMPT→SPEECH WORKS (2026-07-18 night):** `NNTR_ENGINE=cpu
nntr_causallm <talker-dir> "<prompt>"` runs Thinker→Talker→DiT→BigVGAN in
one process (talker nntr_config: thinker_model_path + token2wav_model_path
+ speech_output). Verified: thinker reply 16/16 HF-exact, talker codes
127/127 HF-exact, 61440-sample wav out; ~180 s CPU e2e.
⚠️ **MUST build with `-Denable-fp16=false` on x86 for the talker**: with
fp16 ON, the ENABLE_FP16-gated mha_core x86 path (fp16 KV cache) produces
WILDLY wrong codec ids (first code 3505 vs 8028; degenerate 4216 repeats)
— reproduced by Stage A with HF inputs, and confirmed absent pre-merge
(worktree at 138d315fa) AND absent with fp16 off on the merged tree.
This is a real bug (not precision) in the gauss x86 fp16 KV/attention
path for the talker's GQA 14/2 + head_dim 64 shape; the thinker's shape
is unaffected. Root-cause pending — coordinate with e1 before touching
mha_core. Two merge-era fixes already landed: async-tokenizer join in
runEndToEnd, and fp16-off build portability (rms_norm_gpu _FP16 guard,
[[maybe_unused]] in mha_core/causal_lm).

**NEXT ACTION:** ① root-cause the x86 fp16 KV mha_core bug (Stage A repro
above makes it a 30-second test), ② the two CUDA perf items, ③ streaming/
chunked synthesis, ④ ECAPA/DiT graph reuse polish (per-utterance Token2Wav
construction in speakCodes re-loads 1.7 GB each call — cache it for
multi-utterance sessions).

**FIRST, in a fresh session, REGENERATE THE /tmp DUMPS (they are wiped on reboot)** — see §3, and MIND THE transformers<5 PIN (§3.7).

---

## 1. Status matrix

| Piece | State | Committed | HF-verified |
|---|---|---|---|
| Thinker (text/audio/image/video → text) | DONE | `85ed8af04` | yes (see [[omni-audio-architecture]]) |
| Talker Phase 1 (Thinker → codec ids) | DONE | `5ee53d971` | yes (127/127 codec ids exact) |
| **Phase 2A BigVGAN** (mel → 24kHz wav) | **DONE** | `0b313ad87` (+4 below) | yes (app end-to-end, 1 int16 LSB) |
| **Phase 2B DiT** (codec ids → mel) | **DONE** | dit_attention + qwen25_omni_dit + converter (2026-07-18) | yes (Stage A 1.3e-5, Stage B 2.9e-5) |
| Phase 2C end-to-end (Talker→DiT→BigVGAN) | file-chained wav VERIFIED (max 2 LSB); in-process wrapper + ECAPA port remain | — | yes (vs /tmp/omni_t2w_dump/wav.npy) |

## 2. Commits (this effort, newest first)
```
0b313ad87 [CausalLM] Add Qwen25OmniBigVGAN model (mel -> 24 kHz WAV), Phase 2A complete
004492d4a [CausalLM] BigVGAN: scale layer + weight converter (Stage C validated)
04dd24e31 [CausalLM] Add antialiased_snake layer (HF TorchActivation1d) for BigVGAN
9f93fc319 [CausalLM] Add snake_beta + conv1d_transpose layers for Token2Wav BigVGAN
710c93877 [nntrainer] Fix two latent Conv2DTransposeLayer bugs (output width + input bounds)
6eff6e200 [nntrainer] Restore InputDtype property removed during mixed-precision PoC
5ee53d971 [CausalLM] Qwen2.5-Omni Talker (Phase 1)
85ed8af04 [CausalLM] Qwen2.5-Omni Thinker
```

### UNCOMMITTED work on disk (survives sessions — real files, NOT /tmp)
- **New (untracked)** DiT helper layers: `Applications/CausalLM/layers/{dit_rope,dit_modulate,dit_gate}.{h,cpp}` (6 files) — **compile & link OK**.
- **Modified (tracked)**: `Applications/CausalLM/layers/meson.build` (dit foreach block), `Applications/CausalLM/meson.build` (dit deps + local spike exe targets).
- **Gitignored (local-only)** in `Applications/CausalLM/res/qwen2.5-omni/` (dir is in .gitignore; force-add real deliverables like the committed bigvgan converter): `dump_dit_refs.py` (DiT Stage-A dumper — KEEP), `dump_token2wav_refs.py`, `test_talker.py`, `proto_antialias.py`, `spike_ups0*.{cpp,py}`, `spike_antialias*.{cpp,py}`, `spike_bigvgan*.{cpp,py}`, `spike_*_prep.py`.
- **Commit strategy for DiT**: commit the helper layers + model + converter together AFTER Stage-A/B passes (like BigVGAN was). Keep spikes local. `token2wav_dit_converter.py` will need `git add -f` (gitignored dir) like `token2wav_bigvgan_converter.py` did.

---

## 3. ⚠️ ENVIRONMENT GOTCHAS (forgetting these WILL waste time)

1. **/tmp dumps are EPHEMERAL** — wiped on reboot/power-loss. A fresh session almost certainly has NO dumps. Regenerate (all offline, ~30–60s model load each):
   ```bash
   cd Applications/CausalLM/res/qwen2.5-omni
   SNAP=/home/jijoongmoon/WorkSpace1/.hf_cache/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e
   # (a) Talker codes  -> /tmp/omni_talker_dump/talker_codes.npy
   HF_HUB_OFFLINE=1 HF_HOME=/home/jijoongmoon/WorkSpace1/.hf_cache \
     python3 test_talker.py --model_path "$SNAP" --outdir /tmp/omni_talker_dump
   # (b) Token2Wav refs (BigVGAN + DiT sample) -> /tmp/omni_t2w_dump/
   HF_HUB_OFFLINE=1 HF_HOME=/home/jijoongmoon/WorkSpace1/.hf_cache \
     python3 dump_token2wav_refs.py --model_path "$SNAP" --talker_dump /tmp/omni_talker_dump --outdir /tmp/omni_t2w_dump
   # (c) DiT Stage-A per-step refs -> /tmp/omni_dit_dump/   (needs (b) done first)
   HF_HUB_OFFLINE=1 HF_HOME=/home/jijoongmoon/WorkSpace1/.hf_cache \
     python3 dump_dit_refs.py --model_path "$SNAP" --t2w_dump /tmp/omni_t2w_dump --outdir /tmp/omni_dit_dump --step 1
   # spike inputs (regen as needed): spike_ups0_prep.py, spike_antialias_prep.py, spike_bigvgan_prep.py
   ```
2. **HF scripts MUST run offline against the local snapshot**: always pass `--model_path "$SNAP"` (the path above) **and** `HF_HUB_OFFLINE=1 HF_HOME=/home/jijoongmoon/WorkSpace1/.hf_cache`. Otherwise `resolve_model_dir` re-downloads ~11 GB into `~/.cache` and fills the (previously 100%-full) `/home` disk.
3. **Build dir = `build_debug`** (it builds the CausalLM app + custom layers). Targets are **output paths, not names**: `ninja -C build_debug Applications/CausalLM/layers/libX_layer.so` or `.../nntr_causallm` or `.../spike_bigvgan`.
4. **meson regen is SLOW (~5–7 min) and looks hung** at `[0/1] Regenerating build files`. Cause: an unrelated `jni/prepare_encoder.sh` runs a **network download** (PicoGPT encoder) on every regen and fails/retries offline. This is NOT an error — be patient. It only regens when a `meson.build` changes.
5. Model checkpoint (HF Qwen2.5-Omni-3B) local snapshot `$SNAP` above; DiT/BigVGAN weights live in `model-00003-of-00003.safetensors`; all Token2Wav tensors are **F32**.
6. Generated model dirs under `Applications/CausalLM/models/qwen2.5-omni-3b-*/` are gitignored (bin+config); regenerate with the converters.
7. **transformers MUST be <5 (pin 4.57.6).** transformers 5.x refactored the
   DiT rotary to half-split `cat((freqs,freqs))` cos/sin while keeping the
   adjacent-pair rotate — silently DIFFERENT model output (dit_mel std
   2.376→1.641). Dumps made with 5.x are tainted references. Sanity: dit_mel
   stats must be min −11.616 / max −0.095 / mean −3.989 / std 2.376.
8. **rotary inv_freq must come from the CHECKPOINT** (`rotary_embed.inv_freq`,
   bf16-rounded values like 0.75^j), NOT the 10000^(−2j/64) formula — the
   4.4e-4 relative gap grows to 5e-2 in cos at s=127. The DiT converter emits
   `inv_freq.bin`; `Qwen25OmniDiT::load_weight` requires it.
9. Machine paths (Galaxy-Book6-Ultra): `SNAP=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/f75b40e3da2003cdd6e1829b1f420ca70797c34e`,
   python env `~/venv-omni` (torch CPU + transformers 4.57.6), build dir
   `build_gpu_verify` (opencl+cuda, clblast OFF). Stage prep:
   `dit_stage_prep.py --outdir <dir>` then `NNTR_DIT_STAGEA=1 nntr_causallm
   <dit model dir> <dir>` (Stage A) / same without env (Stage B full RK4).

---

## 4. Ground-truth dumps (what each contains)
- `/tmp/omni_talker_dump/` — `talker_codes.npy` [127] etc. (Talker output).
- `/tmp/omni_t2w_dump/` — BigVGAN refs (`mel/processed_mel/conv_pre/ups0..5/stage0..5/activation_post/wav`) + DiT `sample()` inputs (`dit_mel[1,80,128]`, `initial_state_full[1,30000,80]`, `cond[1,192]`, `ref_mel[1,400,80]`, `codes[64]`).
- `/tmp/omni_dit_dump/` — **DiT Stage-A per-step refs** (one ODE step, sway[1]=0.0152, apply_cfg): `x_in[1,128,80]`, `time_emb[1,1024]`, `ecapa_out[2,128]` (row0=ECAPA(ref_mel), row1=ECAPA(0)=null → **inject these as the 128-d cond**), `code_embed[1,128,512]`+`code_embed_uncond`, `rotary_0`(cos)/`rotary_1`(sin) `[2,128,64]`, `input_embed_out[2,128,1024]`, `block0_out[2,128,1024]`, `norm_out[2,128,1024]`, `proj_out[2,128,80]` (per-row velocity), `velocity_cfg[1,128,80]` (post-CFG).

---

## 5. Phase 2A BigVGAN — DONE (reference + reusable patterns)
- Model: `Applications/CausalLM/models/qwen25_omni/qwen25_omni_bigvgan.{h,cpp}` (arch `Qwen25OmniBigVGAN`, registered in `main.cpp`). Builds graph imperatively in `initialize()`, host `process_mel` + WAV writer + `vocode(mel→wav)`.
- Converter: `res/qwen2.5-omni/token2wav_bigvgan_converter.py` (committed, force-added). Output dir `models/qwen2.5-omni-3b-bigvgan/`.
- **Reusable spike harness** (mirror for DiT): `spike_bigvgan.cpp` + `spike_bigvgan_prep.py` — a 1-layer/1-graph ccapi model that `model->load(.bin)`, feeds a dumped `.npy` (as raw f32 `.bin`), runs `inference`, compares to a reference. `spike_ups0.*`, `spike_antialias.*` are per-layer versions.
- **Key learnings carried forward:**
  - Fixed **two latent `conv2d_transpose_layer.cpp` core bugs** (output width used `eff_k_height`; input-bounds check used kernel dims not input dims). The layer had **no wired unit test** — that's why they were latent.
  - **Weight load order = graph DFS-from-output = topological (input-side first, output last); sibling order at a branch = the `input_layers` order.** BigVGAN's forward-order converter matched on first try. DiT has a shared `time_mlp` branch → **don't guess; dump the graph's actual order and align the converter** (see §6.3).
  - nntrainer `multiply`/`Tensor::multiply` **do NOT broadcast** (drove the DiT decomposed design).
  - Custom-layer weight requests: `context.requestWeight(dim, Initializer::ZEROS, WeightRegularizer::NONE, 1.0f, 0.0f, "name", /*trainable*/false)`.
  - Base `Layer::incremental_forwarding` calls `forwarding` (`layer_devel.h:225`); BigVGAN/DiT run via `model->inference` (not incremental).
  - App model flow: `main.cpp` reads `<dir>/config.json` (architectures→factory) + `nntr_config.json` (`model_file_name`) → `initialize()` → `load_weight()` → `run(prompt)`.

---

## 6. Phase 2B DiT — THE ACTIVE WORK

**Read `~/.claude/plans/dit-2B-confirmed.md` first** — full confirmed spec (5-dim verification workflow vs HF+checkpoint+dump; found 8 corrections). DiT = flow-matching transformer producing mel `[1,80,128]` for BigVGAN, driven by a **host RK4 loop** over ONE compiled non-incremental graph.

### 6.1 CRITICAL corrections (each would SILENTLY fail if missed)
- **C1 ECAPA is IN-GRAPH, roles swapped:** the 128-d cond = `ECAPA_TDNN(ref_mel)` (40 `input_embed.spk_encoder.*` tensors); the 192-d = precomputed `cond.npy` raw. HF `sample()`↔`forward()` swap the `speaker_embedding`/`condition_vector` arg names. **Bring-up: precompute ECAPA in Python & inject** (use `/tmp/omni_dit_dump/ecapa_out.npy` row0=pos, row1=neg=ECAPA(0)). Port ECAPA to C++ LAST.
- **C2 mask = per-layer BOOLEAN block-diagonal** (block_size 24). Layers `look_backward=[0,20]`, `look_ahead=[10]`; **only L0/L20 attend one block back, L10 one block ahead; the other 19 are strict block-diagonal**. Convert bool→additive (0 keep / −inf).
- **C3 AdaLN chunk-6 order = `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`** (shift-first); 2 **no-affine** LayerNorms/block (can't use `layer_normalization` — forces affine).
- **C4 norm_out (final) chunk-2 = `[scale, shift]`** (SCALE-first, opposite of blocks).
- **C5 RoPE = adjacent-pair `(x0,x1)→(-x1,x0)` + interleaved cos/sin `[f0,f0,f1,f1,…]`, HEAD 0 ONLY** → `mrope_apply`/`vision_rope` CANNOT be reused (half-split). → `dit_rope`.
- **C6 CFG = ONE batch=2 forward.** Recommend running as **two batch-1 forwards** (see [[omni-talker-batched-prefill-mha-bug]]). 36 ODE calls (RK4 9 intervals × 4). Combine `1.5·guided − 0.5·null`.
- **C7 config.json `dit_config` aliases (dim/depth/heads) are IGNORED by HF** → hard-code: hidden 1024, 22 blocks, 16 heads, head_dim 64, ff_inner 2048, codec_embed [8194,512], repeats 2, block_size 24, look_ahead [10], look_backward [0,20], rope_theta 10000, seq 128.
- **C8 CFG null branch uses codec_embed ROW 0** (drop_code zeros ids, NOT a zero vector) + ECAPA(zeros) + speaker=0.

### 6.2 Architecture (decomposed — reuse FC + attention; multiply can't broadcast)
Custom layers (in `Applications/CausalLM/layers/`):
- **`dit_rope` (DONE, builds):** inputs (x[B,1,seq,1024], cos, sin); adjacent-pair rotate on channels 0..63 (head 0), rest pass-through. cos/sin `[.,seq,64]` host-filled interleaved.
- **`dit_modulate` (DONE, builds):** inputs (x[B,1,seq,1024], cond[B,1,1,M]); `noaffineLN(x,eps1e-6)*(1+scale)+shift`; `scale`/`shift` are 1024-wide slices of `cond` at props `scale_off`/`shift_off`.
- **`dit_gate` (DONE, builds):** inputs (residual, x, cond); `residual + cond[gate_off:+1024]·x` (broadcast).
- **`dit_attention` (TODO):** fork `vision_attention.{h,cpp}` (already does masked FP32 SDPA from q/k/v inputs, no weights). Swap the window mask for the block-diff mask; props `block_size`, `look_ahead`, `look_backward`; non-causal; scale 1/√64; 16 heads; head-0 q/k already RoPE'd upstream. (Per-layer instantiation: L0/L20 back=1, L10 ahead=1, else 0.)
Reuse: `fully_connected` (Wq/k/v/o, ff.0/ff.3, attn_norm.linear, norm_out.linear, proj_out), `activation` `swish`(=SiLU on time) + `tanh_gelu`(MLP), + the three helpers above.

**Per-block wiring** (`h`=hidden; `cond_i = FC_attn_norm(swish(time_emb))` → [B,6144]; **offsets**: shift_msa 0, scale_msa 1024, gate_msa 2048, shift_mlp 3072, scale_mlp 4096, gate_mlp 5120):
```
mod_a = dit_modulate(h, cond_i, scale_off=1024, shift_off=0)
q=FC_q(mod_a); k=FC_k(mod_a); v=FC_v(mod_a)
q=dit_rope(q,cos,sin); k=dit_rope(k,cos,sin)
a = dit_attention(q,k,v)          # per-layer block mask
ao = FC_o(a)
h2 = dit_gate(h, ao, cond_i, gate_off=2048)          # h + gate_msa·ao
mod_f = dit_modulate(h2, cond_i, scale_off=4096, shift_off=3072)
f = FC_ff3(tanh_gelu(FC_ff0(mod_f)))
h3 = dit_gate(h2, f, cond_i, gate_off=5120)          # h2 + gate_mlp·f
```
**Final:** `cond_o = FC_norm_out(swish(time_emb))` [B,2048]; `dit_modulate(h, cond_o, scale_off=0, shift_off=1024)` (scale-first!) → `FC_proj_out(1024→80)` → velocity `[B,128,80]`.

### 6.3 Remaining steps (in order)
1. **`dit_attention`** — fork `vision_attention`, block-diff mask (build additive 0/−inf mask at finalize from block_size + look props). Validate the whole block via `block0_out` (needs the model graph).
2. **`qwen25_omni_dit.{h,cpp}` model** (mirror `qwen25_omni_bigvgan.cpp`): `registerCustomLayers` (dit_rope/attention/modulate/gate); `initialize()` builds the per-step graph imperatively (input0 = host-assembled concat `[mel80|ecapa128|code512|spk192]`=912 → FC proj → 22 blocks → final). **Host driver:** precompute conditioning ONCE (ECAPA pos/neg from dump for bring-up; codec gather+`repeat_interleave(2)` per Talker idiom `qwen25_omni_talker_causallm.h`; time sinusoid `SinusPos(256,scale=1000)`; cos/sin interleaved [128,64]); **RK4 3/8-rule** 9 intervals × 4 evals; **CFG** two batch-1 forwards combined `1.5·guided−0.5·null`. Sway t = `[0,.0152,.0603,.134,.234,.357,.5,.658,.826,1.0]`. Then `permute → [1,80,128]` → hand to `Qwen25OmniBigVGAN::vocode`. meson (`models/qwen25_omni/meson.build`) + layer `foreach` already has rope/modulate/gate — add `dit_attention`; register arch in `main.cpp`.
3. **Converter `res/qwen2.5-omni/token2wav_dit_converter.py`** — **AFTER the graph compiles, dump its actual weight-load order** (e.g. via `forEachLayer` / weight iteration) and emit to match. `dit.bin` = **318 graph weights** (proj 2 + time_mlp 4 + 22×[attn_norm.linear, to_q/k/v, to_out.0, ff.0, ff.3 = 14] + norm_out 2 + proj_out 2). **Exclude** the 40 ECAPA `spk_encoder.*` + `text_embed.codec_embed` (host gather, separate `codec_embed.bin`) + `rotary_embed.inv_freq` (host recompute). FC weights `[out,in]→transpose→[in,out]`; biases raw. `git add -f` (gitignored dir).
4. **Stage A** — spike (mirror `spike_bigvgan.cpp`): host-assemble the concat from `/tmp/omni_dit_dump` inputs, run per-step graph, compare `input_embed_out` → `block0_out` → `proj_out`/`velocity_cfg` (~1e-4). Bisect via the intermediate dumps.
5. **Stage B** — host RK4 from `initial_state_full.npy[:, :128]` (inject HF noise; NO C++ RNG) → compare final mel to `dit_mel.npy [1,80,128]`.
6. **Phase 2C end-to-end** — Talker codes → DiT → BigVGAN → wav; compare to `/tmp/omni_t2w_dump/wav.npy`.

### 6.4 Open risks (from the plan; watch these in Stage A)
Per-layer block mask (R1), interleaved head-0 RoPE (R2 — if `block0_out` mismatches, dump per-op q/k), ECAPA(zeros)≠0 for null (R4), no-affine LN (R5), chunk orders shift/scale vs scale/shift (R6), CFG batch=2 → use batch-1 ×2 (R7).

---

## 7. File index
- Plans: `~/.claude/plans/{HANDOFF-omni-speech.md (this), dit-2B-confirmed.md, bigvgan-2A-confirmed.md, purring-wondering-barto.md}`
- Memories: `~/.claude/projects/-home-jijoongmoon-WorkSpace1-nntrainer-p/memory/{MEMORY.md, omni-talker-speech-output.md, omni-audio-architecture.md, weight-bin-load-order-dfs.md, omni-talker-batched-prefill-mha-bug.md}`
- DiT layers (uncommitted): `Applications/CausalLM/layers/{dit_rope,dit_modulate,dit_gate}.{h,cpp}` (+ TODO `dit_attention`)
- BigVGAN layers (committed): `Applications/CausalLM/layers/{snake_beta,conv1d_transpose,antialiased_snake,scale_layer}.{h,cpp}`
- Models: `Applications/CausalLM/models/qwen25_omni/qwen25_omni_bigvgan.{h,cpp}` (+ TODO `qwen25_omni_dit`)
- HF source: `/home/jijoongmoon/.local/lib/python3.10/site-packages/transformers/models/qwen2_5_omni/modeling_qwen2_5_omni.py` (DiT ~2455–3050: rotary 2455, InputEmbed 2780, CodecEmbed 2814, AdaLN 2831/2850, blocks/forward/sample 3486–3590; BigVGAN 3053–3379)
- Dump/spike scripts (gitignored, local): `Applications/CausalLM/res/qwen2.5-omni/{dump_dit_refs.py, dump_token2wav_refs.py, test_talker.py, proto_antialias.py, spike_*.{cpp,py}}`

## 8. Model selection
DiT implementation is **well-scoped engineering → Sonnet is sufficient** (the Opus verification workflow itself concluded this). Escalate to **Opus only for Stage-A numeric-mismatch debugging** (RoPE/mask). Do the build/wire/converter work on Sonnet.

## 9. Sanity: confirm the environment before coding
```bash
cd /home/jijoongmoon/WorkSpace1/nntrainer-p && git branch --show-current   # qwen25-omni-multimodal
git log --oneline -1                                                       # 0b313ad87
ls Applications/CausalLM/layers/dit_*.{h,cpp}                              # 6 files present (uncommitted)
ls /tmp/omni_dit_dump/ 2>/dev/null || echo "REGEN DUMPS (see §3)"
ls build_debug/Applications/CausalLM/layers/libdit_*.so                    # 3 helper .so built
```

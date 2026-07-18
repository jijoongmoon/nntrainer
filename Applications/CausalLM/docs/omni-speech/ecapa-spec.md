# ECAPA-TDNN Speaker Encoder Spec — Qwen2.5-Omni-3B Token2Wav DiT

Source of truth: `ECAPA_TimeDelayNet` in transformers **4.57.6**
`modeling_qwen2_5_omni.py` (spk_encoder of `DiTInputEmbedding`).
Validated bit-for-bit (max abs diff 6.6e-07 pos / 8.9e-08 neg vs
`/tmp/omni_dit_dump/ecapa_out.npy`) by the pure-numpy script `ecapa_ref.py`
in this directory.

All weights are float32. Checkpoint tensor prefix (abbreviated `P.` below):

```
P. = token2wav.code2wav_dit_model.input_embed.spk_encoder.
```

Config values (Qwen2.5-Omni-3B `dit_config`): mel_dim=80, enc_dim=128,
enc_channels=[256,256,256,256,768], enc_kernel_sizes=[5,3,3,3,1],
enc_dilations=[1,2,3,4,1], enc_attention_channels=64, enc_res2net_scale=2,
enc_se_channels=64.

## Primitive ops

### CONV1D_SAME_REFLECT(x, W, b, dilation)
PyTorch `nn.Conv1d(padding="same", padding_mode="reflect")`, stride 1.
- `x: [C_in, T]`, `W: [C_out, C_in, K]`, `b: [C_out]`, output `[C_out, T]`.
- Every kernel here is odd, so pad = `dilation*(K-1)/2` samples on **each**
  side, **reflect** mode (mirror around the edge sample, edge NOT repeated:
  left pads are `x[pad], ..., x[2], x[1]`, right pads are
  `x[T-2], x[T-3], ...`). Requires `pad < T` (T=400 in practice, max pad = 4).
- `out[o, t] = b[o] + sum_{c,k} W[o,c,k] * x_padded[c, t + k*dilation]`.
- For K=1 this is a plain per-frame matmul; padding irrelevant.

### TDNN(x, name, K, dilation)
`TimeDelayNetBlock` = `ReLU( CONV1D_SAME_REFLECT(x, P.name.conv.weight,
P.name.conv.bias, dilation) )`. No normalization layer anywhere in this model.

## Forward pass (batch entry processed independently)

Input: mel `[T, 80]` (T = 400 for the reference clip).
**Step 0 — transpose** to channels-first: `x0 = mel^T`, shape `[80, T]`.

### Step 1 — initial TDNN (`blocks.0`)
`h = TDNN(x0)`, K=5, dilation=1, pad 2 reflect. 80 -> 256. Out `[256, T]`.
- `P.blocks.0.conv.weight [256, 80, 5]`, `P.blocks.0.conv.bias [256]`

### Steps 2-4 — three SE-Res2Net blocks (`blocks.1`, `blocks.2`, `blocks.3`)
Identical structure; only the Res2Net dilation differs:
block 1 -> dilation 2, block 2 -> dilation 3, block 3 -> dilation 4.
Each block, with input `h [256, T]`:

1. `residual = h`
2. **tdnn1**: `h = TDNN(h)`, K=1, 256 -> 256.
   `P.blocks.i.tdnn1.conv.weight [256,256,1]`, `.bias [256]`
3. **Res2Net (scale = 2)**: split `h` along channels into two halves of 128
   (`h0 = h[0:128]`, `h1 = h[128:256]`).
   - `h0` passes through **unchanged** (identity — chunk index 0).
   - `h1 = TDNN(h1)`, K=3, dilation=d_i (2/3/4), pad d_i reflect, 128 -> 128.
     `P.blocks.i.res2net_block.blocks.0.conv.weight [128,128,3]`, `.bias [128]`
   - `h = concat([h0, h1], channel axis)` -> `[256, T]`.
   (With scale=2 there is exactly one conv branch and no cross-chunk
   accumulation; the `x_prev + out_prev` add only occurs for chunk index >= 2.)
4. **tdnn2**: `h = TDNN(h)`, K=1, 256 -> 256.
   `P.blocks.i.tdnn2.conv.weight [256,256,1]`, `.bias [256]`
5. **Squeeze-Excitation gate** (se_channels = 64):
   - `m = mean over time of h` -> `[256, 1]` (plain mean over all T frames,
     no masking).
   - `m = ReLU( conv1(m) )`, 1x1, 256 -> 64.
     `P.blocks.i.se_block.conv1.weight [64,256,1]`, `.bias [64]`
   - `m = sigmoid( conv2(m) )`, 1x1, 64 -> 256.
     `P.blocks.i.se_block.conv2.weight [256,64,1]`, `.bias [256]`
   - `h = h * m` (broadcast per channel over time).
6. **Residual add**: `h = h + residual`. Out `[256, T]`.

Keep each block's output: `f1, f2, f3` (each `[256, T]`).

### Step 5 — multi-layer feature aggregation (`mfa`)
`g = concat([f1, f2, f3], channel axis)` -> `[768, T]`
(the initial-TDNN output is **excluded** — only the 3 SE-Res2Net outputs).
`g = TDNN(g)`, K=1, dilation=1, 768 -> 768. Out `[768, T]`.
- `P.mfa.conv.weight [768, 768, 1]`, `P.mfa.conv.bias [768]`

### Step 6 — Attentive Statistics Pooling (`asp`), eps = 1e-12
The HF code builds a length mask, but lengths are hardcoded to the full
sequence (`lengths = ones * T`), so the mask is all-ones and the
`masked_fill(-inf)` is a no-op. A C++ port can drop masking entirely; the
formulas below are the effective math. Input `g [768, T]`, output `[1536]`.

1. **Global context stats** (uniform weights `1/T`):
   - `mu[c]   = (1/T) * sum_t g[c,t]`
   - `sd[c]   = sqrt( max( (1/T) * sum_t (g[c,t] - mu[c])^2 , 1e-12 ) )`
     (clamp is `clamp(min=eps)` BEFORE the sqrt; population variance,
     divisor T, not T-1).
2. **Attention input**: `A = concat([g, tile(mu,T), tile(sd,T)], channels)`
   -> `[2304, T]` (mu and sd broadcast to every frame).
3. `A = TDNN(A)`, K=1, 2304 -> 64 (conv + **ReLU**).
   `P.asp.tdnn.conv.weight [64, 2304, 1]`, `.bias [64]`
4. `A = tanh(A)`  (yes: ReLU then tanh — output in [0, tanh_max)).
5. `A = CONV1D(A)`, 1x1, 64 -> 768, **no activation**.
   `P.asp.conv.weight [768, 64, 1]`, `.bias [768]`
6. `att = softmax(A over the TIME axis)`, independently per channel:
   `att[c,t] = exp(A[c,t]) / sum_t' exp(A[c,t'])` (each channel row sums to 1).
7. **Attention-weighted stats**:
   - `mean[c] = sum_t att[c,t] * g[c,t]`
   - `std[c]  = sqrt( max( sum_t att[c,t] * (g[c,t] - mean[c])^2 , 1e-12 ) )`
8. `pooled = concat([mean, std])` -> `[1536]` (treated as `[1536, 1]`).

### Step 7 — final projection (`fc`)
1x1 Conv1d (i.e. affine), 1536 -> 128, **no activation**:
`out = fc.weight[:, :, 0] @ pooled + fc.bias` -> `[128]`.
- `P.fc.weight [128, 1536, 1]`, `P.fc.bias [128]`

Result: 128-dim speaker embedding per batch entry.

## Edge cases
- Zero input (the CFG "uncond" row) goes through the exact same path; the
  eps clamp makes std = 1e-6 in both stats steps, softmax is uniform (1/T).
  Ground-truth row1 is exactly this.
- No dropout, no norm layers, no masking effects at inference. Everything is
  deterministic float32.
- Reflect padding matters only for the K=5 (pad 2) and K=3 dilated convs
  (pad 2/3/4); all other convs are 1x1.

## Weight consumption order (all 40 tensors, exact names)

```
P.blocks.0.conv.weight                    [256, 80, 5]
P.blocks.0.conv.bias                      [256]
P.blocks.1.tdnn1.conv.weight              [256, 256, 1]
P.blocks.1.tdnn1.conv.bias                [256]
P.blocks.1.res2net_block.blocks.0.conv.weight [128, 128, 3]   (dilation 2)
P.blocks.1.res2net_block.blocks.0.conv.bias   [128]
P.blocks.1.tdnn2.conv.weight              [256, 256, 1]
P.blocks.1.tdnn2.conv.bias                [256]
P.blocks.1.se_block.conv1.weight          [64, 256, 1]
P.blocks.1.se_block.conv1.bias            [64]
P.blocks.1.se_block.conv2.weight          [256, 64, 1]
P.blocks.1.se_block.conv2.bias            [256]
P.blocks.2.tdnn1.conv.weight              [256, 256, 1]
P.blocks.2.tdnn1.conv.bias                [256]
P.blocks.2.res2net_block.blocks.0.conv.weight [128, 128, 3]   (dilation 3)
P.blocks.2.res2net_block.blocks.0.conv.bias   [128]
P.blocks.2.tdnn2.conv.weight              [256, 256, 1]
P.blocks.2.tdnn2.conv.bias                [256]
P.blocks.2.se_block.conv1.weight          [64, 256, 1]
P.blocks.2.se_block.conv1.bias            [64]
P.blocks.2.se_block.conv2.weight          [256, 64, 1]
P.blocks.2.se_block.conv2.bias            [256]
P.blocks.3.tdnn1.conv.weight              [256, 256, 1]
P.blocks.3.tdnn1.conv.bias                [256]
P.blocks.3.res2net_block.blocks.0.conv.weight [128, 128, 3]   (dilation 4)
P.blocks.3.res2net_block.blocks.0.conv.bias   [128]
P.blocks.3.tdnn2.conv.weight              [256, 256, 1]
P.blocks.3.tdnn2.conv.bias                [256]
P.blocks.3.se_block.conv1.weight          [64, 256, 1]
P.blocks.3.se_block.conv1.bias            [64]
P.blocks.3.se_block.conv2.weight          [256, 64, 1]
P.blocks.3.se_block.conv2.bias            [256]
P.mfa.conv.weight                         [768, 768, 1]
P.mfa.conv.bias                           [768]
P.asp.tdnn.conv.weight                    [64, 2304, 1]
P.asp.tdnn.conv.bias                      [64]
P.asp.conv.weight                         [768, 64, 1]
P.asp.conv.bias                           [768]
P.fc.weight                               [128, 1536, 1]
P.fc.bias                                 [128]
```

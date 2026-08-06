// SPDX-License-Identifier: Apache-2.0
/**
 * P1 — nntrainer CPU GatedDeltaNet reference + golden validation.
 *
 * Reproduces the Qwen3.6-35B (qwen3_5_moe) GatedDeltaNet forward, stage by
 * stage, using nntrainer::Tensor for the projection GEMMs and explicit,
 * spec-faithful loops for the GDN-specific math (causal depthwise conv1d,
 * L2-norm, decay-first delta recurrence, gated-RMSNorm). Validates every stage
 * against the P0 goldens (<1e-5), for both a full-sequence prefill and a
 * per-token step decode that carries recurrent state.
 *
 * Ground truth: transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
 *   - torch_recurrent_gated_delta_rule  (decay-first naive recurrence)
 *   - Qwen3_5MoeRMSNormGated            (rmsnorm(core)*weight*silu(z))
 *   - Qwen3_5MoeGatedDeltaNet.forward   (projection / conv / split / GQA order)
 *
 * Config (B, S, golden dir) is read from env GDN_B / GDN_S / GDN_DIR so the same
 * binary validates multiple golden cases. Head dims are the frozen tiny config.
 *
 * Build+run: Applications/CausalLM/gdn_ref/run_gdn_p1.sh
 */

#include <tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using nntrainer::Tensor;
using nntrainer::TensorDim;

// ---- frozen tiny config (mirrors the 35B GQA ratio; head dims fixed) ----
static constexpr int HID = 32;  // hidden_size
static constexpr int NVH = 4;   // linear_num_value_heads
static constexpr int NKH = 2;   // linear_num_key_heads
static constexpr int HKD = 8;   // linear_key_head_dim
static constexpr int HVD = 8;   // linear_value_head_dim
static constexpr int KEY_DIM = HKD * NKH;              // 16
static constexpr int VAL_DIM = HVD * NVH;              // 32
static constexpr int CONV_DIM = KEY_DIM * 2 + VAL_DIM; // 64
static constexpr int KS = 4;    // linear_conv_kernel_dim
static constexpr int GQA = NVH / NKH;                  // 2
static constexpr float EPS = 1e-6f;

// ---- scalar activations (explicit, numerically stable) ----
static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float siluf(float x) { return x * sigmoidf(x); }
static inline float softplusf(float x) {       // == torch.nn.functional.softplus
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

// ---- golden loader: flat little-endian float32, no header ----
static std::string g_dir;
static std::vector<float> loadBin(const std::string &name) {
  const std::string p = g_dir + "/" + name + ".bin";
  std::ifstream f(p, std::ios::in | std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("cannot open " + p);
  std::streamsize bytes = f.tellg();
  if (bytes % sizeof(float) != 0)
    throw std::runtime_error("not float32-aligned: " + p);
  f.seekg(0, std::ios::beg);
  std::vector<float> v(bytes / sizeof(float));
  f.read(reinterpret_cast<char *>(v.data()), bytes);
  if (!f)
    throw std::runtime_error("short read: " + p);
  return v;
}

static Tensor makeT(const std::vector<float> &v, int b, int c, int h, int w) {
  if ((size_t)(b * c * h * w) != v.size())
    throw std::runtime_error("makeT size mismatch");
  return Tensor(TensorDim(b, c, h, w), v.data());
}

// ---- comparison harness ----
static int g_fail = 0;
static void check(const std::string &label, const std::vector<float> &got,
                  const std::vector<float> &ref, float tol = 1e-5f) {
  if (got.size() != ref.size()) {
    std::cout << "[FAIL] " << label << " : size " << got.size() << " vs "
              << ref.size() << "\n";
    ++g_fail;
    return;
  }
  float d = 0.0f;
  for (size_t i = 0; i < got.size(); ++i)
    d = std::max(d, std::fabs(got[i] - ref[i]));
  bool ok = d < tol;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << " : max|d| = " << d
            << "\n";
  if (!ok)
    ++g_fail;
}

// ===========================================================================
//  GDN reference forward (CPU, FP32)
// ===========================================================================
struct GDNOut {
  std::vector<float> mixed_qkv; // [B*S, CONV_DIM]  (token-major)
  std::vector<float> conv_out;  // [B*S, CONV_DIM]
  std::vector<float> conv_out_decode; // [B*S, CONV_DIM] via per-step conv state
  std::vector<float> z;         // [B*S, VAL_DIM]   (= [.,NVH,HVD])
  std::vector<float> q, k;      // [B*S, NVH, HKD]  (post GQA repeat, pre-l2norm)
  std::vector<float> v;         // [B*S, NVH, HVD]
  std::vector<float> q_l2, k_l2;// [B*S, NVH, HKD]  (post-l2norm, no scale)
  std::vector<float> beta, g;   // [B*S, NVH]
  std::vector<float> core;      // [B*S, NVH, HVD]
  std::vector<float> state;     // [B, NVH, HKD, HVD]  final recurrent state
  std::vector<float> normed;    // [B*S, VAL_DIM]
  std::vector<float> out;       // [B*S, HID]
};

// decay-first delta recurrence for one (batch, v-head) over tokens [t0,t1) of
// sequence bi; updates Sh in place, writes core. Mirrors
// torch_recurrent_gated_delta_rule exactly:
//   S*=exp(g); kv=(S*k).sum_kd; delta=(v-kv)*beta; S+=k(x)delta; o=(S*q).sum_kd
static void recurStep(int bi, int vh, int t0, int t1, int Sn, float scale,
                      const std::vector<float> &q_l2,
                      const std::vector<float> &k_l2,
                      const std::vector<float> &v, const std::vector<float> &g,
                      const std::vector<float> &beta,
                      std::vector<float> &Sh /*[HKD*HVD]*/,
                      std::vector<float> &core) {
  for (int t = t0; t < t1; ++t) {
    const int tok = bi * Sn + t;
    const float gt = std::exp(g[tok * NVH + vh]);
    const float bt = beta[tok * NVH + vh];
    const float *qrow = &q_l2[(tok * NVH + vh) * HKD];
    const float *krow = &k_l2[(tok * NVH + vh) * HKD];
    const float *vrow = &v[(tok * NVH + vh) * HVD];

    for (int i = 0; i < HKD * HVD; ++i) // S *= exp(g)   (decay first)
      Sh[i] *= gt;

    float kv[HVD] = {0}; // kv_mem[vd] = sum_kd S[kd,vd] * k[kd]
    for (int kd = 0; kd < HKD; ++kd) {
      const float kk = krow[kd];
      const float *srow = &Sh[kd * HVD];
      for (int vd = 0; vd < HVD; ++vd)
        kv[vd] += srow[vd] * kk;
    }
    float delta[HVD]; // delta = (v - kv_mem) * beta
    for (int vd = 0; vd < HVD; ++vd)
      delta[vd] = (vrow[vd] - kv[vd]) * bt;
    for (int kd = 0; kd < HKD; ++kd) { // S += k (outer) delta
      const float kk = krow[kd];
      float *srow = &Sh[kd * HVD];
      for (int vd = 0; vd < HVD; ++vd)
        srow[vd] += kk * delta[vd];
    }
    float *orow = &core[(tok * NVH + vh) * HVD]; // o[vd] = sum_kd S[kd,vd]*q[kd]*scale
    for (int vd = 0; vd < HVD; ++vd)
      orow[vd] = 0.0f;
    for (int kd = 0; kd < HKD; ++kd) {
      const float qq = qrow[kd] * scale;
      const float *srow = &Sh[kd * HVD];
      for (int vd = 0; vd < HVD; ++vd)
        orow[vd] += srow[vd] * qq;
    }
  }
}

static GDNOut forwardGDN(int Bn, int Sn, bool step_decode) {
  GDNOut o;
  const int BS = Bn * Sn;
  const float scale = 1.0f / std::sqrt((float)HKD);

  // --- weights (HF [out,in]) + input ---
  Tensor hidden = makeT(loadBin("hidden"), Bn, 1, Sn, HID);
  Tensor Wqkv = makeT(loadBin("w_in_proj_qkv"), 1, 1, CONV_DIM, HID);
  Tensor Wz = makeT(loadBin("w_in_proj_z"), 1, 1, VAL_DIM, HID);
  Tensor Wb = makeT(loadBin("w_in_proj_b"), 1, 1, NVH, HID);
  Tensor Wa = makeT(loadBin("w_in_proj_a"), 1, 1, NVH, HID);
  Tensor Wout = makeT(loadBin("w_out_proj"), 1, 1, HID, VAL_DIM);
  std::vector<float> wconv = loadBin("w_conv1d");  // [CONV_DIM,1,KS]
  std::vector<float> A_log = loadBin("A_log");     // [NVH]
  std::vector<float> dt_bias = loadBin("dt_bias"); // [NVH]
  std::vector<float> wnorm = loadBin("w_norm");    // [HVD]

  // --- projections via nntrainer GEMM:  y = x @ W_hf^T   (trans_in=true) ---
  // dot flattens (b*c*h)=B*S tokens; each token is projected independently.
  Tensor t_qkv = hidden.dot(Wqkv, false, true); // [B,1,S,CONV_DIM]
  Tensor t_z = hidden.dot(Wz, false, true);     // [B,1,S,VAL_DIM]
  Tensor t_b = hidden.dot(Wb, false, true);     // [B,1,S,NVH]
  Tensor t_a = hidden.dot(Wa, false, true);     // [B,1,S,NVH]
  const float *pq = t_qkv.getData<float>();
  const float *pz = t_z.getData<float>();
  const float *pb = t_b.getData<float>();
  const float *pa = t_a.getData<float>();
  o.mixed_qkv.assign(pq, pq + BS * CONV_DIM);
  o.z.assign(pz, pz + BS * VAL_DIM);

  // --- beta / g (per token, per v-head) ---
  o.beta.resize(BS * NVH);
  o.g.resize(BS * NVH);
  for (int i = 0; i < BS; ++i)
    for (int h = 0; h < NVH; ++h) {
      o.beta[i * NVH + h] = sigmoidf(pb[i * NVH + h]);
      o.g[i * NVH + h] =
        -std::exp(A_log[h]) * softplusf(pa[i * NVH + h] + dt_bias[h]);
    }

  // --- causal depthwise conv1d (K=4, per-sequence left-pad 3) + SiLU ---
  //   conv_out[c,t] = silu( sum_{j=0..3} w[c,0,j] * x[c, t-3+j] ), per sequence
  o.conv_out.resize(BS * CONV_DIM);
  for (int bi = 0; bi < Bn; ++bi)
    for (int c = 0; c < CONV_DIM; ++c)
      for (int t = 0; t < Sn; ++t) {
        float acc = 0.0f;
        for (int j = 0; j < KS; ++j) {
          int ti = t - (KS - 1) + j;
          float x = (ti < 0) ? 0.0f : o.mixed_qkv[(bi * Sn + ti) * CONV_DIM + c];
          acc += wconv[c * KS + j] * x;
        }
        o.conv_out[(bi * Sn + t) * CONV_DIM + c] = siluf(acc);
      }
  // per-step variant with a (KS-1)-wide ring buffer (the decode-time conv state)
  o.conv_out_decode.resize(BS * CONV_DIM);
  for (int bi = 0; bi < Bn; ++bi) {
    std::vector<float> st(CONV_DIM * (KS - 1), 0.0f); // [c][KS-1] oldest..newest
    for (int t = 0; t < Sn; ++t)
      for (int c = 0; c < CONV_DIM; ++c) {
        float win[KS];
        for (int j = 0; j < KS - 1; ++j)
          win[j] = st[c * (KS - 1) + j];
        win[KS - 1] = o.mixed_qkv[(bi * Sn + t) * CONV_DIM + c];
        float acc = 0.0f;
        for (int j = 0; j < KS; ++j)
          acc += wconv[c * KS + j] * win[j];
        o.conv_out_decode[(bi * Sn + t) * CONV_DIM + c] = siluf(acc);
        for (int j = 0; j < KS - 2; ++j) // shift ring buffer
          st[c * (KS - 1) + j] = st[c * (KS - 1) + j + 1];
        st[c * (KS - 1) + (KS - 2)] = win[KS - 1];
      }
  }

  // --- split [q|k|v] + GQA repeat_interleave(rep=GQA): vh -> k-head vh/GQA ---
  o.q.resize(BS * NVH * HKD);
  o.k.resize(BS * NVH * HKD);
  o.v.resize(BS * NVH * HVD);
  for (int i = 0; i < BS; ++i)
    for (int vh = 0; vh < NVH; ++vh) {
      const int kh = vh / GQA;
      for (int d = 0; d < HKD; ++d) {
        o.q[(i * NVH + vh) * HKD + d] = o.conv_out[i * CONV_DIM + kh * HKD + d];
        o.k[(i * NVH + vh) * HKD + d] =
          o.conv_out[i * CONV_DIM + KEY_DIM + kh * HKD + d];
      }
      for (int d = 0; d < HVD; ++d)
        o.v[(i * NVH + vh) * HVD + d] =
          o.conv_out[i * CONV_DIM + 2 * KEY_DIM + vh * HVD + d];
    }

  // --- L2-norm(q,k) over last dim:  x * rsqrt(sum(x^2)+eps)  (no scale) ---
  o.q_l2 = o.q;
  o.k_l2 = o.k;
  auto l2 = [&](std::vector<float> &x) {
    for (int i = 0; i < BS * NVH; ++i) {
      float *row = &x[i * HKD];
      float ss = 0.0f;
      for (int d = 0; d < HKD; ++d)
        ss += row[d] * row[d];
      const float inv = 1.0f / std::sqrt(ss + EPS);
      for (int d = 0; d < HKD; ++d)
        row[d] *= inv;
    }
  };
  l2(o.q_l2);
  l2(o.k_l2);

  // --- recurrence (decay-first); prefill = one [0,S) call, decode = per token ---
  o.core.assign(BS * NVH * HVD, 0.0f);
  o.state.assign(Bn * NVH * HKD * HVD, 0.0f);
  for (int bi = 0; bi < Bn; ++bi)
    for (int vh = 0; vh < NVH; ++vh) {
      std::vector<float> Sh(HKD * HVD, 0.0f);
      if (step_decode)
        for (int t = 0; t < Sn; ++t)
          recurStep(bi, vh, t, t + 1, Sn, scale, o.q_l2, o.k_l2, o.v, o.g,
                    o.beta, Sh, o.core);
      else
        recurStep(bi, vh, 0, Sn, Sn, scale, o.q_l2, o.k_l2, o.v, o.g, o.beta, Sh,
                  o.core);
      for (int i = 0; i < HKD * HVD; ++i)
        o.state[(bi * NVH + vh) * HKD * HVD + i] = Sh[i];
    }

  // --- gated RMSNorm:  rmsnorm(core) * weight * silu(z)   (mean over HVD) ---
  o.normed.resize(BS * VAL_DIM);
  for (int i = 0; i < BS; ++i)
    for (int vh = 0; vh < NVH; ++vh) {
      const float *crow = &o.core[(i * NVH + vh) * HVD];
      const float *zrow = &o.z[i * VAL_DIM + vh * HVD];
      float var = 0.0f;
      for (int d = 0; d < HVD; ++d)
        var += crow[d] * crow[d];
      var /= HVD;
      const float inv = 1.0f / std::sqrt(var + EPS);
      for (int d = 0; d < HVD; ++d)
        o.normed[i * VAL_DIM + vh * HVD + d] =
          crow[d] * inv * wnorm[d] * siluf(zrow[d]);
    }

  // --- out_proj:  out = normed @ W_out^T ---
  Tensor t_normed = makeT(o.normed, Bn, 1, Sn, VAL_DIM);
  Tensor t_out = t_normed.dot(Wout, false, true); // [B,1,S,HID]
  const float *pout = t_out.getData<float>();
  o.out.assign(pout, pout + BS * HID);
  return o;
}

// Transpose token-major [B*S, C] -> golden layout [B, C, S].
static std::vector<float> toBCS(const std::vector<float> &x, int Bn, int Sn,
                                int C) {
  std::vector<float> y(Bn * C * Sn);
  for (int bi = 0; bi < Bn; ++bi)
    for (int t = 0; t < Sn; ++t)
      for (int c = 0; c < C; ++c)
        y[(bi * C + c) * Sn + t] = x[(bi * Sn + t) * C + c];
  return y;
}

static int runCase(int Bn, int Sn) {
  std::cout << "=== case B=" << Bn << " S=" << Sn << "  dir=" << g_dir
            << " ===\n";
  int before = g_fail;
  GDNOut o = forwardGDN(Bn, Sn, /*step_decode=*/false);

  std::cout << "-- prefill (full-sequence recurrence) --\n";
  check("mixed_qkv", toBCS(o.mixed_qkv, Bn, Sn, CONV_DIM), loadBin("s_mixed_qkv"));
  check("conv_out", toBCS(o.conv_out, Bn, Sn, CONV_DIM), loadBin("s_conv_out"));
  check("conv_decode==conv (state path)", o.conv_out_decode, o.conv_out, 1e-6f);
  check("z", o.z, loadBin("s_z"));
  check("q (post-GQA)", o.q, loadBin("s_q"));
  check("k (post-GQA)", o.k, loadBin("s_k"));
  check("v", o.v, loadBin("s_v"));
  check("q_l2", o.q_l2, loadBin("s_q_l2"));
  check("k_l2", o.k_l2, loadBin("s_k_l2"));
  check("beta", o.beta, loadBin("s_beta"));
  check("g", o.g, loadBin("s_g"));
  check("core_attn_out", o.core, loadBin("s_core_attn_out"));
  check("final_state", o.state, loadBin("s_final_state"));
  check("normed", o.normed, loadBin("s_normed"));
  check("out (layer output)", o.out, loadBin("out"));

  std::cout << "-- step decode (per-token recurrence carrying state) --\n";
  GDNOut d = forwardGDN(Bn, Sn, /*step_decode=*/true);
  check("core (decode) vs golden", d.core, loadBin("s_core_attn_out"));
  check("final_state (decode) vs golden", d.state, loadBin("s_final_state"));
  check("decode core == prefill core", d.core, o.core, 1e-6f);
  check("decode state == prefill state", d.state, o.state, 1e-6f);
  check("decode out == prefill out", d.out, o.out, 1e-6f);
  std::cout << "  (" << (g_fail - before) << " failed in this case)\n\n";
  return g_fail - before;
}

int main() {
  std::cout << "=== P1 GatedDeltaNet CPU reference vs P0 goldens ===\n\n";
  const char *eb = std::getenv("GDN_B");
  const char *es = std::getenv("GDN_S");
  const char *ed = std::getenv("GDN_DIR");
  int Bn = eb ? std::atoi(eb) : 1;
  int Sn = es ? std::atoi(es) : 8;
  g_dir = ed ? ed : "/home/aisjetson/jijoongmoon/gdn_p0/bin";
  runCase(Bn, Sn);
  std::cout << "=== " << (g_fail == 0 ? "ALL CHECKS PASS" : "FAILURES") << " ("
            << g_fail << " failed) ===\n";
  return g_fail == 0 ? 0 : 1;
}

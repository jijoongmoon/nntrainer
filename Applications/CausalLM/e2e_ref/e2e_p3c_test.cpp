// SPDX-License-Identifier: Apache-2.0
/**
 * P3c — nntrainer CPU Qwen3_5Moe FULL end-to-end reference + golden check.
 *
 * Closes P3's acceptance criterion: e2e logits match HF Qwen3_5MoeForCausalLM.
 * Pipeline: embed -> [decoder layers, residual-stream threaded by layer_types]
 *           -> final RMSNorm -> lm_head -> logits.
 * Reuses the P1 (GDN) / P2 (MoE) / P3a (full-attn) math, all already validated.
 *
 * N-layer model dispatched by layer_types.bin (0=GDN, 1=full-attn), vocab 64,
 * S and layer count from goldens (E2E_DIR). Cases: 2-layer [G,G] + 4-layer [G,G,G,A].
 * Goldens: attn_p3/e2e_bin{,4}. Build+run: Applications/CausalLM/e2e_ref/run_e2e_p3c.sh
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

static constexpr int HID = 32, VOCAB = 64;
static constexpr float EPS = 1e-6f;
static constexpr int NVH = 4, NKH = 2, HKD = 8, HVD = 8;
static constexpr int KEY_DIM = HKD * NKH, VAL_DIM = HVD * NVH;
static constexpr int CONV_DIM = KEY_DIM * 2 + VAL_DIM, KS = 4, GQA_G = NVH / NKH;
static constexpr int AH = 4, AKV = 2, AHD = 16, ROT = 4, QD = AH * AHD, AREP = AH / AKV;
static constexpr int E = 8, KK = 2, INTER = 16, SINTER = 16;
static int g_T = 0; // tokens (runtime)

static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float siluf(float x) { return x * sigmoidf(x); }
static inline float softplusf(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

static std::string g_dir;
static std::vector<float> loadBin(const std::string &name) {
  const std::string p = g_dir + "/" + name + ".bin";
  std::ifstream f(p, std::ios::in | std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + p);
  std::streamsize bytes = f.tellg();
  f.seekg(0, std::ios::beg);
  std::vector<float> v(bytes / sizeof(float));
  f.read(reinterpret_cast<char *>(v.data()), bytes);
  if (!f) throw std::runtime_error("short read: " + p);
  return v;
}
static Tensor makeT(const std::vector<float> &v, int h, int w) {
  return Tensor(TensorDim(1, 1, h, w), v.data());
}
static int g_fail = 0;
static void check(const std::string &label, const std::vector<float> &got,
                  const std::vector<float> &ref, float tol = 1e-5f) {
  if (got.size() != ref.size()) {
    std::cout << "[FAIL] " << label << " size " << got.size() << " vs " << ref.size() << "\n";
    ++g_fail; return;
  }
  float d = 0.0f;
  for (size_t i = 0; i < got.size(); ++i) d = std::max(d, std::fabs(got[i] - ref[i]));
  bool ok = d < tol;
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << label << " : max|d| = " << d << "\n";
  if (!ok) ++g_fail;
}

static std::vector<float> rmsnorm_1plus(const std::vector<float> &x, const std::vector<float> &w) {
  std::vector<float> y(g_T * HID);
  for (int t = 0; t < g_T; ++t) {
    const float *xr = &x[t * HID];
    float var = 0.0f;
    for (int d = 0; d < HID; ++d) var += xr[d] * xr[d];
    var /= HID;
    float inv = 1.0f / std::sqrt(var + EPS);
    for (int d = 0; d < HID; ++d) y[t * HID + d] = xr[d] * inv * (1.0f + w[d]);
  }
  return y;
}

static std::vector<float> gdn_forward(const std::string &pfx, const std::vector<float> &x) {
  const int T = g_T;
  const std::string g = pfx + "w_linear_attn_";
  Tensor X = makeT(x, T, HID);
  Tensor Wqkv = makeT(loadBin(g + "in_proj_qkv_weight"), CONV_DIM, HID);
  Tensor Wz = makeT(loadBin(g + "in_proj_z_weight"), VAL_DIM, HID);
  Tensor Wb = makeT(loadBin(g + "in_proj_b_weight"), NVH, HID);
  Tensor Wa = makeT(loadBin(g + "in_proj_a_weight"), NVH, HID);
  Tensor Wout = makeT(loadBin(g + "out_proj_weight"), HID, VAL_DIM);
  auto wconv = loadBin(g + "conv1d_weight");
  auto A_log = loadBin(g + "A_log");
  auto dt_bias = loadBin(g + "dt_bias");
  auto wnorm = loadBin(g + "norm_weight");
  const float scale = 1.0f / std::sqrt((float)HKD);
  Tensor t_qkv = X.dot(Wqkv, false, true);
  std::vector<float> mixed(t_qkv.getData<float>(), t_qkv.getData<float>() + T * CONV_DIM);
  Tensor t_z = X.dot(Wz, false, true);
  std::vector<float> z(t_z.getData<float>(), t_z.getData<float>() + T * VAL_DIM);
  Tensor t_b = X.dot(Wb, false, true), t_a = X.dot(Wa, false, true);
  const float *pb = t_b.getData<float>(), *pa = t_a.getData<float>();
  std::vector<float> conv(T * CONV_DIM);
  for (int c = 0; c < CONV_DIM; ++c)
    for (int t = 0; t < T; ++t) {
      float acc = 0.0f;
      for (int j = 0; j < KS; ++j) {
        int ti = t - (KS - 1) + j;
        acc += wconv[c * KS + j] * (ti < 0 ? 0.0f : mixed[ti * CONV_DIM + c]);
      }
      conv[t * CONV_DIM + c] = siluf(acc);
    }
  std::vector<float> q(T * NVH * HKD), k(T * NVH * HKD), v(T * NVH * HVD), beta(T * NVH), gg(T * NVH);
  for (int t = 0; t < T; ++t)
    for (int vh = 0; vh < NVH; ++vh) {
      const int kh = vh / GQA_G;
      for (int d = 0; d < HKD; ++d) {
        q[(t * NVH + vh) * HKD + d] = conv[t * CONV_DIM + kh * HKD + d];
        k[(t * NVH + vh) * HKD + d] = conv[t * CONV_DIM + KEY_DIM + kh * HKD + d];
      }
      for (int d = 0; d < HVD; ++d)
        v[(t * NVH + vh) * HVD + d] = conv[t * CONV_DIM + 2 * KEY_DIM + vh * HVD + d];
      beta[t * NVH + vh] = sigmoidf(pb[t * NVH + vh]);
      gg[t * NVH + vh] = -std::exp(A_log[vh]) * softplusf(pa[t * NVH + vh] + dt_bias[vh]);
    }
  auto l2 = [&](std::vector<float> &a) {
    for (int i = 0; i < T * NVH; ++i) {
      float *r = &a[i * HKD], ss = 0.0f;
      for (int d = 0; d < HKD; ++d) ss += r[d] * r[d];
      float inv = 1.0f / std::sqrt(ss + EPS);
      for (int d = 0; d < HKD; ++d) r[d] *= inv;
    }
  };
  l2(q); l2(k);
  std::vector<float> core(T * NVH * HVD, 0.0f);
  for (int vh = 0; vh < NVH; ++vh) {
    std::vector<float> Sh(HKD * HVD, 0.0f);
    for (int t = 0; t < T; ++t) {
      const float gt = std::exp(gg[t * NVH + vh]), bt = beta[t * NVH + vh];
      const float *qr = &q[(t * NVH + vh) * HKD], *kr = &k[(t * NVH + vh) * HKD], *vr = &v[(t * NVH + vh) * HVD];
      for (int i = 0; i < HKD * HVD; ++i) Sh[i] *= gt;
      float kv[HVD] = {0};
      for (int a = 0; a < HKD; ++a) for (int b = 0; b < HVD; ++b) kv[b] += Sh[a * HVD + b] * kr[a];
      float dl[HVD];
      for (int b = 0; b < HVD; ++b) dl[b] = (vr[b] - kv[b]) * bt;
      for (int a = 0; a < HKD; ++a) for (int b = 0; b < HVD; ++b) Sh[a * HVD + b] += kr[a] * dl[b];
      float *o = &core[(t * NVH + vh) * HVD];
      for (int b = 0; b < HVD; ++b) o[b] = 0.0f;
      for (int a = 0; a < HKD; ++a) { float qq = qr[a] * scale; for (int b = 0; b < HVD; ++b) o[b] += Sh[a * HVD + b] * qq; }
    }
  }
  std::vector<float> normed(T * VAL_DIM);
  for (int t = 0; t < T; ++t)
    for (int vh = 0; vh < NVH; ++vh) {
      const float *cr = &core[(t * NVH + vh) * HVD], *zr = &z[t * VAL_DIM + vh * HVD];
      float var = 0.0f;
      for (int d = 0; d < HVD; ++d) var += cr[d] * cr[d];
      var /= HVD;
      float inv = 1.0f / std::sqrt(var + EPS);
      for (int d = 0; d < HVD; ++d) normed[t * VAL_DIM + vh * HVD + d] = cr[d] * inv * wnorm[d] * siluf(zr[d]);
    }
  Tensor t_out = makeT(normed, T, VAL_DIM).dot(Wout, false, true);
  const float *po = t_out.getData<float>();
  return std::vector<float>(po, po + T * HID);
}

static std::vector<float> attn_forward(const std::string &pfx, const std::vector<float> &x) {
  const int T = g_T;
  const std::string g = pfx + "w_self_attn_";
  Tensor X = makeT(x, T, HID);
  Tensor Wq = makeT(loadBin(g + "q_proj_weight"), AH * AHD * 2, HID);
  Tensor Wk = makeT(loadBin(g + "k_proj_weight"), AKV * AHD, HID);
  Tensor Wv = makeT(loadBin(g + "v_proj_weight"), AKV * AHD, HID);
  Tensor Wo = makeT(loadBin(g + "o_proj_weight"), HID, QD);
  auto wqn = loadBin(g + "q_norm_weight"), wkn = loadBin(g + "k_norm_weight");
  auto cosb = loadBin("cos"), sinb = loadBin("sin"); // model-global rotary
  Tensor t_q = X.dot(Wq, false, true), t_k = X.dot(Wk, false, true), t_v = X.dot(Wv, false, true);
  const float *pq = t_q.getData<float>(), *pk = t_k.getData<float>(), *pv = t_v.getData<float>();
  auto qknorm = [&](float *r, const float *w) {
    float var = 0.0f;
    for (int d = 0; d < AHD; ++d) var += r[d] * r[d];
    var /= AHD;
    float inv = 1.0f / std::sqrt(var + EPS);
    for (int d = 0; d < AHD; ++d) r[d] = r[d] * inv * (1.0f + w[d]);
  };
  auto rope = [&](float *r, const float *cs, const float *sn) {
    const int half = ROT / 2; float xr[ROT];
    for (int i = 0; i < ROT; ++i) xr[i] = r[i];
    for (int i = 0; i < ROT; ++i) { float rh = (i < half) ? -xr[i + half] : xr[i - half]; r[i] = xr[i] * cs[i] + rh * sn[i]; }
  };
  std::vector<float> q(T * AH * AHD), gate(T * AH * AHD), k(T * AKV * AHD), v(T * AKV * AHD);
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < AH; ++h) {
      const float *blk = &pq[t * (AH * AHD * 2) + h * (AHD * 2)];
      float *qh = &q[(t * AH + h) * AHD];
      for (int d = 0; d < AHD; ++d) { qh[d] = blk[d]; gate[(t * AH + h) * AHD + d] = blk[AHD + d]; }
      qknorm(qh, wqn.data()); rope(qh, &cosb[t * ROT], &sinb[t * ROT]);
    }
    for (int j = 0; j < AKV; ++j) {
      float *kh = &k[(t * AKV + j) * AHD], *vh = &v[(t * AKV + j) * AHD];
      for (int d = 0; d < AHD; ++d) { kh[d] = pk[t * (AKV * AHD) + j * AHD + d]; vh[d] = pv[t * (AKV * AHD) + j * AHD + d]; }
      qknorm(kh, wkn.data()); rope(kh, &cosb[t * ROT], &sinb[t * ROT]);
    }
  }
  const float scaling = 1.0f / std::sqrt((float)AHD);
  std::vector<float> ao(T * QD, 0.0f), sc(T);
  for (int h = 0; h < AH; ++h) {
    const int kvh = h / AREP;
    for (int t = 0; t < T; ++t) {
      const float *qv = &q[(t * AH + h) * AHD];
      float mx = -1e30f;
      for (int u = 0; u <= t; ++u) {
        const float *kv = &k[(u * AKV + kvh) * AHD];
        float s = 0.0f;
        for (int d = 0; d < AHD; ++d) s += qv[d] * kv[d];
        sc[u] = s * scaling; mx = std::max(mx, sc[u]);
      }
      float sum = 0.0f;
      for (int u = 0; u <= t; ++u) { sc[u] = std::exp(sc[u] - mx); sum += sc[u]; }
      float *o = &ao[t * QD + h * AHD];
      for (int u = 0; u <= t; ++u) {
        float w = sc[u] / sum; const float *vv = &v[(u * AKV + kvh) * AHD];
        for (int d = 0; d < AHD; ++d) o[d] += w * vv[d];
      }
    }
  }
  std::vector<float> gated(T * QD);
  for (int t = 0; t < T; ++t)
    for (int h = 0; h < AH; ++h)
      for (int d = 0; d < AHD; ++d)
        gated[t * QD + h * AHD + d] = ao[t * QD + h * AHD + d] * sigmoidf(gate[(t * AH + h) * AHD + d]);
  Tensor t_out = makeT(gated, T, QD).dot(Wo, false, true);
  const float *po = t_out.getData<float>();
  return std::vector<float>(po, po + T * HID);
}

static std::vector<float> moe_forward(const std::string &pfx, const std::vector<float> &x) {
  const int T = g_T;
  const std::string g = pfx + "w_mlp_";
  Tensor X = makeT(x, T, HID);
  Tensor Wr = makeT(loadBin(g + "gate_weight"), E, HID);
  auto wgu = loadBin(g + "experts_gate_up_proj");
  auto wdn = loadBin(g + "experts_down_proj");
  Tensor Wg = makeT(loadBin(g + "shared_expert_gate_proj_weight"), SINTER, HID);
  Tensor Wu = makeT(loadBin(g + "shared_expert_up_proj_weight"), SINTER, HID);
  Tensor Wd = makeT(loadBin(g + "shared_expert_down_proj_weight"), HID, SINTER);
  Tensor Wgl = makeT(loadBin(g + "shared_expert_gate_weight"), 1, HID);
  const float *px = X.getData<float>();
  Tensor t_logits = X.dot(Wr, false, true);
  const float *pl = t_logits.getData<float>();
  std::vector<float> routed(T * HID, 0.0f), gu(2 * INTER), hact(INTER);
  for (int t = 0; t < T; ++t) {
    const float *lg = &pl[t * E];
    float mx = lg[0];
    for (int e = 1; e < E; ++e) mx = std::max(mx, lg[e]);
    std::vector<float> pr(E); float sum = 0.0f;
    for (int e = 0; e < E; ++e) { pr[e] = std::exp(lg[e] - mx); sum += pr[e]; }
    for (int e = 0; e < E; ++e) pr[e] /= sum;
    std::vector<int> ord(E);
    for (int e = 0; e < E; ++e) ord[e] = e;
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return pr[a] != pr[b] ? pr[a] > pr[b] : a < b; });
    float ts = 0.0f;
    for (int j = 0; j < KK; ++j) ts += pr[ord[j]];
    for (int j = 0; j < KK; ++j) {
      const int e = ord[j]; const float w = pr[e] / ts; const float *xr = &px[t * HID];
      const float *guw = &wgu[(size_t)e * (2 * INTER) * HID];
      for (int r = 0; r < 2 * INTER; ++r) { float acc = 0.0f; for (int h = 0; h < HID; ++h) acc += xr[h] * guw[(size_t)r * HID + h]; gu[r] = acc; }
      for (int i = 0; i < INTER; ++i) hact[i] = siluf(gu[i]) * gu[INTER + i];
      const float *dnw = &wdn[(size_t)e * HID * INTER];
      for (int hd = 0; hd < HID; ++hd) { float acc = 0.0f; for (int i = 0; i < INTER; ++i) acc += hact[i] * dnw[(size_t)hd * INTER + i]; routed[t * HID + hd] += w * acc; }
    }
  }
  Tensor t_sg = X.dot(Wg, false, true), t_su = X.dot(Wu, false, true);
  const float *psg = t_sg.getData<float>(), *psu = t_su.getData<float>();
  std::vector<float> sact(T * SINTER);
  for (int i = 0; i < T * SINTER; ++i) sact[i] = siluf(psg[i]) * psu[i];
  Tensor t_sp = makeT(sact, T, SINTER).dot(Wd, false, true);
  const float *psp = t_sp.getData<float>();
  Tensor t_gl = X.dot(Wgl, false, true);
  const float *pgl = t_gl.getData<float>();
  std::vector<float> out(T * HID);
  for (int t = 0; t < T; ++t) { float sg = sigmoidf(pgl[t]); for (int hd = 0; hd < HID; ++hd) out[t * HID + hd] = routed[t * HID + hd] + sg * psp[t * HID + hd]; }
  return out;
}

static std::vector<float> add(const std::vector<float> &a, const std::vector<float> &b) {
  std::vector<float> o(a.size());
  for (size_t i = 0; i < a.size(); ++i) o[i] = a[i] + b[i];
  return o;
}

// one decoder layer: in_norm -> mixer -> +res -> post_norm -> MoE -> +res
static std::vector<float> decoder(const std::string &pfx, bool is_gdn, const std::vector<float> &in) {
  auto norm1 = rmsnorm_1plus(in, loadBin(pfx + "w_input_layernorm_weight"));
  auto mix = is_gdn ? gdn_forward(pfx, norm1) : attn_forward(pfx, norm1);
  auto after_mix = add(in, mix);
  auto norm2 = rmsnorm_1plus(after_mix, loadBin(pfx + "w_post_attention_layernorm_weight"));
  auto moe = moe_forward(pfx, norm2);
  return add(after_mix, moe);
}

int main() {
  std::cout << "=== P3c Qwen3_5Moe FULL e2e (embed->2 layers->norm->lm_head) vs goldens ===\n\n";
  const char *ed = std::getenv("E2E_DIR");
  g_dir = ed ? ed : "/home/aisjetson/jijoongmoon/attn_p3/e2e_bin";
  auto ids_f = loadBin("input_ids");
  g_T = (int)ids_f.size();
  std::cout << "S(tokens)=" << g_T << " vocab=" << VOCAB << "\n";

  // embed lookup
  auto embed = loadBin("embed"); // [VOCAB, HID]
  std::vector<float> h(g_T * HID);
  for (int t = 0; t < g_T; ++t) {
    int id = (int)std::lround(ids_f[t]);
    for (int d = 0; d < HID; ++d) h[t * HID + d] = embed[(size_t)id * HID + d];
  }
  check("embed out (hs_0)", h, loadBin("hs_0"));

  // decoder layers (threaded residual stream), dispatched by layer_types.
  // HF hidden_states are collected BEFORE each layer; the last tuple entry is
  // the FINAL-NORMED output.  0 = linear_attention (GDN), 1 = full_attention.
  auto ltypes = loadBin("layer_types");
  const int NLAYERS = (int)ltypes.size();
  std::cout << "layers=" << NLAYERS << " [";
  for (int i = 0; i < NLAYERS; ++i) std::cout << (ltypes[i] < 0.5f ? "G" : "A");
  std::cout << "]\n";
  for (int li = 0; li < NLAYERS; ++li) {
    const bool is_gdn = ltypes[li] < 0.5f;
    h = decoder("l" + std::to_string(li) + "_", is_gdn, h);
    if (li < NLAYERS - 1)
      check("layer" + std::to_string(li) + " out (hs_" + std::to_string(li + 1) + ")",
            h, loadBin("hs_" + std::to_string(li + 1)), 1e-4f); // chained FP drift
  }

  // final norm -> lm_head -> logits
  auto fin = rmsnorm_1plus(h, loadBin("final_norm"));
  check("final_norm(last) (hs_" + std::to_string(NLAYERS) + ")", fin,
        loadBin("hs_" + std::to_string(NLAYERS)), 1e-4f);
  Tensor Wlm = makeT(loadBin("lm_head"), VOCAB, HID);
  Tensor t_logits = makeT(fin, g_T, HID).dot(Wlm, false, true); // [.,.,T,VOCAB]
  const float *pl = t_logits.getData<float>();
  std::vector<float> logits(pl, pl + g_T * VOCAB);
  check("logits (e2e vs HF)", logits, loadBin("logits"), 1e-4f);

  // also report argmax-token agreement (greedy-decode parity)
  auto ref = loadBin("logits");
  int mismatch = 0;
  for (int t = 0; t < g_T; ++t) {
    int am = 0, ar = 0;
    for (int e = 1; e < VOCAB; ++e) {
      if (logits[t * VOCAB + e] > logits[t * VOCAB + am]) am = e;
      if (ref[t * VOCAB + e] > ref[t * VOCAB + ar]) ar = e;
    }
    if (am != ar) ++mismatch;
  }
  std::cout << "argmax-token agreement: " << (g_T - mismatch) << "/" << g_T
            << (mismatch == 0 ? "  [PASS]" : "  [FAIL]") << "\n";
  if (mismatch) ++g_fail;

  std::cout << "\n=== " << (g_fail == 0 ? "ALL CHECKS PASS" : "FAILURES") << " (" << g_fail << " failed) ===\n";
  return g_fail == 0 ? 0 : 1;
}

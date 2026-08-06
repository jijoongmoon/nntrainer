// SPDX-License-Identifier: Apache-2.0
/**
 * P3a — nntrainer CPU Qwen3_5Moe full-attention layer reference + golden check.
 *
 * Reproduces the qwen3_5_moe full-attention layer (output-gated attention with
 * partial RoPE + per-head QK-norm + GQA), stage by stage, using nntrainer::Tensor
 * for the q/k/v/o GEMMs and explicit loops for QK-norm, RoPE, SDPA, and gating.
 * Validates every stage against the P3a goldens (<1e-5).
 *
 * Ground truth: transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
 *   Qwen3_5MoeAttention.forward + apply_rotary_pos_emb + eager_attention_forward
 *   q_proj -> [.,nH,hd*2]; chunk -> query, gate;  q=q_norm(query), k=k_norm(k_proj)
 *   QK-norm = Qwen3_5MoeRMSNorm: x*rsqrt(mean(x^2)+eps) * (1 + weight)   [head_dim]
 *   partial RoPE (first rotary_dim of head_dim; cos/sin from HF rotary, text path)
 *   GQA eager: softmax(QK^T*scale + causal)·V ; attn_out *= sigmoid(gate); o_proj
 *
 * Dims from env ATT_* ; cos/sin are consumed from goldens (mRoPE handled by HF).
 * Build+run: Applications/CausalLM/attn_ref/run_attn_p3a.sh
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

static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

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
static int envi(const char *k, int dflt) {
  const char *e = std::getenv(k);
  return e ? std::atoi(e) : dflt;
}

// RMSNorm over head_dim with the (1 + weight) scale (Qwen3_5MoeRMSNorm).
static void rmsnorm_head(float *row, int hd, const float *w, float eps) {
  float var = 0.0f;
  for (int d = 0; d < hd; ++d)
    var += row[d] * row[d];
  var /= hd;
  const float inv = 1.0f / std::sqrt(var + eps);
  for (int d = 0; d < hd; ++d)
    row[d] = row[d] * inv * (1.0f + w[d]);
}

// partial RoPE on a head vector x[hd] using cos/sin[rotary_dim] (in place).
static void rope_head(float *x, int hd, int rot, const float *cos,
                      const float *sin) {
  const int half = rot / 2;
  std::vector<float> xr(rot);
  for (int i = 0; i < rot; ++i)
    xr[i] = x[i];
  for (int i = 0; i < rot; ++i) {
    const float rh = (i < half) ? -xr[i + half] : xr[i - half]; // rotate_half
    x[i] = xr[i] * cos[i] + rh * sin[i];
  }
  // x[rot:] (pass-through) untouched
}

int main() {
  std::cout << "=== P3a Qwen3_5Moe full-attention CPU reference vs goldens ===\n";
  const int B = envi("ATT_B", 1);
  const int S = envi("ATT_S", 6);
  const int nH = envi("ATT_NH", 4);
  const int nKV = envi("ATT_NKV", 2);
  const int hd = envi("ATT_HD", 16);
  const int HID = envi("ATT_HID", 32);
  const int ROT = envi("ATT_ROT", 4);
  const float eps = 1e-6f;
  const int rep = nH / nKV;
  const int T = B * S;
  const int QD = nH * hd; // attn dim
  const char *ed = std::getenv("ATT_DIR");
  g_dir = ed ? ed : "/home/aisjetson/jijoongmoon/attn_p3/bin";
  std::cout << "B=" << B << " S=" << S << " nH=" << nH << " nKV=" << nKV
            << " hd=" << hd << " HID=" << HID << " ROT=" << ROT
            << "  dir=" << g_dir << "\n\n";

  Tensor H = makeT(loadBin("hidden"), 1, 1, T, HID);
  Tensor Wq = makeT(loadBin("w_q"), 1, 1, nH * hd * 2, HID);
  Tensor Wk = makeT(loadBin("w_k"), 1, 1, nKV * hd, HID);
  Tensor Wv = makeT(loadBin("w_v"), 1, 1, nKV * hd, HID);
  Tensor Wo = makeT(loadBin("w_o"), 1, 1, HID, QD);
  std::vector<float> wqn = loadBin("w_qn"), wkn = loadBin("w_kn");
  std::vector<float> cosb = loadBin("cos"), sinb = loadBin("sin"); // [T,ROT]

  // --- projections ---
  Tensor t_q = H.dot(Wq, false, true); // [.,.,T, nH*hd*2]
  Tensor t_k = H.dot(Wk, false, true); // [.,.,T, nKV*hd]
  Tensor t_v = H.dot(Wv, false, true);
  const float *pq = t_q.getData<float>();
  const float *pk = t_k.getData<float>();
  const float *pv = t_v.getData<float>();

  // --- split q into [query|gate] per head; q_norm; rope ---
  std::vector<float> q(T * nH * hd), gate(T * nH * hd), qrope(T * nH * hd);
  for (int tk = 0; tk < T; ++tk)
    for (int h = 0; h < nH; ++h) {
      const float *blk = &pq[tk * (nH * hd * 2) + h * (hd * 2)];
      float *qh = &q[(tk * nH + h) * hd];
      float *gh = &gate[(tk * nH + h) * hd];
      for (int d = 0; d < hd; ++d) {
        qh[d] = blk[d];          // first half = query
        gh[d] = blk[hd + d];     // second half = gate
      }
      rmsnorm_head(qh, hd, wqn.data(), eps);
    }
  // --- k_norm + v (per kv-head) ---
  std::vector<float> k(T * nKV * hd), v(T * nKV * hd), krope(T * nKV * hd);
  for (int tk = 0; tk < T; ++tk)
    for (int j = 0; j < nKV; ++j) {
      float *kh = &k[(tk * nKV + j) * hd];
      float *vh = &v[(tk * nKV + j) * hd];
      for (int d = 0; d < hd; ++d) {
        kh[d] = pk[tk * (nKV * hd) + j * hd + d];
        vh[d] = pv[tk * (nKV * hd) + j * hd + d];
      }
      rmsnorm_head(kh, hd, wkn.data(), eps);
    }
  // --- partial RoPE (q per head, k per kv-head) ---
  qrope = q;
  krope = k;
  for (int bi = 0; bi < B; ++bi)
    for (int t = 0; t < S; ++t) {
      const int tk = bi * S + t;
      const float *cs = &cosb[tk * ROT];
      const float *sn = &sinb[tk * ROT];
      for (int h = 0; h < nH; ++h)
        rope_head(&qrope[(tk * nH + h) * hd], hd, ROT, cs, sn);
      for (int j = 0; j < nKV; ++j)
        rope_head(&krope[(tk * nKV + j) * hd], hd, ROT, cs, sn);
    }

  // --- GQA causal SDPA ---
  const float scaling = 1.0f / std::sqrt((float)hd);
  std::vector<float> ao(T * QD, 0.0f); // [T, nH*hd]
  std::vector<float> scores(S);
  for (int bi = 0; bi < B; ++bi)
    for (int h = 0; h < nH; ++h) {
      const int kvh = h / rep;
      for (int t = 0; t < S; ++t) {
        const float *qv = &qrope[((bi * S + t) * nH + h) * hd];
        float mx = -1e30f;
        for (int u = 0; u <= t; ++u) {
          const float *kv = &krope[((bi * S + u) * nKV + kvh) * hd];
          float s = 0.0f;
          for (int d = 0; d < hd; ++d)
            s += qv[d] * kv[d];
          s *= scaling;
          scores[u] = s;
          mx = std::max(mx, s);
        }
        float sum = 0.0f;
        for (int u = 0; u <= t; ++u) {
          scores[u] = std::exp(scores[u] - mx);
          sum += scores[u];
        }
        float *out = &ao[(bi * S + t) * QD + h * hd];
        for (int u = 0; u <= t; ++u) {
          const float w = scores[u] / sum;
          const float *vv = &v[((bi * S + u) * nKV + kvh) * hd];
          for (int d = 0; d < hd; ++d)
            out[d] += w * vv[d];
        }
      }
    }

  // --- output gate + o_proj ---
  std::vector<float> gated(T * QD);
  for (int tk = 0; tk < T; ++tk)
    for (int h = 0; h < nH; ++h)
      for (int d = 0; d < hd; ++d) {
        const int idx = tk * QD + h * hd + d;
        gated[idx] = ao[idx] * sigmoidf(gate[(tk * nH + h) * hd + d]);
      }
  Tensor t_gated = makeT(gated, 1, 1, T, QD);
  Tensor t_out = t_gated.dot(Wo, false, true); // [.,.,T,HID]
  const float *pout = t_out.getData<float>();
  std::vector<float> out(pout, pout + T * HID);

  // --- checks (golden internals are token-major [T,heads,hd]) ---
  check("gate", gate, loadBin("s_gate"));
  check("q (qnorm)", q, loadBin("s_q"));
  check("k (knorm)", k, loadBin("s_k"));
  check("v", v, loadBin("s_v"));
  check("q_rope", qrope, loadBin("s_q_rope"));
  check("k_rope", krope, loadBin("s_k_rope"));
  check("attn_out (pre-gate)", ao, loadBin("s_attn_out"));
  check("gated", gated, loadBin("s_gated"));
  check("out (layer output)", out, loadBin("out"));

  std::cout << "\n=== " << (g_fail == 0 ? "ALL CHECKS PASS" : "FAILURES") << " ("
            << g_fail << " failed) ===\n";
  return g_fail == 0 ? 0 : 1;
}

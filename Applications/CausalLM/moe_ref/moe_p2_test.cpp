// SPDX-License-Identifier: Apache-2.0
/**
 * P2 — nntrainer CPU Qwen3_5MoeSparseMoeBlock reference + golden validation.
 *
 * Reproduces the qwen3_5_moe MoE block (routed top-k experts + always-on shared
 * expert gated by sigmoid), stage by stage, using nntrainer::Tensor for the
 * GEMMs and explicit loops for routing / top-k / expert dispatch / combine.
 * Validates every stage against the P2 goldens (<1e-5).
 *
 * Ground truth: transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
 *   Qwen3_5MoeSparseMoeBlock.forward:
 *     out = routed + sigmoid(shared_gate(x)) * shared_mlp(x)
 *   router: probs=softmax(x@Wr^T, fp32); topk(probs,k); w = topv/topv.sum  (always renorm)
 *   routed e: gate,up = chunk(x@gate_up_proj[e]^T, 2); silu(gate)*up; @down_proj[e]^T; *w
 *   shared : SwiGLU(shared_expert_intermediate_size)
 *
 * Dims (T,E,K,HID,INTER,SINTER) + golden dir from env MOE_*; defaults to the
 * dumped tiny case. Build+run: Applications/CausalLM/moe_ref/run_moe_p2.sh
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
static inline float siluf(float x) { return x * sigmoidf(x); }

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

int main() {
  std::cout << "=== P2 Qwen3_5Moe SparseMoeBlock CPU reference vs goldens ===\n";
  const int T = envi("MOE_T", 6);
  const int E = envi("MOE_E", 8);
  const int K = envi("MOE_K", 2);
  const int HID = envi("MOE_HID", 32);
  const int INTER = envi("MOE_INTER", 16);
  const int SINTER = envi("MOE_SINTER", 16);
  const char *ed = std::getenv("MOE_DIR");
  g_dir = ed ? ed : "/home/aisjetson/jijoongmoon/moe_p2/bin";
  std::cout << "T=" << T << " E=" << E << " K=" << K << " HID=" << HID
            << " INTER=" << INTER << " SINTER=" << SINTER << "  dir=" << g_dir
            << "\n\n";

  // --- weights + input ---
  Tensor H = makeT(loadBin("hidden"), 1, 1, T, HID); // [.,.,T,HID]
  Tensor Wr = makeT(loadBin("w_router"), 1, 1, E, HID);
  std::vector<float> wgu = loadBin("w_gate_up");   // [E, 2*INTER, HID]
  std::vector<float> wdn = loadBin("w_down");      // [E, HID, INTER]
  Tensor WshG = makeT(loadBin("w_sh_gate"), 1, 1, SINTER, HID);
  Tensor WshU = makeT(loadBin("w_sh_up"), 1, 1, SINTER, HID);
  Tensor WshD = makeT(loadBin("w_sh_down"), 1, 1, HID, SINTER);
  Tensor WshGate = makeT(loadBin("w_sh_gate_lin"), 1, 1, 1, HID);
  const float *ph = H.getData<float>();

  // ====================== router ======================
  // logits = H @ Wr^T   [T,E]
  Tensor t_logits = H.dot(Wr, false, true); // [.,.,T,E]
  const float *pl = t_logits.getData<float>();
  std::vector<float> router_logits(pl, pl + T * E);

  // softmax over E (fp32), then top-k by prob, then renorm top-k weights
  std::vector<float> sel(T * K), wts(T * K);
  std::vector<float> probs(T * E);
  for (int t = 0; t < T; ++t) {
    const float *lg = &router_logits[t * E];
    float mx = lg[0];
    for (int e = 1; e < E; ++e)
      mx = std::max(mx, lg[e]);
    float sum = 0.0f;
    for (int e = 0; e < E; ++e) {
      float pe = std::exp(lg[e] - mx);
      probs[t * E + e] = pe;
      sum += pe;
    }
    for (int e = 0; e < E; ++e)
      probs[t * E + e] /= sum;
    // top-k: sort experts by prob desc, idx asc on tie (matches torch.topk)
    std::vector<int> order(E);
    for (int e = 0; e < E; ++e)
      order[e] = e;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
      float pa = probs[t * E + a], pb = probs[t * E + b];
      return pa != pb ? pa > pb : a < b;
    });
    float topsum = 0.0f;
    for (int j = 0; j < K; ++j)
      topsum += probs[t * E + order[j]];
    for (int j = 0; j < K; ++j) {
      sel[t * K + j] = (float)order[j];
      wts[t * K + j] = probs[t * E + order[j]] / topsum;
    }
  }

  // ====================== routed experts ======================
  std::vector<float> routed(T * HID, 0.0f);
  std::vector<float> gu(2 * INTER), hact(INTER);
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < K; ++j) {
      const int e = (int)sel[t * K + j];
      const float w = wts[t * K + j];
      const float *xrow = &ph[t * HID];
      const float *guw = &wgu[(size_t)e * (2 * INTER) * HID]; // [2*INTER,HID]
      for (int r = 0; r < 2 * INTER; ++r) {
        float acc = 0.0f;
        const float *wr = &guw[(size_t)r * HID];
        for (int h = 0; h < HID; ++h)
          acc += xrow[h] * wr[h];
        gu[r] = acc;
      }
      for (int i = 0; i < INTER; ++i)
        hact[i] = siluf(gu[i]) * gu[INTER + i]; // gate=first half, up=second
      const float *dnw = &wdn[(size_t)e * HID * INTER]; // [HID,INTER]
      float *orow = &routed[t * HID];
      for (int hd = 0; hd < HID; ++hd) {
        float acc = 0.0f;
        const float *wr = &dnw[(size_t)hd * INTER];
        for (int i = 0; i < INTER; ++i)
          acc += hact[i] * wr[i];
        orow[hd] += w * acc;
      }
    }

  // ====================== shared expert ======================
  // SwiGLU: silu(H@Wg^T) * (H@Wu^T) then @Wd^T
  Tensor t_sg = H.dot(WshG, false, true); // [.,.,T,SINTER]
  Tensor t_su = H.dot(WshU, false, true);
  const float *psg = t_sg.getData<float>();
  const float *psu = t_su.getData<float>();
  std::vector<float> sact(T * SINTER);
  for (int i = 0; i < T * SINTER; ++i)
    sact[i] = siluf(psg[i]) * psu[i];
  Tensor t_act = makeT(sact, 1, 1, T, SINTER);
  Tensor t_shared_pre = t_act.dot(WshD, false, true); // [.,.,T,HID]
  const float *pspre = t_shared_pre.getData<float>();
  std::vector<float> shared_pre(pspre, pspre + T * HID);

  // sigmoid gate (Linear hidden->1)
  Tensor t_gate = H.dot(WshGate, false, true); // [.,.,T,1]
  const float *pg = t_gate.getData<float>();
  std::vector<float> shared_gate(T), shared_out(T * HID);
  for (int t = 0; t < T; ++t) {
    shared_gate[t] = sigmoidf(pg[t]);
    for (int hd = 0; hd < HID; ++hd)
      shared_out[t * HID + hd] = shared_gate[t] * shared_pre[t * HID + hd];
  }

  // ====================== combine ======================
  std::vector<float> out(T * HID);
  for (int i = 0; i < T * HID; ++i)
    out[i] = routed[i] + shared_out[i];

  // ====================== checks ======================
  check("router_logits", router_logits, loadBin("s_router_logits"));
  check("selected_experts", sel, loadBin("s_selected_experts"));
  check("routing_weights", wts, loadBin("s_routing_weights"));
  check("routed_output (Qwen3-30B parity)", routed, loadBin("s_routed"));
  check("shared_pre (SwiGLU)", shared_pre, loadBin("s_shared_pre"));
  check("shared_gate (sigmoid)", shared_gate, loadBin("s_shared_gate"));
  check("shared_out (gated)", shared_out, loadBin("s_shared_out"));
  check("out (MoE block)", out, loadBin("out"));

  std::cout << "\n=== " << (g_fail == 0 ? "ALL CHECKS PASS" : "FAILURES") << " ("
            << g_fail << " failed) ===\n";
  return g_fail == 0 ? 0 : 1;
}

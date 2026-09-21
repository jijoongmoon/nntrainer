// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   kv_ring.h
 * @date   01 September 2026
 * @brief  Single source of truth for the sliding-window KV ring and the
 *         chunked-prefill size.
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details The ring rule is consumed by two independent translation units that
 * must not disagree:
 *
 *   - the MODEL side (models/transformer.h) sizes the KV placeholder and the
 *     KVCacheManager allocation -- how many PHYSICAL rows exist;
 *   - the LAYER side (layers/mha_core.cpp) modulo-maps the write position and
 *     the attention read view into those rows.
 *
 * A disagreement in the direction "model allocates Wcap, layer writes absolute"
 * is an out-of-bounds write, not a wrong answer, so the rule lives here and
 * both sides call it. Nothing in this header keeps state; every entry point is
 * a pure function of its arguments plus the process environment.
 */

#ifndef __CAUSALLM_KV_RING_H__
#define __CAUSALLM_KV_RING_H__

#include <algorithm>
#include <climits> // UINT_MAX -- the full-attention window sentinel
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <env_compat.h> // nntr_env_on (an auto-injected flag needs =0 to work)

namespace causallm {

/**
 * @brief Which reproducibility arm this process runs.
 * @details Bit-identical output is the BASELINE, not a flag. The three arms:
 *
 *   kPerLayer   (the DEFAULT, no environment)
 *               Per-layer arm selection on the Adreno image bundle: a
 *               FULL-attention layer takes the reproducible buffer/flash
 *               kernels, a window-bounded layer keeps the image path (and its
 *               ring). See imageAttnLayer() for the rule and the evidence.
 *   kAllLayers  (NNTR_DETERMINISTIC=1)
 *               The strongest arm: the buffer/flash kernels serve EVERY layer
 *               and the image bundle is off process-wide. Kept because it is
 *               the arm with the longest measurement history, and as the
 *               fallback if a pack's geometry defeats the per-layer split.
 *   kFast       (NNTR_ALLOW_NONDETERMINISTIC=1, or NNTR_DETERMINISTIC=0, or the
 *               pack's own nntr_config opt-out)
 *               The old throughput default: the image path serves every layer.
 *               NOT bit-reproducible past ~2K keys on this driver. The opt-out
 *               exists for a throughput-critical consumer that has accepted
 *               that; it announces itself once on stderr.
 *
 * Every consumer -- the mirror prebuild, the Q staging, the decode-RoPE gate,
 * the engage and the ring rule -- resolves the arm through this one function,
 * so they cannot disagree about which arm the process is on.
 */
enum class DetArm { kFast = 0, kPerLayer = 1, kAllLayers = 2 };

/**
 * @brief The pack's own opt-out, from nntr_config.json.
 * @details A pack that has accepted non-reproducible output for throughput says
 * so in its own config ("allow_nondeterministic": true), which the Transformer
 * base feeds here before any layer finalizes. It is weaker than an explicit
 * NNTR_DETERMINISTIC (the caller outranks the pack) and stronger than the
 * default. Reference semantics so the getter and the setter cannot drift.
 */
inline bool &packAllowsNondeterministic() {
  static bool allow = false;
  return allow;
}

/**
 * @brief Resolve the reproducibility arm (see DetArm).
 * @details NNTR_DETERMINISTIC is value-checked in BOTH directions, because it
 * is the pre-existing cross-lane knob: =1 asks for the strongest arm, =0 is the
 * explicit opt-out that used to be "just don't set it".
 * NNTR_ALLOW_NONDETERMINISTIC is the new, positively-named opt-out. Neither is
 * needed for the default.
 */
inline DetArm detArm() {
  DetArm arm = DetArm::kPerLayer; // the baseline: no environment needed
  const char *d = std::getenv("NNTR_DETERMINISTIC");
  if (d != nullptr && d[0] != '\0')
    // Explicit, both ways: =1 asks for the whole-process arm, =0 is the
    // pre-existing spelling of the opt-out and keeps working as one.
    arm =
      nntr_env_on("NNTR_DETERMINISTIC") ? DetArm::kAllLayers : DetArm::kFast;
  else if (nntr_env_on("NNTR_ALLOW_NONDETERMINISTIC") ||
           packAllowsNondeterministic())
    arm = DetArm::kFast;
  // Not cached: every input is a pure read, and a cache would make the answer
  // depend on WHO asked first -- which is exactly the failure mode the pack
  // opt-out (set at model construction, consumed at layer finalize) would hit,
  // and which would make the rule untestable across environments in one
  // process. The announcement, which must happen once, is the only state.
  if (arm == DetArm::kFast) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      std::fprintf(
        stderr,
        "[DETERMINISM] the FAST non-deterministic attention arm is selected "
        "(NNTR_ALLOW_NONDETERMINISTIC=1, NNTR_DETERMINISTIC=0, or the pack's "
        "nntr_config \"allow_nondeterministic\"). Output is NOT "
        "bit-reproducible "
        "run to run past ~2K keys on this driver. Unset it for the "
        "reproducible "
        "default.\n");
    }
  }
  return arm;
}

/**
 * @brief Has the caller asked for bit-identical output above everything else?
 * @details True for both reproducible arms; it is what the CUDA submission
 * policy, the cuBLAS FP32 math mode and the OpenCL subgroup reduction order
 * key on, and those want the guarantee whichever Adreno arm is in play. Only
 * the explicit opt-out turns it off.
 */
inline bool determinismFirst() { return detArm() != DetArm::kFast; }

/**
 * @brief Is the Adreno OHWI image-attention arm in play for ANY layer?
 * @details Value-checked: NNTR_KV_IMG_ATTN=0 disables, as before. This is the
 * PROCESS-level question -- "was the image bundle asked for, and can some layer
 * take it" -- which is what the shared program build, the RoPE LUT and the ring
 * rule need. Whether a GIVEN layer takes it is imageAttnLayer().
 *
 * Only the all-layers arm answers NO outright: there the buffer/flash kernels
 * serve every layer.
 */
inline bool imageAttnRequested() {
  const char *e = std::getenv("NNTR_KV_IMG_ATTN");
  if (e == nullptr || std::atoi(e) == 0)
    return false;
  if (detArm() == DetArm::kAllLayers) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      std::fprintf(
        stderr,
        "[DETERMINISM] NNTR_DETERMINISTIC=1: the buffer/flash attention arm "
        "serves EVERY layer (the Adreno image K/V arm is off process-wide). "
        "This is the strongest arm and the most expensive one; the default "
        "(no environment) routes only the full-attention layers here.\n");
    }
    return false;
  }
  return true;
}

/**
 * @brief The widest attention window whose image-path read is measured
 *        bit-reproducible on this driver, in KV rows.
 * @details A safety bound on the per-layer split, not a tuning knob. The split
 * assumes "window-bounded read span == reproducible", which is measured for
 * W=512 against a 4096-row prefill block (a ~4.6K-row bounded span, 0 events in
 * every windowed layer of every run pair). It is NOT measured for a pack whose
 * window is itself long enough to be a full-attention read in disguise, and
 * such a pack must not silently inherit the assumption -- so a window wider
 * than this falls to the buffer/flash arm like a full layer. 2048 is the
 * longest read the image arm is bit-reproducible over as an UNBOUNDED span
 * (0.3B 2047: 28/28 run pairs identical; 4095: divergence appears), i.e. the
 * conservative reading of the same evidence. NNTR_DET_IMG_WINDOW_MAX overrides
 * it for experiments.
 */
inline unsigned int detImageWindowMax() {
  // Not cached, for the same reason detArm() is not: the arm rule must be a
  // pure function of the environment at the moment it is asked, or the answer
  // depends on who asked first and the rule cannot be tested across
  // environments.
  const char *e = std::getenv("NNTR_DET_IMG_WINDOW_MAX");
  if (e != nullptr && e[0] != '\0') {
    char *end = nullptr;
    const long v = std::strtol(e, &end, 10);
    if (end != e && *end == '\0' && v >= 0)
      return (unsigned int)v;
  }
  return 2048u;
}

/**
 * @brief Does THIS layer take the Adreno image attention arm?
 * @param local_window_size the layer's props::SlidingWindow (UINT_MAX = full).
 * @param max_timestep the model's max sequence; a window at least this wide is
 *        a full-attention layer however it is spelled.
 * @details The per-layer rule, and the whole point of the default.
 *
 * MEASURED (Adreno 840, per-node output hashes over whole runs,
 * an earlier attribution): with the mirror image-written and the attention
 * split removed, 20 of 21 diverging run pairs have their FIRST differing node
 * in a FULL-attention layer's attention op, and a window-bounded layer NEVER
 * originates a divergence -- including with the ring forced off, so that its
 * mirror is full-context height and only its READ stays short. The failing
 * thing is a read_imageui pass over many texel rows of a buffer-backed
 * image2d, and its rate grows with the number of rows one pass walks:
 *   full-attention read span 2047 rows: 0 events in 15 pairs
 *                            4095     : 1 node in 1 of 3 runs
 *                            8191     : 32-462 nodes
 *                            16383    : 1-769 nodes
 * A window-bounded layer's pass walks ~W + block rows whatever the context, so
 * it stays in the clean regime at every context length. That is why the arm can
 * be chosen per layer instead of per process, and why doing so costs a small
 * fraction of the all-layers arm: the measured models are mostly windowed (0.3B
 * full at 7,15 of 16; 1.5B at 5,11,17,23 of 24; gemma4 3 of 15).
 *
 * The decision is STATIC per layer -- taken once at finalize, never flipped per
 * call. It has to be: on the image arm this step's K/V is written into the
 * image, not the buffer that aliases it, so a layer that served one call from
 * the image and the next from the flash kernels would read a mirror the buffer
 * does not have. Per-layer is safe for exactly the reason per-call is not: each
 * layer owns its own K/V cache and its own mirror, so the two arms never share
 * a store.
 */
inline bool imageAttnLayer(size_t local_window_size,
                           unsigned int max_timestep) {
  if (!imageAttnRequested())
    return false; // bundle off, or the all-layers arm
  if (detArm() == DetArm::kFast)
    return true; // the old throughput default: every layer
  // kPerLayer: the read span must be bounded by the window, and the window must
  // be no wider than the span the image arm is measured reproducible over.
  if (local_window_size == (size_t)UINT_MAX ||
      (max_timestep != 0u && local_window_size >= (size_t)max_timestep))
    return false; // full attention: the read walks every key
  return local_window_size <= (size_t)detImageWindowMax();
}

/**
 * @brief Say once, on stderr, which reproducibility arm this process resolved.
 * @details The arm changes both the numbers and (on some packs) the token
 * sequence, so a log that does not say which arm produced it is not evidence.
 * Called from the first mha_core finalize; idempotent. The marker string is
 * also what a deployment check greps for to prove the deployed .so is this
 * build.
 */
inline void announceDetArmOnce() {
  static bool done = false;
  if (done)
    return;
  done = true;
  const DetArm a = detArm();
  const char *name = a == DetArm::kFast        ? "fast-NONDETERMINISTIC"
                     : a == DetArm::kAllLayers ? "deterministic-all-layers"
                                               : "deterministic-per-layer";
  std::fprintf(stderr,
               "[DETERMINISM] DETDEF_MARKER_B arm=%s image_bundle=%d "
               "window_max=%u\n",
               name, (int)imageAttnRequested(), detImageWindowMax());
}

/**
 * @brief Whether the engine this process resolved can host the ring at all.
 * @details The ring is only correct where the attention kernels modulo-map the
 * cache row, which today means the GPU attention paths. The host CPU attention
 * fallback walks absolute rows, so a CPU run must keep the linear cache.
 */
inline bool kvRingEngineEligible() {
  const char *e = std::getenv("NNTR_ENGINE");
  if (e != nullptr && std::string(e) == "cpu")
    return false;
  if (e != nullptr && std::string(e) == "cuda")
    return true;
#if defined(ENABLE_OPENCL)
  return true; // no explicit engine + an OpenCL build == the gpu engine
#else
  return false;
#endif
}

/**
 * @brief Whether a ring-AWARE attention arm is reachable in this configuration.
 * @details mha_core resolves attention through a cascade, and only three of its
 * arms take the ring capacity and read row (n % cap):
 * flash_attention_prefill_f16_cl, flash_decode_f16_cl and
 * cuda_attention_interleaved_fp16; the Adreno OHWI image arm reaches the same
 * result without a modulo (a sliding mirror, see below). The remaining arms
 * (the two_conv family, the OHWI-direct path, and the host compute_kcaches /
 * gemm_attention fallback) index the cache linearly from the LOGICAL key count,
 * so pointing them at a Wcap-high buffer reads past its end.
 *
 * The arms that do map the row are behind env gates that are readable here, so
 * the ring refuses to turn on unless one of them is actually selectable. This
 * is deliberately evaluated BEFORE any allocation happens: the answer feeds
 * kvRingCap(), which both sides use, so a refusal leaves the ordinary
 * full-height linear cache in place rather than leaving a ringed allocation
 * with a linear reader.
 *
 * Runtime failure of the selected arm still drops into a non-ring arm, so
 * mha_core additionally refuses those at dispatch time; this function only
 * removes the statically-knowable mismatches.
 */
inline bool kvRingArmAvailable() {
  const char *e = std::getenv("NNTR_ENGINE");
  if (e != nullptr && std::string(e) == "cuda")
    return nntr_env_on("NNTR_CUDA_ATTN"); // cuda_attention_interleaved_fp16
#if defined(ENABLE_OPENCL)
  // The two OpenCL flash arms sit in the CONCAT-layout GPU attention block,
  // which NNTR_MHA_GPU alone opens. NNTR_KV_OHWI is the opposite of a
  // precondition: it switches the K write to the per-head OHWI scatter, which
  // places rows by ABSOLUTE position against the plane height (a heap overwrite
  // on a Wcap-high plane, reproduced on Xe), and its readers -- the OHWI-direct
  // two_conv arm and the OHWI->concat gather -- are linear too. mha_core
  // presence-checks that variable, so presence is what disqualifies here. The
  // two_conv image arm (NNTR_MHA_GPU_IMG) is linear too.
  if (!nntr_env_on("NNTR_MHA_GPU"))
    return false;
  if (std::getenv("NNTR_KV_OHWI") != nullptr)
    return false;
  if (nntr_env_on("NNTR_MHA_GPU_IMG"))
    return false;
  if (detArm() == DetArm::kAllLayers &&
      std::getenv("NNTR_KV_IMG_ATTN") != nullptr &&
      std::atoi(std::getenv("NNTR_KV_IMG_ATTN")) != 0)
    // [determinism] The image bundle under the ALL-LAYERS arm falls back to the
    // buffer/flash kernels for everything (imageAttnRequested), and the only
    // configuration measured bit-reproducible there is the one WITHOUT the
    // ring: the flash kernels do modulo-map the row, but a 2048-row ring under
    // the image bundle was measured WRONG on Adreno 840. The reproducible
    // configuration is image off AND ring off, so refuse it here rather than
    // leave a ringed allocation in a profile whose whole purpose is a
    // guarantee.
    //
    // The DEFAULT per-layer arm does NOT refuse: there the windowed layers are
    // still on the image path with their sliding mirrors, which is the ring's
    // validated reader, and the full-attention layers are not ringed anyway
    // (kvRingCap returns 0 for them), so each layer's store and reader match.
    // Keeping the ring is what makes the default a memory win as well.
    return false;
  if (imageAttnRequested()) {
    // The Adreno OHWI image arm serves a ringed layer through a SLIDING
    // mirror: the per-layer K/V mirror is ring-cap rows high and holds the
    // absolute rows [base, base + rows), re-based (and back-filled from the
    // ringed cache) whenever the window moves past it, so the image kernels
    // keep their linear addressing over a bounded height. Two things still
    // close it: the opt-in staged chain (NNTR_KV_STAGE), whose mirrors are
    // the ONLY K/V store during prefill, so there is no cache to back-fill
    // from; and NNTR_KV_IMG_RING=0, the control arm that restores the linear
    // full-height cache under the image bundle.
    if (std::getenv("NNTR_KV_STAGE") != nullptr &&
        std::getenv("NNTR_NO_KV_STAGE") == nullptr)
      return false;
    const char *ir = std::getenv("NNTR_KV_IMG_RING");
    if (ir != nullptr && ir[0] == '0')
      return false;
  }
  return true;
#else
  return false;
#endif
}

/**
 * @brief Whether the KV ring is enabled.
 * @param model_default what the MODEL asks for when the environment is silent.
 * @details NNTR_KV_WINDOW_RING, when set, is the whole answer to "is the ring
 * requested": '0' keeps the linear cache (the opt-out, bit-identical to the
 * pre-ring path) and anything else requests the ring. When it is unset the
 * request is the model's own default -- false for every model that has not
 * said otherwise, so those keep the linear cache exactly as before; a model
 * whose sliding layers are the point of its architecture (a dense stack that
 * is 5/6 sliding at W=512) declares true and gets the ring with no variable
 * set. The model feeds the same boolean to both consumers (its own sizing
 * through Transformer::kvRingByDefault(), mha_core through the
 * `kv_window_ring` property), so the two sides still cannot disagree.
 *
 * A request is granted only where it is also correct -- the engine can host it
 * (kvRingEngineEligible) and a ring-aware attention arm is reachable
 * (kvRingArmAvailable). A refused EXPLICIT request is reported once so the
 * reason is visible instead of showing up as a silent performance or memory
 * difference; a refused model default is not an error (the cpu engine and the
 * Adreno image-attention bundle simply keep the linear cache).
 */
inline bool kvRingEnabled(bool model_default = false) {
  const char *req = std::getenv("NNTR_KV_WINDOW_RING");
  const bool explicit_req = (req != nullptr && req[0] != '\0');
  if (explicit_req ? !nntr_env_on("NNTR_KV_WINDOW_RING") : !model_default)
    return false;
  const bool ok = kvRingEngineEligible() && kvRingArmAvailable();
  if (!ok && explicit_req) {
    static bool reported = false;
    if (!reported) {
      reported = true;
      std::fprintf(stderr,
                   "[kv-window-ring] NNTR_KV_WINDOW_RING is set but no "
                   "ring-aware attention arm resolves in this configuration "
                   "(engine_eligible=%d arm_available=%d); keeping the linear "
                   "full-height KV cache. The ring needs NNTR_MHA_GPU=1 "
                   "without NNTR_KV_OHWI / the image-attention arms on OpenCL, "
                   "or NNTR_CUDA_ATTN=1 on NNTR_ENGINE=cuda.\n",
                   (int)kvRingEngineEligible(), (int)kvRingArmAvailable());
    }
  }
  return ok;
}

/**
 * @brief Per-layer structural preconditions for the ring, evaluated the same
 *        way on the model side and on the layer side.
 * @param attention_sink the model uses the attention-sink attention variant,
 *        which reads the cache through the host compute path (no modulo map).
 * @param external_cache mha_core's external (5-input) KV cache mode; the
 *        layer-internal cache is allocated at full max_seq and is not ringed.
 * @details The int8 KV cache is likewise allocated at full max_seq and written
 * with absolute rows, so NNTR_KV_INT8 disqualifies the ring on both sides. That
 * condition used to exist only on the layer side, which left the model free to
 * ring the ALLOCATION while the layer wrote absolute rows into it.
 */
inline bool kvRingLayerEligible(bool attention_sink, bool external_cache) {
  if (attention_sink || !external_cache)
    return false;
  if (std::getenv("NNTR_KV_INT8") != nullptr)
    return false;
  return true;
}

/**
 * @brief The prefill BLOCK the measured sweep picks, in query rows.
 * @details One constant, read by both halves of the block policy so the two
 * cannot drift:
 *
 *   - requestedPrefillChunk() below -- how many rows one prefill launch feeds;
 *   - Transformer::prefillPlaneFor() -- how tall the activation plane is built,
 *     which is the CEILING on the above (a chunk is fed at row 0 of the plane,
 *     so effectivePrefillChunk() clamps the request to the plane).
 *
 * 4096 because the equal-thermal ring-on sweep is monotone in the block but
 * with a poor marginal ratio past 4096 (the next step up buys under a percent
 * of prefill for another GB of working set), and the CUDA tensor-core GEMMs
 * want a large block anyway -- so one number, no backend branch. It is 64-row
 * aligned, which the qk m-tiling and the attention score sub-blocking both
 * assume of any block they are handed.
 */
inline constexpr unsigned int kPrefillBlockCap = 4096u;

/**
 * @brief Requested prefill chunk size (0 = no chunking / single-block prefill).
 * @details An explicit NNTR_PREFILL_CHUNK always wins (user override, per-GPU
 * tuning); a non-positive or unparseable value is REJECTED (treated as unset)
 * rather than wrapped into a ~4e9 unsigned, which the (W/C + 2) * C ring
 * arithmetic would have consumed. Otherwise, chunking follows the ring:
 * chunking is what bounds a launch's live key span, so the ring picks the
 * chunk, 4096 for every backend. The equal-thermal ring-on sweep is monotone in
 * the chunk but with a poor marginal ratio past 4096 (the next step up buys
 * under a percent of prefill for another GB of working set), and the CUDA
 * tensor-core GEMMs want a large chunk anyway -- so one constant, no backend
 * branch.
 *
 * This is the REQUEST, not what the prefill actually runs: a chunk cannot
 * exceed the activation-plane height it has to fit in. Use
 * effectivePrefillChunk() (or Transformer::prefillChunk(), which calls it)
 * anywhere the answer feeds sizing or control flow.
 */
inline unsigned int requestedPrefillChunk(bool ring_model_default = false) {
  const char *pc = std::getenv("NNTR_PREFILL_CHUNK");
  if (pc != nullptr && pc[0] != '\0') {
    char *end = nullptr;
    const long v = std::strtol(pc, &end, 10);
    if (end != pc && *end == '\0' && v > 0)
      return static_cast<unsigned int>(v); // explicit override wins
    static bool reported = false;
    if (!reported) {
      reported = true;
      std::fprintf(stderr,
                   "[prefill-chunk] NNTR_PREFILL_CHUNK='%s' is not a positive "
                   "integer; ignoring it.\n",
                   pc);
    }
  }
  if (!kvRingEnabled(ring_model_default))
    return 0u; // chunking is auto-enabled only by the ring
  return kPrefillBlockCap;
}

/**
 * @brief The prefill chunk that actually runs, given the activation-plane
 *        height it is fed through (0 = no chunking).
 * @param plane_height INIT_SEQ_LEN -- the height of the plane one chunk is fed
 *        at row 0 of. 0 means "unknown", which leaves the request unclamped.
 * @details Every consumer must read the SAME clamped number: the prompt budget,
 * the prefill drive loop, and the ring capacity (Wcap is a multiple of the
 * chunk). They used to disagree, and sizing the ring off the unclamped request
 * (a 4096 request against a 1024-row plane) leaves Wcap up to 4x too large.
 */
inline unsigned int effectivePrefillChunk(unsigned int plane_height,
                                          bool ring_model_default = false) {
  const unsigned int c = requestedPrefillChunk(ring_model_default);
  if (c == 0u || plane_height == 0u)
    return c;
  return std::min(c, plane_height);
}

/**
 * @brief Whether a TALLER prefill plane (a bigger block) pays for itself on the
 *        attention arm this configuration resolves.
 * @details The plane is charged per run; the block it enables is only worth it
 * where a launch carries a fixed per-launch cost that fewer, bigger launches
 * amortize. Measured, both directions:
 *
 *   Adreno image arm (NNTR_KV_IMG_ATTN) -- WITHDRAWN. The prefill-TPS numbers
 *   this used to cite (0.3B 16K 4370 vs 3078 TPS, 1.5B 16K 1037 vs 776) were
 *   read off an instrument that stopped the prefill clock before the GPU queue
 *   drained, so a bigger block -- which leaves more work in flight -- credited
 *   its own tail to the first decode token. Re-measured with the boundary drain
 *   armed (NNTR_PERF_SPLIT_DRAIN=1, see performance_metrics.h), end to end for
 *   128 tokens, Adreno 840 at 1300 MHz, warm kernel cache, ids identical in
 *   every pair, two orders (block 1024 first and block 4096 first):
 *
 *     prompt          block 1024        block 4096        e2e     honest
 *     0.3B  4095   2413 / 2501 ms    2824 / 2656 ms    -5.8%   826 vs 1286 MiB
 *     0.3B  8191   3804 / 3806       3978 / 3947       -3.6%   858 vs 1031
 *     0.3B 16383   7642 / 7622       7816 / 7809       -2.4%   858 vs 1032
 *     1.5B  3925   5284 / 5183       5530 / 5490       -5.6%  1706 vs 2343
 *     1.5B  8103   9266 / 9293       9775 / 9697       -4.2%  1963 vs 2151
 *     1.5B 16203  21100             22068             -4.4%  1963 vs 2152
 *   (the 1.5B 16K row is the only same-clock pair that cell yields -- 128
 *   tokens on a 16K context throttles the GPU below 1300 MHz about half the
 *   time, so cells at 1100 MHz or less were discarded rather than compared.)
 *
 *   The big block is slower at every length on both packs and costs +140 to
 *   +637 MiB, and it is slower on TTFT too, partly because growing the plane
 *   rebuilds the graph and replays the weight load -- 188..578 ms of setup that
 *   the plane-1024 arm does not pay. So: no growth on OpenCL, and the earlier
 *   "the block dominates on the image arm" reading was the measurement, not the
 *   machine.
 *
 *   Intel Xe flash/XMX arm -- was already a LOSS, at two prompt lengths:
 *     1.5B 1899 tok  xmx 4427 vs 4366 TPS (+1.4%), host peak 1663 vs 978 MiB
 *                    dp4a 1582 vs 1548 TPS (+2.2%), 1721 vs 972 MiB
 *     1.5B 3925 tok  xmx 3878 vs 3909 TPS (-0.8%), 1853 vs 1122 MiB
 *   and with the drain armed the +2% goes away too: at 1919 tokens prefill is
 *   439 vs 440 ms (xmx) and 1057 vs 1092 ms (dp4a) with the plane grown or not,
 *   while the grow path's graph rebuild adds ~190 ms of TTFT.
 *
 *   CUDA tensor-core arm -- UNCHANGED here, deliberately. The claim it rests on
 *   (1.5B 1899-token prompt, 8516 vs 8293 TPS prefill, VRAM 1226 vs 1166 MiB)
 *   was not re-measured at a length that exercises the block: the host 1.5B
 *   pack's window is 2048, so a prompt long enough to need more than one block
 *   is truncated. What that length does show is 230 vs 231 ms of prefill with
 *   the plane grown or not and ~140 ms more TTFT from the rebuild, i.e. no
 *   prefill gain either -- so this arm is a candidate for the same treatment
 *   once a long-window CUDA pack is available to measure it on.
 *
 * NNTR_PREFILL_GROW forces the answer either way (1 = grow anyway, 0 = never),
 * so an arm stays one env away from an A/B rather than needing a build.
 */
inline bool prefillBlockPays() {
  if (const char *g = std::getenv("NNTR_PREFILL_GROW"))
    if (g[0] == '0' || g[0] == '1')
      return g[0] == '1';
  const char *e = std::getenv("NNTR_ENGINE");
  const std::string eng = (e != nullptr) ? std::string(e) : std::string();
  if (eng == "cuda")
    return nntr_env_on("NNTR_CUDA_ATTN");
  // Every OpenCL arm measured -- Adreno image, Intel Xe flash/XMX, Intel dp4a
  // -- is slower end to end AND larger with the plane grown, so the plane stays
  // at the height the pack shipped and the chunk clamps to it.
  // NNTR_PREFILL_GROW=1 opens it back up for an A/B.
  return false;
}

/**
 * @brief The activation-plane height (== the largest prefill block) a request
 *        of `prompt_tokens` tokens wants.
 * @param prompt_tokens the prompt this request prefills. 0 = unknown, which
 *        keeps `cur_plane` (no growth without a number to grow to).
 * @param cur_plane the plane the graph is currently built at (INIT_SEQ_LEN).
 * @param pack_plane the height the PACK shipped -- the floor, so no pack loses
 *        a plane it was tuned with.
 * @param max_seq_len the window, which bounds the block from above together
 *        with kPrefillBlockCap.
 * @details The pure arithmetic of the policy, free of the Transformer so it can
 * be pinned by a unit test. Rounds to the 64-row grid the qk m-tiling and the
 * attention score sub-blocking assume of any block, bounds it by
 * min(kPrefillBlockCap, max_seq_len), and never returns less than the pack's
 * own height. A prompt past the bound is NOT truncated -- it is fed in
 * bound-sized chunks (Transformer::prefillDriveChunk()).
 */
inline unsigned int prefillPlaneFor(unsigned int prompt_tokens,
                                    unsigned int cur_plane,
                                    unsigned int pack_plane,
                                    unsigned int max_seq_len) {
  if (prompt_tokens == 0u)
    return cur_plane;
  const unsigned int cap = std::min(kPrefillBlockCap, max_seq_len);
  if (cap < 8u)
    return cur_plane;
  const unsigned int floor_h = std::min(pack_plane, cap);
  unsigned int want = (prompt_tokens + 63u) & ~63u;
  if (want > cap || want < prompt_tokens) // the round-up can overflow
    want = cap;
  if (want < floor_h)
    want = floor_h;
  return want;
}

/**
 * @brief Sliding-window KV ring capacity.
 * @details A sliding-window attention layer with local window W only ever
 * attends to the last W keys, so with chunked prefill -- which bounds one
 * launch's live key span to W+C -- its KV storage can be a ring of Wcap rows
 * instead of the full max_seq.
 *
 * Returns Wcap (the physical row capacity to allocate and modulo-index) for a
 * sliding layer, or 0 meaning "no ring, keep full max_seq" (full-attention
 * layer, ring disabled, no chunking, or no benefit). Every site -- placeholder
 * shape, KV allocation, cache write, attention kernel dispatch -- computes Wcap
 * from THIS one function so they stay consistent.
 *
 * Wcap is a multiple of C and >= W + C: a multiple of C means a C-aligned chunk
 * write never straddles the wrap seam (it stays one contiguous slice), and
 * >= W + C means the live window [pos-W+1, pos+C) never self-collides mod Wcap.
 * Returning 0 keeps the exact pre-ring behaviour, so ring-off is bit-identical.
 *
 * @param local_window W, the layer's sliding window (0 = full attention).
 * @param max_seq the full context window this layer would otherwise allocate.
 * @param chunk C, the chunk the prefill ACTUALLY runs --
 *        effectivePrefillChunk(), not requestedPrefillChunk(). It is a
 *        parameter rather than a call so that the caller's chunk and this cap
 *        cannot drift apart.
 * @param ring_model_default the model's own default for the ring (see
 *        kvRingEnabled); the SAME value the chunk was computed with.
 */
inline unsigned int kvRingCap(unsigned int local_window, unsigned int max_seq,
                              unsigned int chunk,
                              bool ring_model_default = false) {
  if (!kvRingEnabled(ring_model_default))
    return 0; // ring off -> full max_seq (bit-identical legacy)
  if (local_window == 0 || local_window >= max_seq)
    return 0; // full-attention layer -> no ring
  const unsigned int C = chunk;
  if (C == 0)
    return 0; // the ring requires chunked prefill to bound the live span
  // multiple of C, >= W + C (headroom so the window never wraps onto itself).
  const unsigned int cap = (local_window / C + 2u) * C;
  return (cap < max_seq) ? cap : 0u; // no benefit if it would not shrink
}

/**
 * @brief Rows the Adreno OHWI K/V image mirror of a RINGED layer must hold.
 * @param ring_cap Wcap from kvRingCap() (0 = not ringed, this returns 0 too and
 *        the caller keeps its own derivation -- max_timestep for a linear
 *        layer).
 * @param local_window W, the layer's sliding window.
 * @param chunk C, the chunk the prefill ACTUALLY runs
 * (effectivePrefillChunk()).
 * @details The mirror is NOT the ring. The ring is storage, so it is sized by
 * the longest span any single launch must READ BACK, which the chunk-aligned
 * seam rule then rounds up to a multiple of C. The mirror is the linear window
 * the image kernels address, so it only has to hold the rows ONE launch looks
 * at: a launch at [f, f+S) with S <= C reads keys [f+1-W, f+S), and the base is
 * floored to the 64-row grid (so that mirror-local == absolute shifted, which
 * is what keeps ring-on bit-identical to ring-off), giving at most
 *
 *     (W - 1) + C + 63   rows.
 *
 * Sizing the mirror at Wcap instead -- which is what the ring-aware mirror
 * landed with -- ties it to the PREFILL chunk twice over: 8192 rows at C=4096
 * against 2048 at C=1024 for the same W=512 layer. That is pure cost: per
 * layer, K and V together are 2 * hKV * d * rows * 2 bytes, i.e. ~88 MiB across
 * the 0.3B's 14 sliding layers and ~160 MiB across the 1.5B's 20.
 *
 * Never more than ring_cap (a bigger mirror than the store it back-fills from
 * cannot be filled), and never less than what one launch needs, so the
 * mirror_fits backstop in mha_core stays a backstop rather than a live gate.
 * NNTR_KV_MIRROR_TIGHT=0 restores the Wcap-high mirror.
 */
inline unsigned int kvMirrorRows(unsigned int ring_cap,
                                 unsigned int local_window,
                                 unsigned int chunk) {
  if (ring_cap == 0u || local_window == 0u || chunk == 0u)
    return 0u; // linear layer / no ring -> the caller's own derivation
  if (const char *t = std::getenv("NNTR_KV_MIRROR_TIGHT"))
    if (t[0] == '0')
      return ring_cap; // control arm: the Wcap-high mirror
  // (W - 1) + C + 63, rounded up to the 64-row grid the base is floored to.
  const unsigned long need =
    (unsigned long)local_window + (unsigned long)chunk + 64ul;
  const unsigned long rows = (need + 63ul) & ~63ul;
  if (rows >= (unsigned long)ring_cap)
    return ring_cap;
  return (unsigned int)rows;
}

/**
 * @brief Mirror-local span a DECODE step may occupy before the mirror re-bases.
 * @param local_window W, the layer's sliding window (0 = no bound).
 * @param mirror_rows the mirror's physical height (kv_mirror_S_max).
 * @details This is the number the task "a big prefill block must not cost
 * decode" comes down to. The sliding mirror holds absolute rows
 * [base, cache_to) and the image kernels are handed N_kv = cache_to - base --
 * so N_kv, and with it every per-step attention kernel (qk, row softmax, sv),
 * is the mirror's OCCUPANCY, not the window. Re-basing costs a back-fill of ~W
 * rows, so the mirror slid only when it was FULL, which made the occupancy --
 * and the decode cost -- a function of the mirror height, hence of Wcap, hence
 * of the PREFILL chunk:
 *
 *     C=1024 -> Wcap 2048 -> N_kv cycles ~1536..2048  (0.3B 16K: 48.7 TPS)
 *     C=4096 -> Wcap 8192 -> N_kv cycles ~4608..8192  (0.3B 16K: 31.8 TPS)
 *
 * Decode only ever needs W + 63 rows (one query row, the 64-floored base), so
 * bound the occupancy at W rounded up to the grid plus one slack band and
 * re-base when it is reached. The slack is what the re-base amortizes over: a
 * slide back-fills ~W rows of K and V once every `slack` steps, i.e. ~W/slack
 * extra rows per step against the ~W rows of attention every step already pays,
 * while the average N_kv drops from (height + W)/2 to W + slack/2.
 *
 * Returns 0 when there is nothing to bound (full-attention layer, or a mirror
 * already no taller than the bound), which leaves the pre-existing
 * "slide when full" policy in place. NNTR_KV_DECODE_SLACK sets the band (>= 64,
 * rounded to 64); 0 disables the eager slide entirely (the control arm).
 */
inline unsigned int kvDecodeSlack() {
  // Not cached: these are called once per layer at finalize and once per
  // prefill<->decode transition, never per step, and a cached value would make
  // the unit tests order-dependent.
  const char *e = std::getenv("NNTR_KV_DECODE_SLACK");
  if (e == nullptr || e[0] == '\0')
    return 512u;
  char *end = nullptr;
  const long v = std::strtol(e, &end, 10);
  if (end == e || *end != '\0' || v < 0)
    return 512u;
  if (v == 0)
    return 0u; // control arm: one mirror, "slide only when it is full"
  return ((unsigned int)v < 64u) ? 64u : (((unsigned int)v + 63u) & ~63u);
}

/**
 * @brief Rows a ringed layer's mirror needs for DECODE alone.
 * @param local_window W, the layer's sliding window (0 = full attention).
 * @details One query row against the window, with the base floored to the
 * 64-row grid: W + 63 + 1 rows, rounded to the grid and given one slack band so
 * the re-base is amortised rather than every step. This is what a mirror is
 * re-materialised at when prefill hands over to decode -- see
 * kvMirrorRows() for why the PREFILL height has to be (W-1) + C + 63 and
 * therefore cannot be this.
 *
 * 0 means "do not resize" (full-attention layer, or NNTR_KV_DECODE_SLACK=0).
 */
inline unsigned int kvDecodeMirrorRows(unsigned int local_window) {
  if (local_window == 0u)
    return 0u;
  const unsigned int slack = kvDecodeSlack();
  if (slack == 0u)
    return 0u;
  return (((local_window + 63u) & ~63u) + slack);
}

/**
 * @brief Mirror-local occupancy a DECODE step may reach before the mirror
 *        re-bases, given the height the mirror actually has.
 * @details A mirror already re-materialised at kvDecodeMirrorRows() bounds the
 * occupancy by being that tall, and this returns 0 (nothing left to bound, the
 * pre-existing "slide when the mirror is full" policy is the right one). It is
 * non-zero only where the mirror is TALLER than decode needs and cannot be
 * resized -- then the base is slid eagerly instead, which bounds N_kv but not
 * the image geometry.
 */
inline unsigned int kvDecodeMirrorSpan(unsigned int local_window,
                                       unsigned int mirror_rows) {
  if (local_window == 0u || mirror_rows == 0u)
    return 0u;
  const unsigned int span = kvDecodeMirrorRows(local_window);
  if (span == 0u || span >= mirror_rows)
    return 0u; // disabled, or the mirror is already that tight
  return span;
}

/**
 * @brief Physical cache row for an absolute position under a ring of `cap`
 *        rows (cap == 0 => linear, the identity).
 * @details The host-side twin of the kernels' `n % ring_cap`.
 */
inline unsigned long kvCacheRow(unsigned long abs_pos, unsigned int cap) {
  return cap ? (abs_pos % static_cast<unsigned long>(cap)) : abs_pos;
}

/**
 * @brief One contiguous piece of a mirror -> concat-slab boundary sync.
 * @details `mirror_row` is the row inside the K/V mirror (mirror-base relative,
 * which is how every mirror kernel addresses it), `slab_row` the PHYSICAL row
 * in the concat cache slab (ring-mapped), and `rows` a count that is contiguous
 * in both.
 */
struct KvSlabSegment {
  unsigned int mirror_row;
  unsigned int slab_row;
  unsigned int rows;
};

/**
 * @brief Split an ABSOLUTE position range into pieces a mirror->slab gather may
 *        actually copy.
 * @details The boundary sync (mha_core's sync_kv_slab, NNTR_MHA_CLMEM) used to
 * take the absolute range straight to both sides: it wrote the concat slab at
 * `abs` and read the mirror at `abs`. Both are wrong the moment either side
 * moves:
 *
 *   - a RINGED layer's slab is only `ring_cap` rows tall, so writing row `abs`
 *     runs past the layer's allocation once abs >= ring_cap -- an out-of-bounds
 *     write, not a wrong answer;
 *   - a mirror that has SLID holds absolute rows [mirror_base, +mirror_rows),
 * so reading it at `abs` reads the wrong rows (and past its height).
 *
 * This maps both sides the way every other consumer does -- slab through
 * `kvCacheRow`, mirror through `- mirror_base` -- splits at the ring seam so a
 * piece is contiguous on both sides, and DROPS whatever the mirror does not
 * hold instead of reading outside it. An empty result means "the mirror has
 * nothing to contribute here", which is the correct refusal.
 *
 * @param from first absolute position to sync (inclusive)
 * @param to one past the last absolute position (exclusive)
 * @param ring_cap physical rows of the concat slab, 0 for a linear cache
 * @param mirror_base first absolute position the mirror holds
 * @param mirror_rows mirror height in rows; 0 means "unbounded" (do not clamp)
 * @return the pieces to copy, in increasing absolute order
 */
inline std::vector<KvSlabSegment>
kvSlabSyncSegments(unsigned int from, unsigned int to, unsigned int ring_cap,
                   unsigned int mirror_base, unsigned int mirror_rows) {
  std::vector<KvSlabSegment> out;
  if (from < mirror_base)
    from = mirror_base; // below the mirror: it does not hold these rows
  if (mirror_rows != 0u) {
    const unsigned long lim =
      static_cast<unsigned long>(mirror_base) + mirror_rows;
    if (static_cast<unsigned long>(to) > lim)
      to = static_cast<unsigned int>(lim);
  }
  if (to <= from)
    return out;
  unsigned int pos = from;
  while (pos < to) {
    const unsigned int slab =
      static_cast<unsigned int>(kvCacheRow(pos, ring_cap));
    unsigned int rows = to - pos;
    if (ring_cap != 0u && rows > ring_cap - slab)
      rows = ring_cap - slab; // stop at the ring seam
    out.push_back(KvSlabSegment{pos - mirror_base, slab, rows});
    pos += rows;
  }
  return out;
}

} // namespace causallm

#endif // __CAUSALLM_KV_RING_H__

// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_kernel.h
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for kernel management
 *
 */

#ifndef __OPENCL_KERNEL_H__
#define __OPENCL_KERNEL_H__

#include <cstddef>
#include <cstdint>
#include <string>

#include "CL/cl.h"
#include "opencl_program.h"

namespace nntrainer::opencl {

/**
 * @class Kernel contains wrappers for managing OpenCL kernels
 * @brief OpenCL kernel wrapper
 *
 */
class Kernel {
  cl_kernel kernel_{nullptr};

  /// Kernel-argument value cache. Decode re-binds ~6.6 arguments per dispatch
  /// and ~9 700 per token, almost all of them the same handle or the same
  /// scalar as the token before; each one is a host call the GPU waits behind
  /// (P2 measured host issue cost converting ~1:1 into wall time). The cache
  /// remembers the bytes last passed to clSetKernelArg for this kernel object
  /// and skips the call when they are unchanged.
  ///
  /// Soundness rests on two things. (1) The cl_kernel object holds argument
  /// state, so "unchanged bytes" really means "already bound" -- and a
  /// cl_kernel here is per (program, name, build options) and dispatched from
  /// the single-threaded forward path. (2) A remembered cl_mem/SVM VALUE can
  /// only lie if the handle was released and reissued for a different object,
  /// which is exactly what opencl::clHandleEpoch() tracks: the epoch is stored
  /// with the cache and any creation anywhere drops it.
  static constexpr int kArgCacheSlots = 20;
  static constexpr int kArgCacheBytes = 16;
  unsigned char arg_bytes_[kArgCacheSlots][kArgCacheBytes] = {};
  unsigned char arg_size_[kArgCacheSlots] = {}; ///< bound size, 0 = unset
  unsigned char arg_kind_[kArgCacheSlots] = {}; ///< 0 unset, 1 plain, 2 SVM
  unsigned long long arg_epoch_ = 0;

  /**
   * @brief Look the argument up in the cache; on a miss, record it.
   * @return true when the argument is already bound and the driver call can be
   *         skipped
   */
  bool argCacheHit(cl_uint arg_index, const void *arg_value, size_t size,
                   unsigned char kind);

public:
  /**
   * @brief Create a Kernel From Program object
   *
   * @param program
   * @param function_name the kernel string name
   * @return true if successful or false otherwise
   */
  bool CreateKernelFromProgram(Program program,
                               const std::string &function_name);

  /**
   * @brief Set the Kernel Arguments
   *
   * @param arg_index index of the argument
   * @param arg_value value of the argument
   * @param size size of the argument
   * @return true if successful or false otherwise
   */
  bool SetKernelArguments(cl_uint arg_index, const void *arg_value,
                          size_t size);

  /**
   * @brief Set the Kernel Arguments
   *
   * @param arg_index index of the argument
   * @param arg_value value of the argument
   * @return true if successful or false otherwise
   */
  bool SetKernelSVMArguments(cl_uint arg_index, const void *arg_value);

  /**
   * @brief Get the Kernel object
   *
   * @return const cl_kernel
   */
  const cl_kernel GetKernel();

  /**
   * @brief Read and clear whether a shared-virtual-memory pointer was bound
   *        since the previous dispatch.
   *
   * The coherence drain in CommandQueueManager calls this when it enqueues a
   * kernel, so that it flushes the queue only after a dispatch that actually
   * touched shared memory -- the real producer-to-consumer boundary -- rather
   * than after every dispatch. The flag is set by SetKernelSVMArguments and
   * cleared here on read, so it describes exactly the dispatch being enqueued
   * whatever order the arguments were bound in. The flag is thread_local, so
   * it pairs each bind with the enqueue on the same thread.
   *
   * @return true when the dispatch being enqueued bound shared memory
   */
  static bool takeDispatchTouchedSVM();

  /**
   * @brief Declare what the NEXT dispatch writes on the device activation
   *        plane, for the producer-to-consumer handoff log below.
   *
   * A caller that knows its kernel writes exactly one device buffer (and
   * nothing else another op could be reading) calls this immediately before
   * the dispatch. A dispatch for which nothing is declared is logged as
   * UNKNOWN, and an UNKNOWN dispatch invalidates every outstanding handoff --
   * so a site that is not annotated costs a re-computation, never a wrong
   * answer.
   *
   * @param written device buffer handle (cl_mem) the next dispatch writes
   */
  static void noteDispatchWrites(void *written);

  /**
   * @brief Declare that the NEXT dispatch writes nothing that a handoff can
   *        be tracking (its outputs are private scratch).
   */
  static void noteDispatchWritesNothing();

  /**
   * @brief Append the pending declaration to the log. Called once per
   *        enqueued NDRange by CommandQueueManager, whatever the outcome of
   *        the declaration, so the log has exactly one entry per dispatch.
   */
  static void commitDispatch();

  /**
   * @brief Sequence number of the last logged dispatch.
   */
  static unsigned long long dispatchSeq();

  /**
   * @brief Whether the device buffer @a handle still holds what it held at
   *        @a since_seq.
   *
   * True only when EVERY dispatch logged after @a since_seq declared what it
   * wrote and none of them wrote @a handle. An unannotated dispatch, or a
   * window longer than the log, answers false.
   *
   * @param handle device buffer handle (cl_mem)
   * @param since_seq dispatchSeq() sampled when the content was established
   * @return true when nothing can have overwritten the buffer since
   */
  static bool bufferUnchangedSince(void *handle, unsigned long long since_seq);

  /**
   * @brief Declare that the dispatch just enqueued wrote a shared-memory
   *        plane and was NOT drained.
   *
   * @details A kernel that writes shared virtual memory and only flushes the
   * queue leaves a write whose visibility to a LATER submission a coarse-grain
   * driver does not order. The producer knows it is doing that -- it chose not
   * to drain -- so it says so here, naming the plane it wrote. The set is
   * consulted by every subsequent dispatch that binds a shared-memory pointer
   * (takeDispatchSvmHazard), and the drain happens there, once, only when a
   * consumer actually binds into the plane. That is the risk condition stated
   * in code, rather than guessed from a per-pack flag.
   *
   * The plane is a RANGE rather than a pointer because producer and consumer do
   * not have to name the same address: on this chain the producer binds an
   * offset-baked slice and the consumer binds the tensor's stable base with the
   * offset as a kernel scalar (NNTR_KV_SCALAR_OFF). Both lie inside the
   * tensor's plane, which is what makes containment the right test.
   *
   * A plane declared and never consumed costs nothing; it is dropped by the
   * next drain. A consumer missed by the test is the pre-existing behaviour, so
   * an incomplete annotation can only fail to add a drain, never remove one.
   *
   * The check is DEFAULT OFF (NNTR_SVM_HAZARD_DRAIN=1 opts in); see the
   * predicate in the .cpp for the measurement behind that call.
   *
   * Scope. NNTR_SVM_HAZARD_SCOPE picks which declarations are recorded:
   * `site` (the default) takes only the ones a call site made because it knows
   * the plane a later submission consumes -- the KV cache slices in the
   * attention core, which is where every model on this chain creates the
   * pairing; `all` additionally takes the generic declaration each undrained
   * shared write makes for itself, which is the exhaustive reading of the
   * condition and measurably more expensive (it drains the decode chain too).
   * Both are structural; `all` is a superset.
   *
   * @param base first byte of the shared-memory plane written
   * @param bytes size of that plane
   * @param declared_by_consumer_site true when the caller named this plane
   *        because it knows a later submission reads it
   */
  static void noteUndrainedSvmPlane(const void *base, size_t bytes,
                                    bool declared_by_consumer_site = false);

  /**
   * @brief Whether the dispatch about to be enqueued binds a shared-memory
   *        pointer inside a plane an earlier undrained dispatch wrote.
   *
   * Clears the recorded planes when it answers true, because the caller drains
   * the queue on that answer and the drain retires every outstanding write, not
   * only the one that was hit.
   *
   * @return true when this dispatch has to drain before it is enqueued
   */
  static bool takeDispatchSvmHazard();

  /**
   * @brief Forget the outstanding undrained planes. Call after any full queue
   *        drain, which is what makes them visible.
   */
  static void clearUndrainedSvmPlanes();

  /**
   * @brief How many drains takeDispatchSvmHazard() has asked for, for the
   *        run's ledger.
   */
  static unsigned long long svmHazardDrains();

private:
  /**
   * @brief Remember a shared-memory pointer bound for the dispatch being
   *        assembled, so takeDispatchSvmHazard() can test it once.
   *
   * @param p the bound shared-memory pointer
   */
  static void noteBoundSvmPointer(const void *p);
};
} // namespace nntrainer::opencl
#endif // __OPENCL_KERNEL_H__

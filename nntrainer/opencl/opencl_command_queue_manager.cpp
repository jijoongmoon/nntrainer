// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_command_queue_manager.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for command queue management
 *
 */

#include "opencl_command_queue_manager.h"

#include "opencl_context_manager.h"
#include "opencl_loader.h"

#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer::opencl {

// ---------------------------------------------------------------------------
// cl_qcom_recordable_queues tokens/types — absent from the base CL/cl.h shipped
// in this tree; the Adreno 840 driver implements them at runtime. Values from
// CL/cl_ext_qcom.h (Qualcomm AI stack), same local-define pattern as the QCOM
// perf/priority hints in opencl_context_manager.cpp.
// ---------------------------------------------------------------------------
#ifndef CL_QUEUE_RECORDABLE_QCOM
#define CL_QUEUE_RECORDABLE_QCOM (1u << 30u) /* 0x40000000 */
#endif

// cl_recording_qcom + cl_array_arg_qcom / cl_offset_qcom / cl_workgroup_qcom are
// now declared in opencl_command_queue_manager.h (shared with the R3 override
// builder). The function-pointer typedefs + entry-point resolution stay here.

typedef cl_recording_qcom(CL_API_CALL *PFN_clNewRecordingQCOM)(cl_command_queue,
                                                               cl_int *);
typedef cl_int(CL_API_CALL *PFN_clEndRecordingQCOM)(cl_recording_qcom);
typedef cl_int(CL_API_CALL *PFN_clReleaseRecordingQCOM)(cl_recording_qcom);
typedef cl_int(CL_API_CALL *PFN_clEnqueueRecordingQCOM)(
  cl_command_queue, cl_recording_qcom, size_t, const cl_array_arg_qcom *,
  size_t, const cl_offset_qcom *, size_t, const cl_workgroup_qcom *, size_t,
  const cl_workgroup_qcom *, cl_uint, const cl_event *, cl_event *);

namespace {
// Per-kernel GPU profiling registry, populated by enqueueKernel when
// NNTR_OPENCL_PROFILING is set. Each entry owns one cl_event reference that
// dumpProfile() releases. Single-threaded dispatch path, so no lock needed.
struct ProfRec {
  std::string name;
  cl_event evt;
};

std::vector<ProfRec> &profRecs() {
  static std::vector<ProfRec> v;
  return v;
}

bool profEnabled() {
  static const int e = std::getenv("NNTR_OPENCL_PROFILING") ? 1 : 0;
  return e != 0;
}

// recordable-queue feasibility trace (NNTR_RECQ_TRACE): one shared op counter
// across kernel dispatches and host ops (SVM map/unmap, buffer read/map), so we
// can see whether host ops intersperse the decode kernel run (which would break
// the single-recording replay model).
void rqt_op(const char *tag, const char *name) {
  static const bool on = std::getenv("NNTR_RECQ_TRACE") != nullptr;
  if (!on)
    return;
  static int n = 0;
  fprintf(stderr, "[rqt] %5d %s %s\n", n++, tag, name ? name : "");
}

// cl_qcom_recordable_queues entry points, resolved once at queue-creation time
// when NNTR_RECQ is set (null otherwise). TU scope so the enqueue chokepoint and
// the decode-loop record/replay (phase-2) can reach them.
PFN_clNewRecordingQCOM recq_new_ = nullptr;
PFN_clEndRecordingQCOM recq_end_ = nullptr;
PFN_clEnqueueRecordingQCOM recq_enqueue_ = nullptr;
PFN_clReleaseRecordingQCOM recq_release_ = nullptr;
} // namespace

/**
 * @brief Create a Command Queue object
 *
 * @return true if creation is successful or false otherwise
 */
bool CommandQueueManager::CreateCommandQueue() {
  if (command_queue_) {
    ml_logi("opencl_command_queue_manager: Retained command queue");
    // increments the command_queue reference count
    clRetainCommandQueue(command_queue_);
    return true;
  }

  int error_code;
  ContextManager &context_instance = ContextManager::Global();

  // OpenCL context is created
  cl_context context = context_instance.GetContext();

  // If context is invalid, return false
  if (context == nullptr) {
    return false;
  }

  // getting GPU device ID
  cl_device_id device_id = context_instance.GetDeviceId();

  // Queue ordering policy. Default: out-of-order (unchanged) — the existing
  // path serializes via per-layer host round-trips, so OOO is harmless, and
  // gpu_native manages its own ordering/barriers on top of it. When the graph
  // opts into the GPU-resident SVM pool (NNTR_GPU_SVM_POOL), consecutive CL
  // layers hand off through a shared buffer with no host round-trip, so the
  // queue must execute in submission order -> use an in-order queue.
  // See tensor/cl_operations/GPU_GENERALIZATION_PLAN.md Step 1.
  cl_command_queue_properties qprops = 0;
  if (std::getenv("NNTR_GPU_SVM_POOL") == nullptr) {
    qprops |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
  }
  // Env-gated CL_QUEUE_PROFILING_ENABLE so v8c (and any other) callers can
  // collect per-command start/end timestamps without paying the profiling
  // tax in production runs. Set NNTR_OPENCL_PROFILING=1 to enable.
  if (std::getenv("NNTR_OPENCL_PROFILING")) {
    qprops |= CL_QUEUE_PROFILING_ENABLE;
  }
  // returns NULL with error code if fails
  command_queue_ = clCreateCommandQueue(context, device_id, qprops, &error_code);
  if (!command_queue_) {
    ml_loge("Failed to create a command queue. OpenCL error code: %d : ",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  ml_logi("opencl_command_queue_manager: Created command queue");
  // increments the command_queue reference count
  clRetainCommandQueue(command_queue_);
  ml_logi("opencl_command_queue_manager: Retained command queue");

  // cl_qcom_recordable_queues foundation (phase-1). Opt-in via NNTR_RECQ
  // (default off). Purely additive: the canonical path keeps using
  // command_queue_ regardless. Failure to set up the recordable queues is
  // non-fatal (logged), so a device without the extension still runs normally.
  if (std::getenv("NNTR_RECQ")) {
    initRecordableQueues(context, device_id);
  }
  return true;
}

/**
 * @brief Resolve the cl_qcom_recordable_queues entry points and create the
 * recordable + host-I/O queues (NNTR_RECQ). Non-fatal on failure.
 */
void CommandQueueManager::initRecordableQueues(cl_context context,
                                               cl_device_id device_id) {
  cl_platform_id platform = ContextManager::Global().GetPlatformId();
  if (!platform || !clGetExtensionFunctionAddressForPlatform) {
    ml_logw("NNTR_RECQ: platform id or extension resolver unavailable; "
            "recordable queues disabled.");
    return;
  }

  // The QCOM recording functions are extension entry points, not base ICD
  // symbols, so they must be resolved through the platform (not dlsym).
  recq_new_ = reinterpret_cast<PFN_clNewRecordingQCOM>(
    clGetExtensionFunctionAddressForPlatform(platform, "clNewRecordingQCOM"));
  recq_end_ = reinterpret_cast<PFN_clEndRecordingQCOM>(
    clGetExtensionFunctionAddressForPlatform(platform, "clEndRecordingQCOM"));
  recq_enqueue_ = reinterpret_cast<PFN_clEnqueueRecordingQCOM>(
    clGetExtensionFunctionAddressForPlatform(platform,
                                             "clEnqueueRecordingQCOM"));
  recq_release_ = reinterpret_cast<PFN_clReleaseRecordingQCOM>(
    clGetExtensionFunctionAddressForPlatform(platform,
                                             "clReleaseRecordingQCOM"));

  if (!recq_new_ || !recq_end_ || !recq_enqueue_ || !recq_release_) {
    ml_logw("NNTR_RECQ: the loaded libOpenCL.so does not export the QCOM "
            "recording entry points (New=%p End=%p Enq=%p Rel=%p). The Adreno "
            "profiler shim libOpenCL.so advertises the extension but does not "
            "forward these functions - run against the system ICD loader "
            "(/vendor/lib64/libOpenCL.so) to use NNTR_RECQ. Recordable queues "
            "disabled.",
            (void *)recq_new_, (void *)recq_end_, (void *)recq_enqueue_,
            (void *)recq_release_);
    recq_new_ = nullptr;
    recq_end_ = nullptr;
    recq_enqueue_ = nullptr;
    recq_release_ = nullptr;
    return;
  }

  if (!clCreateCommandQueueWithProperties) {
    ml_logw("NNTR_RECQ: clCreateCommandQueueWithProperties unavailable; "
            "recordable queues disabled.");
    return;
  }

  // The recordable queue carries the CL_QUEUE_RECORDABLE_QCOM property; the
  // separate io queue is a plain default queue for host readback (the
  // recordable queue rejects clEnqueueReadBuffer with CL_INVALID_OPERATION).
  int error_code = CL_SUCCESS;
  const cl_queue_properties recq_props[] = {CL_QUEUE_PROPERTIES,
                                            CL_QUEUE_RECORDABLE_QCOM, 0};
  recordable_command_queue_ = clCreateCommandQueueWithProperties(
    context, device_id, recq_props, &error_code);
  if (!recordable_command_queue_) {
    ml_logw("NNTR_RECQ: failed to create the recordable command queue "
            "(%d : %s); recordable queues disabled.",
            error_code, OpenCLErrorCodeToString(error_code));
    return;
  }

  io_command_queue_ = clCreateCommandQueueWithProperties(context, device_id,
                                                         nullptr, &error_code);
  if (!io_command_queue_) {
    ml_logw("NNTR_RECQ: failed to create the host-I/O command queue "
            "(%d : %s); recordable queues disabled.",
            error_code, OpenCLErrorCodeToString(error_code));
    clReleaseCommandQueue(recordable_command_queue_);
    recordable_command_queue_ = nullptr;
    return;
  }

  ml_logi("NNTR_RECQ: cl_qcom_recordable_queues ready - recordable queue %p + "
          "host-I/O queue %p, all 4 QCOM entry points resolved.",
          (void *)recordable_command_queue_, (void *)io_command_queue_);
}

// ---------------------------------------------------------------------------
// Record/replay API (R1). The three clEnqueueNDRangeKernel chokepoints above
// target active_recording_queue_ while it is non-null, so beginRecording()
// flips the WHOLE kernel-dispatch chain into capture mode without touching any
// caller. Default (null) path is byte-identical.
// ---------------------------------------------------------------------------
bool CommandQueueManager::beginRecording() {
  if (recq_new_ == nullptr || recordable_command_queue_ == nullptr) {
    ml_logw("NNTR_RECQ: beginRecording requested but the recordable queue / "
            "entry points are unavailable (needs NNTR_RECQ on a QCOM device).");
    return false;
  }
  releaseRecording(); // a prior recording must be freed first
  cl_int err = CL_SUCCESS;
  active_recording_handle_ = recq_new_(recordable_command_queue_, &err);
  if (active_recording_handle_ == nullptr || err != CL_SUCCESS) {
    ml_loge("NNTR_RECQ: clNewRecordingQCOM failed (err %d).", err);
    active_recording_handle_ = nullptr;
    return false;
  }
  recq_dispatch_index_ = 0;
  active_recording_queue_ = recordable_command_queue_; // enter capture mode
  return true;
}

bool CommandQueueManager::endRecording() {
  if (active_recording_queue_ == nullptr || active_recording_handle_ == nullptr)
    return false;
  const cl_int err =
    recq_end_ ? recq_end_(active_recording_handle_) : CL_INVALID_OPERATION;
  active_recording_queue_ = nullptr; // leave capture mode regardless
  if (err != CL_SUCCESS) {
    ml_loge("NNTR_RECQ: clEndRecordingQCOM failed (err %d).", err);
    return false;
  }
  return true;
}

bool CommandQueueManager::replayRecording(const cl_array_arg_qcom *args,
                                          size_t n_args,
                                          const cl_workgroup_qcom *gws,
                                          size_t n_gws, cl_event *event) {
  if (active_recording_handle_ == nullptr || recq_enqueue_ == nullptr ||
      command_queue_ == nullptr)
    return false;
  // Replay on the LIVE in-order command_queue_ (NOT the recordable queue). Only
  // scalar-arg + global-work-size overrides are used (decode de-SVM needs no
  // global-offset or local-work-size overrides).
  const cl_int err =
    recq_enqueue_(command_queue_, active_recording_handle_, n_args, args, 0,
                  nullptr, n_gws, gws, 0, nullptr, 0, nullptr, event);
  if (err != CL_SUCCESS) {
    ml_loge("NNTR_RECQ: clEnqueueRecordingQCOM failed (err %d).", err);
    return false;
  }
  return true;
}

void CommandQueueManager::releaseRecording() {
  // NOTE: the Adreno driver's clReleaseRecordingQCOM (libCB cb_release_recording_
  // qcom) has been observed to SIGSEGV at process teardown on a finalized
  // recording. Since recq is experimental/gated and this is a one-shot
  // teardown, we intentionally DO NOT call recq_release_ (the handle leaks for
  // the remainder of the process, which the OS reclaims on exit) -- this keeps
  // recq-replay runs from crashing at shutdown. Revisit if the driver issue is
  // resolved.
  active_recording_handle_ = nullptr;
}

/**
 * @brief Release th OpenCL command queue instance
 *
 */
void CommandQueueManager::ReleaseCommandQueue() {
  if (command_queue_) {
    ml_logi("opencl_command_queue_manager: Released command queue");
    clReleaseCommandQueue(command_queue_);
  }
}

/**
 * @brief Destroy the Command Queue Manager object
 *
 */
CommandQueueManager::~CommandQueueManager() {
  // Release any held record/replay recording before its queues go away.
  releaseRecording();
  // Recordable + host-I/O queues (NNTR_RECQ) are created once with refcount 1,
  // so a single release each is correct.
  if (recordable_command_queue_) {
    clReleaseCommandQueue(recordable_command_queue_);
    recordable_command_queue_ = nullptr;
  }
  if (io_command_queue_) {
    clReleaseCommandQueue(io_command_queue_);
    io_command_queue_ = nullptr;
  }
  if (command_queue_) {
    ml_logi("opencl_command_queue_manager: Destroyed command queue");
    // decrements the command_queue reference count
    clReleaseCommandQueue(command_queue_);
    command_queue_ = nullptr;

    // releasing OpenCL context since it has been created by
    // CommandQueueManager::CreateCommandQueue
    ContextManager::Global().ReleaseContext();
  }
}

/**
 * @brief Get the OpenCL Command Queue object
 *
 * @return const cl_command_queue
 */
const cl_command_queue CommandQueueManager::GetCommandQueue() {
  return command_queue_;
}

/**
 * @brief Reading buffer object. Used from Buffer class
 *
 * @param buffer cl_mem buffer object
 * @param size_in_bytes size of data
 * @param data getting the data stored in buffer
 * @param async flag for asynchronous operation
 * @return true if reading is successful or false otherwise
 */
bool CommandQueueManager::EnqueueReadBuffer(cl_mem buffer, size_t size_in_bytes,
                                            void *data, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  // returns NULL with error code if fails
  rqt_op("HOST_readbuf", nullptr);
  auto error_code =
    clEnqueueReadBuffer(command_queue_, buffer, blocking, 0, size_in_bytes,
                        data, 0, nullptr, nullptr);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to read data from GPU (clEnqueueReadBuffer). OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

bool CommandQueueManager::EnqueueReadBufferRegion(
  cl_mem buffer, size_t size_in_bytes, void *data, size_t host_origin_offset,
  size_t buffer_origin_offset, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;

  // (x, y, z) offset in the memory region associated with buffer
  const size_t buffer_origin[] = {buffer_origin_offset, 0, 0};
  // (x, y, z) offset in the memory region associated with host
  const size_t host_origin[] = {host_origin_offset, 0, 0};
  // region defines the (width in bytes, height in rows, depth in slices)
  const size_t region[] = {size_in_bytes, 1, 1};
  // length of each row in bytes
  size_t row_pitch = region[0];
  // length of each 2D slice in bytes
  size_t slice_pitch = region[0] * region[1];

  // Buffer and host data are interpreted as 1D in this case
  // hence row and slice pitch are same for both
  cl_int error_code = clEnqueueReadBufferRect(
    command_queue_, buffer, blocking, buffer_origin, host_origin, region,
    row_pitch, slice_pitch, row_pitch, slice_pitch, data, 0, nullptr, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to write data region to GPU (clEnqueueReadBufferRect). "
            "OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

/**
 * @brief Writing buffer object. Used from Buffer class
 *
 * @param buffer cl_mem buffer object
 * @param size_in_bytes size of data
 * @param data to be enqueued into the buffer
 * @param async flag for asynchronous operation
 * @return true if writing is successful or false otherwise
 */
bool CommandQueueManager::EnqueueWriteBuffer(cl_mem buffer,
                                             size_t size_in_bytes,
                                             const void *data, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  // returns NULL with error code if fails
  auto error_code =
    clEnqueueWriteBuffer(command_queue_, buffer, blocking, 0, size_in_bytes,
                         data, 0, nullptr, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to upload data to GPU (clEnqueueWriteBuffer). OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

bool CommandQueueManager::EnqueueWriteBufferRegion(
  cl_mem buffer, size_t size_in_bytes, const void *data,
  size_t host_origin_offset, size_t buffer_origin_offset, bool async) {

  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;

  // (x, y, z) offset in the memory region associated with buffer
  const size_t buffer_origin[] = {buffer_origin_offset, 0, 0};
  // (x, y, z) offset in the memory region associated with host
  const size_t host_origin[] = {host_origin_offset, 0, 0};
  // region defines the (width in bytes, height in rows, depth in slices)
  const size_t region[] = {size_in_bytes, 1, 1};
  // length of each row in bytes
  size_t row_pitch = region[0];
  // length of each 2D slice in bytes
  size_t slice_pitch = region[0] * region[1];

  // Buffer and host data are interpreted as 1D in this case
  // hence row and slice pitch are same for both
  cl_int error_code = clEnqueueWriteBufferRect(
    command_queue_, buffer, blocking, buffer_origin, host_origin, region,
    row_pitch, slice_pitch, row_pitch, slice_pitch, data, 0, nullptr, nullptr);

  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to write data region to GPU (clEnqueueWriteBufferRect). "
            "OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }

  return true;
}

/**
 * @brief Mapping a region of a buffer object into the host address space
 *
 * @param buffer cl_mem buffer object
 * @param offset_in_bytes offset of the region in the buffer object that is
 * being mapped
 * @param size_in_bytes size of the buffer object that is being mapped
 * @param read_only flag for read only mapping
 * @param async flag for asynchronous operation
 * @param event Object that identifies this command and can be used to query
 * or wait for this command to complete
 * @return void* pointer to the mapped region
 */
void *CommandQueueManager::EnqueueMapBuffer(cl_mem buffer,
                                            size_t offset_in_bytes,
                                            size_t size_in_bytes,
                                            bool read_only, bool async,
                                            cl_event *event) {
  // managing synchronization
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;
  // managing read/write flags
  const cl_map_flags map_flag = read_only ? CL_MAP_READ : CL_MAP_WRITE;

  cl_int error_code;

  rqt_op("HOST_mapbuf", nullptr);
  void *host_mem_buf = clEnqueueMapBuffer(
    command_queue_, buffer, blocking, map_flag, offset_in_bytes, size_in_bytes,
    0, nullptr, event, &error_code);

  if (error_code != CL_SUCCESS) {
    ml_loge(
      "Failed to map buffer to host memory(clEnqueueMapBuffer). OpenCL error "
      "code: %d : %s",
      error_code, OpenCLErrorCodeToString(error_code));
    return nullptr;
  }
  return host_mem_buf;
}

/**
 * @brief Mapping a region of a buffer object into the host address space
 *
 * @param buffer cl_mem buffer object
 * @param mapped_ptr pointer to the mapped region
 * @param event Object that identifies this command and can be used to query
 * or wait for this command to complete
 * @return true if unmap is successful
 */
bool CommandQueueManager::EnqueueUnmapMemObject(cl_mem buffer, void *mapped_ptr,
                                                cl_event *event) {
  cl_int error_code = clEnqueueUnmapMemObject(command_queue_, buffer,
                                              mapped_ptr, 0, nullptr, event);
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to unmap buffer from host memory(clEnqueueUnmapMemObject). "
            "OpenCL error "
            "code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  return true;
}

void CommandQueueManager::finish() {
  if (command_queue_)
    clFinish(command_queue_);
}

// NNTR_SVM_RESIDENT: keep the activation chain fully GPU-resident. On the
// in-order SVM queue (NNTR_GPU_SVM_POOL) consecutive GPU kernels are already
// device-coherent without host map/unmap; the per-op maps exist only as a
// defensive host-coherence guard that no all-GPU consumer needs. Skipping them
// removes the coarse-grain SVM coherence ops (cache flush/invalidate) that
// serialize the layer-graph forward and lose the GPU/host overlap gpu_native
// keeps. Genuine host boundaries (e.g. lm_head input read) pass force=true to
// map anyway. Default off (original blocking behavior).
static bool svm_resident_mode() {
  static const bool v = std::getenv("NNTR_SVM_RESIDENT") != nullptr;
  return v;
}

bool CommandQueueManager::enqueueSVMMap(void *svm_ptr, size_t size,
                                        bool read_only, bool async,
                                        cl_event *event, bool force) {
  if (svm_resident_mode() && !force)
    return true; // resident: stays device-coherent, no host map needed
  // managing read/write flags
  const cl_map_flags map_flag = read_only ? CL_MAP_READ : CL_MAP_WRITE;

  // async=true => non-blocking map (CL_FALSE). Safe ONLY on an in-order queue
  // (NNTR_GPU_SVM_POOL path) where the map is ordered before the next op's
  // unmap/kernel, AND when no host access of this region happens before that
  // next GPU op. Removes the per-op host stall that otherwise drains the queue
  // to idle. Default (false) keeps the original blocking behavior.
  const cl_bool blocking = async ? CL_FALSE : CL_TRUE;

  rqt_op(blocking ? "HOST_svmmap_BLOCK" : "HOST_svmmap_async", nullptr);
  cl_int error_code = clEnqueueSVMMap(command_queue_, blocking, map_flag,
                                      svm_ptr, size, 0, nullptr, event);

  if (error_code != CL_SUCCESS) {
    ml_loge(
      "Failed to map SVM memory (clEnqueueSVMMap). OpenCL error code: %d : %s",
      error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  return true;
}

bool CommandQueueManager::enqueueSVMUnmap(void *svm_ptr, cl_event *event,
                                          bool force) {
  if (svm_resident_mode() && !force)
    return true; // resident: stays device-coherent, no host unmap needed
  cl_int error_code =
    clEnqueueSVMUnmap(command_queue_, svm_ptr, 0, nullptr, event);

  if (error_code != CL_SUCCESS) {
    ml_loge(
      "Failed to unmap SVM memory (clEnqueueSVMUnmap). OpenCL error code: "
      "%d : %s",
      error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  return true;
}

/**
 * @brief Function to initiate execution of the command queue.
 *
 * @param kernel OpenCL kernel
 * @param work_groups_count Total number of work items that will execute the
 * kernel function
 * @param work_group_size Number of work items that make up a work group
 * @param event Object that identifies this command and can be used to query
 * or wait for this command to complete
 * @return true if command queue execution is successful or false otherwise
 */
bool CommandQueueManager::DispatchCommand(
  Kernel kernel, const int (&work_groups_count)[3],
  const int (&work_group_size)[3], cl_event *event,
  std::vector<cl_event> events_to_wait) {

  // work_dim of 2 has been hardcoded, might be modified later based on
  // requirements

  // setting the local_work_size referred to as the size of the
  // work-group
  const size_t local[3] = {static_cast<size_t>(work_group_size[0]),
                           static_cast<size_t>(work_group_size[1]),
                           static_cast<size_t>(work_group_size[2])};

  // setting the global_work_size that describe the number of global work-items
  const size_t global[3] = {static_cast<size_t>(work_groups_count[0]),
                            static_cast<size_t>(work_groups_count[1]),
                            static_cast<size_t>(work_groups_count[2])};

  cl_kernel kernel_ = kernel.GetKernel();

  // Profiling: capture a tracked event like enqueueKernel does. Without this,
  // every DispatchCommand-dispatched kernel (rmsnorm/geglu/v8c writers/...)
  // is INVISIBLE to dumpProfile and its GPU time is mis-attributed as
  // "inter-kernel idle" of the surrounding tracked kernels.
  cl_event local_evt = nullptr;
  cl_event *evt_arg = event;
  const bool track = profEnabled() && evt_arg == nullptr;
  if (track)
    evt_arg = &local_evt;

  // recq R4 feed pass: skip ALL dispatches (host-only forward) so only the host
  // embedding refreshes its output; the GPU forward comes from the replay.
  if (recq_skip_all_) {
    next_prof_label_.clear();
    return true;
  }
  // returns NULL with error code if fails. R1: while recording, capture onto
  // the recordable queue instead of executing on command_queue_; count the
  // captured dispatch so the caller can map it to a per-token override.
  cl_command_queue rq_target =
    active_recording_queue_ ? active_recording_queue_ : command_queue_;
  const int error_code = clEnqueueNDRangeKernel(
    rq_target, kernel_, 3, nullptr, global, local,
    events_to_wait.size(), events_to_wait.data(), evt_arg);
  if (active_recording_queue_ != nullptr && error_code == CL_SUCCESS)
    ++recq_dispatch_index_;
  static const bool rqt_on2 = std::getenv("NNTR_RECQ_TRACE") != nullptr;
  if (rqt_on2) {
    char nm[96] = {0};
    clGetKernelInfo(kernel_, CL_KERNEL_FUNCTION_NAME, sizeof(nm) - 1, nm,
                    nullptr);
    rqt_op("DISPATCH", nm);
  }
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to clEnqueueNDRangeKernel. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  if (track && local_evt != nullptr) {
    char nm[128] = {0};
    if (clGetKernelInfo(kernel_, CL_KERNEL_FUNCTION_NAME, sizeof(nm) - 1, nm,
                        nullptr) != CL_SUCCESS)
      nm[0] = '\0';
    std::string key(nm);
    if (!next_prof_label_.empty())
      key += next_prof_label_;
    profRecs().push_back({std::move(key), local_evt});
  }
  next_prof_label_.clear();

  return true;
}

bool CommandQueueManager::DispatchCommand(
  const std::shared_ptr<Kernel> &kernel_ptr, const int (&work_groups_count)[3],
  const int (&work_group_size)[3], cl_event *event,
  std::vector<cl_event> events_to_wait) {

  // work_dim of 2 has been hardcoded, might be modified later based on
  // requirements

  // setting the local_work_size referred to as the size of the
  // work-group
  const size_t local[3] = {static_cast<size_t>(work_group_size[0]),
                           static_cast<size_t>(work_group_size[1]),
                           static_cast<size_t>(work_group_size[2])};

  // setting the global_work_size that describe the number of global work-items
  const size_t global[3] = {static_cast<size_t>(work_groups_count[0]),
                            static_cast<size_t>(work_groups_count[1]),
                            static_cast<size_t>(work_groups_count[2])};

  cl_kernel kernel_ = kernel_ptr->GetKernel();

  // Profiling capture: see the by-value overload above.
  cl_event local_evt = nullptr;
  cl_event *evt_arg = event;
  const bool track = profEnabled() && evt_arg == nullptr;
  if (track)
    evt_arg = &local_evt;

  // recq R4 feed pass: skip ALL dispatches (host-only forward) so only the host
  // embedding refreshes its output; the GPU forward comes from the replay.
  if (recq_skip_all_) {
    next_prof_label_.clear();
    return true;
  }
  // returns NULL with error code if fails. R1: while recording, capture onto
  // the recordable queue instead of executing on command_queue_; count the
  // captured dispatch so the caller can map it to a per-token override.
  cl_command_queue rq_target =
    active_recording_queue_ ? active_recording_queue_ : command_queue_;
  const int error_code = clEnqueueNDRangeKernel(
    rq_target, kernel_, 3, nullptr, global, local,
    events_to_wait.size(), events_to_wait.data(), evt_arg);
  if (active_recording_queue_ != nullptr && error_code == CL_SUCCESS)
    ++recq_dispatch_index_;
  static const bool rqt_on2 = std::getenv("NNTR_RECQ_TRACE") != nullptr;
  if (rqt_on2) {
    char nm[96] = {0};
    clGetKernelInfo(kernel_, CL_KERNEL_FUNCTION_NAME, sizeof(nm) - 1, nm,
                    nullptr);
    rqt_op("DISPATCH", nm);
  }
  if (error_code != CL_SUCCESS) {
    ml_loge("Failed to clEnqueueNDRangeKernel. OpenCL error code: %d : %s",
            error_code, OpenCLErrorCodeToString(error_code));
    return false;
  }
  if (track && local_evt != nullptr) {
    char nm[128] = {0};
    if (clGetKernelInfo(kernel_, CL_KERNEL_FUNCTION_NAME, sizeof(nm) - 1, nm,
                        nullptr) != CL_SUCCESS)
      nm[0] = '\0';
    std::string key(nm);
    if (!next_prof_label_.empty())
      key += next_prof_label_;
    profRecs().push_back({std::move(key), local_evt});
  }
  next_prof_label_.clear();

  return true;
}

void CommandQueueManager::enqueueKernel(const cl_kernel kernel,
                                        const cl_uint work_dim,
                                        const size_t *global_work_size,
                                        const size_t *local_work_size,
                                        cl_uint num_events_in_wait_list,
                                        const cl_event *event_wait_list,
                                        cl_event *event) {

  // When profiling and the caller did not request its own event, capture a
  // tracked event so dumpProfile() can read true per-kernel GPU time. We own
  // the single reference and release it in dumpProfile(). (Calls that pass
  // their own event — e.g. the act-quant path — are left untracked to avoid
  // event-ownership complexity; they are a negligible slice anyway.)
  cl_event local_evt = nullptr;
  cl_event *evt_arg = event;
  const bool track = profEnabled() && evt_arg == nullptr;
  if (track)
    evt_arg = &local_evt;

  // recq R4 feed pass: skip ALL dispatches (host-only forward).
  if (recq_skip_all_) {
    next_prof_label_.clear();
    return;
  }
  // R1: while recording, capture onto the recordable queue instead of executing
  // on command_queue_; count the captured dispatch for the override mapping.
  cl_command_queue rq_target =
    active_recording_queue_ ? active_recording_queue_ : command_queue_;
  const auto error_code = clEnqueueNDRangeKernel(
    rq_target, kernel, work_dim, nullptr, global_work_size,
    local_work_size, num_events_in_wait_list, event_wait_list, evt_arg);
  if (active_recording_queue_ != nullptr && error_code == CL_SUCCESS)
    ++recq_dispatch_index_;

  static const bool rqt_on = std::getenv("NNTR_RECQ_TRACE") != nullptr;
  if (rqt_on) {
    char nm[96] = {0};
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(nm) - 1, nm,
                    nullptr);
    rqt_op("KERNEL", nm);
  }

  NNTR_THROW_IF(error_code != CL_SUCCESS, std::runtime_error)
    << "clEnqueueNDRangeKernel failed. OpenCL error code: " << error_code
    << ", error: " << OpenCLErrorCodeToString(error_code);

  if (track && local_evt != nullptr) {
    char nm[128] = {0};
    if (clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(nm) - 1, nm,
                        nullptr) != CL_SUCCESS)
      nm[0] = '\0';
    std::string key(nm);
    if (!next_prof_label_.empty())
      key += next_prof_label_;
    profRecs().push_back({std::move(key), local_evt});
  }
  // consume the per-call shape label regardless of tracking, so it never
  // leaks onto a subsequent kernel's profile entry.
  next_prof_label_.clear();
}

void CommandQueueManager::dumpProfile(const char *tag) {
  if (!profEnabled())
    return;
  auto &recs = profRecs();
  if (command_queue_)
    clFinish(command_queue_);

  struct Agg {
    double total_ns = 0.0;
    unsigned long count = 0;
  };
  std::unordered_map<std::string, Agg> agg;
  double grand_ns = 0.0;
  // ordered timeline (name, start, end) for inter-kernel GPU-idle attribution
  std::vector<std::string> tl_name;
  std::vector<cl_ulong> tl_start, tl_end;
  for (auto &r : recs) {
    cl_ulong start = 0, end = 0;
    if (r.evt) {
      clGetEventProfilingInfo(r.evt, CL_PROFILING_COMMAND_START,
                              sizeof(start), &start, nullptr);
      clGetEventProfilingInfo(r.evt, CL_PROFILING_COMMAND_END, sizeof(end),
                              &end, nullptr);
      if (end > start) {
        double ns = (double)(end - start);
        agg[r.name].total_ns += ns;
        grand_ns += ns;
        tl_name.push_back(r.name);
        tl_start.push_back(start);
        tl_end.push_back(end);
      }
      agg[r.name].count++;
      clReleaseEvent(r.evt);
    }
  }
  recs.clear();

  // Inter-kernel GPU-idle: gap between kernel i's end and i+1's start = host-
  // bound dispatch overhead (the GPU waiting for the host to enqueue/prep the
  // next kernel). Attribute each gap to the "A -> B" transition (base names,
  // shape label stripped) so we see where the idle concentrates.
  auto base = [](const std::string &s) {
    auto p = s.find(':');
    return p == std::string::npos ? s : s.substr(0, p);
  };
  std::unordered_map<std::string, Agg> idle;
  // Per-transition gap list for distribution stats (first/median/max). An
  // aggregate "avg x count" can hide a one-time cost diluted across N
  // instances -- exactly how the first-prefill kernel-program builds
  // masqueraded as a 39ms/layer issue tax (G9).
  std::unordered_map<std::string, std::vector<double>> idle_gaps;
  double total_idle_ns = 0.0;
  for (size_t i = 1; i < tl_start.size(); ++i) {
    if (tl_start[i] > tl_end[i - 1]) {
      double g = (double)(tl_start[i] - tl_end[i - 1]);
      std::string key = base(tl_name[i - 1]) + " -> " + base(tl_name[i]);
      idle[key].total_ns += g;
      idle[key].count++;
      idle_gaps[key].push_back(g);
      total_idle_ns += g;
    }
  }

  std::vector<std::pair<std::string, Agg>> sorted(agg.begin(), agg.end());
  std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) {
    return a.second.total_ns > b.second.total_ns;
  });

  printf("\n==== GPU kernel profile [%s] : true on-device time ====\n",
         tag ? tag : "");
  printf("  %-34s %10s %8s %10s %7s\n", "kernel", "total_ms", "calls",
         "avg_us", "%%");
  for (auto &kv : sorted) {
    double tot_ms = kv.second.total_ns / 1e6;
    double avg_us = kv.second.count
                      ? (kv.second.total_ns / 1e3) / (double)kv.second.count
                      : 0.0;
    double pct = grand_ns > 0.0 ? 100.0 * kv.second.total_ns / grand_ns : 0.0;
    printf("  %-34s %10.2f %8lu %10.2f %6.1f%%\n", kv.first.c_str(), tot_ms,
           kv.second.count, avg_us, pct);
  }
  printf("  %-34s %10.2f\n", "TOTAL (sum of kernel GPU time)", grand_ns / 1e6);

  // host-bound inter-kernel idle (GPU waiting for host between dispatches)
  std::vector<std::pair<std::string, Agg>> idle_sorted(idle.begin(), idle.end());
  std::sort(idle_sorted.begin(), idle_sorted.end(),
            [](const auto &a, const auto &b) {
              return a.second.total_ns > b.second.total_ns;
            });
  printf("\n  --- inter-kernel GPU-idle (host-bound dispatch overhead) ---\n");
  printf("  %-44s %10s %8s %9s %9s %9s %9s\n", "transition (A -> B)", "idle_ms",
         "count", "avg_us", "first_us", "p50_us", "max_us");
  size_t shown = 0;
  for (auto &kv : idle_sorted) {
    if (shown++ >= 15)
      break;
    double ms = kv.second.total_ns / 1e6;
    double avg_us = kv.second.count
                      ? (kv.second.total_ns / 1e3) / (double)kv.second.count
                      : 0.0;
    double pct = total_idle_ns > 0.0 ? 100.0 * kv.second.total_ns / total_idle_ns : 0.0;
    // distribution: first occurrence vs median vs max -- uniform per-layer
    // cost has first~p50~max; a one-time cost has max>>p50.
    auto &gaps = idle_gaps[kv.first];
    double first_us = gaps.empty() ? 0.0 : gaps.front() / 1e3;
    double max_us = 0.0;
    for (double g : gaps)
      max_us = std::max(max_us, g / 1e3);
    std::vector<double> tmp(gaps);
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
    double p50_us = tmp.empty() ? 0.0 : tmp[tmp.size() / 2] / 1e3;
    printf("  %-44s %10.2f %8lu %9.1f %9.1f %9.1f %9.1f  (%4.1f%%)\n",
           kv.first.c_str(), ms, kv.second.count, avg_us, first_us, p50_us,
           max_us, pct);
  }
  printf("  %-44s %10.2f\n", "TOTAL inter-kernel idle", total_idle_ns / 1e6);
  printf("=========================================================\n\n");
  fflush(stdout);
}

} // namespace nntrainer::opencl

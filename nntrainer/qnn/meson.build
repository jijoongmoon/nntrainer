# nntrainer/nntrainer/qnn/meson.build
# Defines qnn_sources_abs and qnn_inc_abs for the ndk-build Android.mk module.
# Gated by the parent (enable-npu). The QNN 2.47 headers + sample utilities
# are auto-generated under jni/vendor/ at configure time from a locally-installed
# Qualcomm QNN SDK. Provide the SDK via -Dqnn-sdk-root=PATH or $QNN_SDK_ROOT.

qnn_src_dir = meson.current_source_dir()

# ── Auto-generate the vendor tree if it is missing ───────────────────────────
# The sentinel file is a stable header that must exist after a successful run.
fs = import('fs')
_vendor_sentinel = qnn_src_dir / 'jni/vendor/QNN/QnnCommon.h'
if not fs.exists(_vendor_sentinel)
  _gen_script = qnn_src_dir / 'jni/tools/update_qnn_version.py'
  _gen_cmd = ['python3', _gen_script]
  _sdk_root_opt = get_option('qnn-sdk-root')
  if _sdk_root_opt != ''
    _gen_cmd += ['--qnn-sdk-root=' + _sdk_root_opt]
  endif
  # When qnn-sdk-root is empty the script falls back to $QNN_SDK_ROOT, which
  # run_command inherits from the meson configure environment.
  _gen_result = run_command(_gen_cmd, check: false)
  if _gen_result.returncode() != 0
    error(
      'QNN vendor tree generation failed.\n' +
      'stdout: ' + _gen_result.stdout() + '\n' +
      'stderr: ' + _gen_result.stderr() + '\n' +
      'Set -Dqnn-sdk-root=<path> or export QNN_SDK_ROOT=<path> to your ' +
      'Qualcomm QNN SDK root (e.g. ~/qairt/2.47.0.260601).'
    )
  endif
  message('QNN vendor tree generated from SDK.')
endif
# ─────────────────────────────────────────────────────────────────────────────

qnn_sources_abs = [
  qnn_src_dir / 'jni/vendor/Log/Logger.cpp',
  qnn_src_dir / 'jni/vendor/Log/LogUtils.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/common/GetOpt.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/common/StringOp.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/linux/Directory.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/linux/DynamicLoading.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/linux/FileOp.cpp',
  qnn_src_dir / 'jni/vendor/PAL/src/linux/Path.cpp',
  qnn_src_dir / 'jni/vendor/Utils/DataUtil.cpp',
  qnn_src_dir / 'jni/vendor/Utils/DynamicLoadUtil.cpp',
  qnn_src_dir / 'jni/vendor/Utils/IOTensor.cpp',
  qnn_src_dir / 'jni/vendor/Utils/QnnDlcUtils.cpp',
  qnn_src_dir / 'jni/vendor/Utils/QnnSampleAppUtils.cpp',
  qnn_src_dir / 'jni/vendor/WrapperUtils/QnnWrapperUtils.cpp',
  qnn_src_dir / 'jni/vendor/qnn-api/dlwrap.cpp',
  qnn_src_dir / 'jni/vendor/qnn-api/BackendExtensions.cpp',
  qnn_src_dir / 'jni/vendor/qnn-api/ResourceManager.cpp',
  qnn_src_dir / 'jni/QNNGraph.cpp',
  qnn_src_dir / 'jni/qnn_properties.cpp',
  qnn_src_dir / 'qnn_context.cpp',
  qnn_src_dir / 'jni/qnn_rpc_manager.cpp',
  qnn_src_dir / 'jni/rpc_mem.cpp',
]

qnn_inc_abs = [
  qnn_src_dir,
  qnn_src_dir / 'jni',
  qnn_src_dir / 'jni/vendor',
  qnn_src_dir / 'jni/vendor/QNN',
  qnn_src_dir / 'jni/vendor/Log',
  qnn_src_dir / 'jni/vendor/PAL/include',
  qnn_src_dir / 'jni/vendor/Utils',
  qnn_src_dir / 'jni/vendor/WrapperUtils',
  qnn_src_dir / 'jni/vendor/qnn-api',
]

qnn_inc = include_directories(
  '.',
  'jni',
  'jni/vendor',
  'jni/vendor/QNN',
  'jni/vendor/Log',
  'jni/vendor/PAL/include',
  'jni/vendor/Utils',
  'jni/vendor/WrapperUtils',
  'jni/vendor/qnn-api',
)

# Expose qnn headers (qnn_context.h 등) to the rest of nntrainer.
nntrainer_inc += qnn_inc
nntrainer_inc_abs += qnn_src_dir

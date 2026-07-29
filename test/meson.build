fs = import('fs')
nntrainer_test_resdir = nntrainer_resdir / 'test'

if build_machine.system() == 'windows'
  nntrainer_test_resdir_win = nntrainer_test_resdir.replace('/', '\\')
  nntrainer_test_resdir_models_win = (nntrainer_test_resdir / 'test_models/').replace('/', '\\')
  test_models_dir_win = (meson.current_source_dir() / 'test_models/').replace('/', '\\')
  if not fs.exists (nntrainer_test_resdir_win)
    run_command([prog_win_cmd, '/C', 'mkdir', nntrainer_test_resdir_win], check: true)
  endif
  if not fs.exists (nntrainer_test_resdir_models_win)
    run_command([prog_win_cmd, '/C', 'mkdir', nntrainer_test_resdir_models_win], check: true)
  endif
  run_command(['xcopy', '/Y', test_models_dir_win, nntrainer_test_resdir_models_win, '/i', '/s'], check: true)
else
  run_command(['mkdir', '-p', nntrainer_test_resdir], check: true)
endif

nntrainer_test_inc = include_directories('./include')


nntrainer_testutil_lib = static_library(
  'nntrainer_test_util',
  'nntrainer_test_util.cpp',
  dependencies: [nntrainer_dep],
  include_directories: nntrainer_test_inc
)

nntrainer_testutil_dep = declare_dependency(
  link_with: nntrainer_testutil_lib,
  include_directories: nntrainer_test_inc
)

nntrainer_test_deps = [
  gmock_dep,
  gtest_dep,
  nntrainer_dep,
  nntrainer_testutil_dep
]

# this is depedency is for the gtest with main included
nntrainer_test_main_deps = [
  gmock_dep,
  gtest_main_dep,
  nntrainer_dep,
  nntrainer_testutil_dep
]

# KAI qai8dxp/qsi4cxp int4 packing sanity checks (standalone: no gtest, own
# main()). ARM-only: they link the KleidiAI reference packer/matmul sources
# under nntrainer/tensor/cpu_backend/arm/, which meson only compiles into
# nntrainer_dep for aarch64/arm/android builds (see
# nntrainer/tensor/cpu_backend/meson.build), and they exercise
# __ARM_FEATURE_MATMUL_INT8 (i8mm) code paths that this project always
# enables for native aarch64 (see the top-level meson.build extra_defines
# block). Gate mirrors the existing aarch64-only
# `unittest_nntrainer_kleidiai` target below.
if host_machine.cpu_family() == 'aarch64'
  q4_kai_standalone_targets = [
    'q4_roundtrip_kai',
    'q4_kai_bytecompat',
  ]
  foreach t : q4_kai_standalone_targets
    q4_kai_exe = executable(
      t,
      t + '.cpp',
      dependencies: nntrainer_test_deps,
      install: get_option('enable-test'),
      install_dir: application_install_dir
    )
    test(t, q4_kai_exe, timeout: test_timeout, suite: 'unittests')
  endforeach

  # The fp16 matmul path additionally needs the enable-fp16-gated
  # matmul_clamp_f16_qai8dxp_qsi4cxp kernels + qai8dxp_f16 LHS packer
  # (nntrainer/tensor/cpu_backend/arm/kai_interface/kai/meson.build only
  # enters that subdir when enable-fp16 is set), so this one test gates on
  # both.
  if get_option('enable-fp16')
    q4_kai_matmul_device_exe = executable(
      'q4_kai_matmul_device',
      'q4_kai_matmul_device.cpp',
      dependencies: nntrainer_test_deps,
      install: get_option('enable-test'),
      install_dir: application_install_dir
    )
    test('q4_kai_matmul_device', q4_kai_matmul_device_exe,
      timeout: test_timeout, suite: 'unittests')
  endif
endif

if enable_capi
  subdir('tizen_capi')
endif

if enable_ccapi
  subdir('input_gen')
  subdir('ccapi')
  subdir('unittest')
endif

nnstreamer_test_dep = dependency('nnstreamer-test-devel', required: false)

if get_option('enable-nnstreamer-tensor-filter').enabled() and nnstreamer_test_dep.found()
  subdir('nnstreamer')
endif

run_command('sh', '-c', 'rm -rf "@0@" && cp -lr "@1@" "@0@"'.format(nntrainer_test_resdir / 'test_models',
meson.current_source_dir() / 'test_models/'), check: true)

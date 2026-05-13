project('quick-dot-ai', 'c', 'cpp',
  version: '0.2.0',
  license: ['apache-2.0'],
  meson_version: '>=0.55.0',
  default_options: [
    'werror=false',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++17',
    'buildtype=release'
  ]
)

# ── Compiler setup ──────────────────────────────────────────────────────
cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

# ── Platform detection ──────────────────────────────────────────────────
_platform_opt = get_option('platform')
if _platform_opt == 'auto'
  is_android = host_machine.system() == 'android'
elif _platform_opt == 'android'
  is_android = true
else
  is_android = false
endif

# ── QNN guard: android only ─────────────────────────────────────────────
enable_qnn = get_option('enable-qnn')
if enable_qnn and not is_android
  warning('enable-qnn is only supported on android. Disabling QNN.')
  enable_qnn = false
endif

# ── Path configuration ─────────────────────────────────────────────────
project_root   = meson.current_source_dir()
nntrainer_root = project_root / 'nntrainer'
causallm_root  = nntrainer_root / 'Applications' / 'CausalLM'

# ── Find pre-built nntrainer libraries ──────────────────────────────────
if is_android
  _nntr_result = nntrainer_root / 'builddir' / 'android_build_result'
  _nntr_libdir = _nntr_result / 'lib' / 'arm64-v8a'
  _nntr_incdir = _nntr_result / 'include' / 'nntrainer'

  libnntrainer = cxx.find_library('nntrainer', dirs: [_nntr_libdir])
  libccapi     = cxx.find_library('ccapi-nntrainer', dirs: [_nntr_libdir])

  nntrainer_inc_args = ['-I' + _nntr_incdir]
else
  _nntr_builddir = nntrainer_root / get_option('nntrainer_builddir')

  libnntrainer = cxx.find_library('nntrainer',
    dirs: [_nntr_builddir / 'nntrainer'])
  libccapi     = cxx.find_library('ccapi-nntrainer',
    dirs: [_nntr_builddir / 'api' / 'ccapi'])

  # nntrainer source-tree includes (x86 build)
  # NOTE: Do NOT include the builddir itself — ruy/time.h shadows system <time.h>
  nntrainer_inc_args = []
  foreach p : [
    nntrainer_root / 'api' / 'ccapi' / 'include',
    nntrainer_root / 'api',
    nntrainer_root / 'nntrainer',
    nntrainer_root / 'nntrainer' / 'layers',
    nntrainer_root / 'nntrainer' / 'layers' / 'loss',
    nntrainer_root / 'nntrainer' / 'models',
    nntrainer_root / 'nntrainer' / 'graph',
    nntrainer_root / 'nntrainer' / 'compiler',
    nntrainer_root / 'nntrainer' / 'optimizers',
    nntrainer_root / 'nntrainer' / 'tensor',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'fallback',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'cblas_interface',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'x86',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'ggml_interface',
    nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'ggml_interface' / 'nntr_ggml_impl',
    nntrainer_root / 'nntrainer' / 'utils',
    nntrainer_root / 'nntrainer' / 'dataset',
    nntrainer_root / 'nntrainer' / 'schema',
  ]
    nntrainer_inc_args += ['-I' + p]
  endforeach
endif

nntrainer_dep = declare_dependency(
  dependencies: [libnntrainer, libccapi],
  compile_args: nntrainer_inc_args,
)

# ── OpenMP / threads ────────────────────────────────────────────────────
openmp_dep = dependency('openmp', required: false)
thread_dep = dependency('threads')

# ── Android log library ─────────────────────────────────────────────────
if is_android
  log_dep = cxx.find_library('log', required: true)
else
  log_dep = dependency('', required: false)
endif

# ── CausalLM includes ──────────────────────────────────────────────────
causallm_auto_inc_dirs = run_command('find',
  causallm_root / 'layers',
  causallm_root / 'models',
  causallm_root / 'res',
  '-type', 'd',
  check: true).stdout().strip().split('\n')

causallm_inc_args = ['-I' + causallm_root]
foreach d : causallm_auto_inc_dirs
  causallm_inc_args += ['-I' + d]
endforeach

# ── Tokenizer: platform-specific ────────────────────────────────────────
if is_android
  tokenizer_path = causallm_root / 'lib' / 'libtokenizers_android_c.a'
else
  tokenizer_path = causallm_root / 'lib' / 'libtokenizers_c.a'
endif

# ── Extra defines ───────────────────────────────────────────────────────
extra_defines = []
if enable_qnn
  extra_defines += '-DENABLE_QNN=1'
endif

# FP16 is ARM-only (__fp16 type not available on x86)
if get_option('enable-fp16') and is_android
  extra_defines += ['-DENABLE_FP16=1', '-DUSE__FP16=1']
endif

foreach d : extra_defines
  add_project_arguments(d, language: ['c', 'cpp'])
endforeach

# Android-specific ARM flags
if is_android
  _arm_args = [
    '-D__ARM_NEON__=1',
    '-DUSE_NEON=1',
  ]
  foreach a : _arm_args
    add_project_arguments(a, language: ['c', 'cpp'])
  endforeach
endif

# ── Build components ────────────────────────────────────────────────────
# Each subdir guards itself with its own enable option.
subdir('src')
subdir('qnn')
subdir('api')
subdir('api-app')

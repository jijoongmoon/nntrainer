# src/meson.build — included via subdir('src') from root
# Receives from parent: nntrainer_dep, causallm_root, causallm_inc_args,
#   tokenizer_path, openmp_dep, is_android, enable_qnn

# ── CausalLM sources ───────────────────────────────────────────────────
# NOTE: Do NOT include causallm_root / 'api' — the API layer is provided
# by quick-dot-ai-api. Including it would shadow symbols via interposition.
causallm_auto_src = run_command('find',
  causallm_root / 'layers',
  causallm_root / 'models',
  causallm_root / 'huggingface_tokenizer.cpp',
  causallm_root / 'llm_util.cpp',
  causallm_root / 'chat_template.cpp',
  '-name', '*.cpp',
  '-not', '-name', 'main.cpp',
  '-not', '-name', 'test_api.cpp',
  check: true).stdout().strip().split('\n')

streamer_src = [
  meson.project_source_root() / 'api' / 'streamer.cpp',
]

# ── Android system libraries ────────────────────────────────────────────
if is_android
  log_dep = cxx.find_library('log')
else
  log_dep = dependency('', required: false)
endif

# ── Build CausalLM as shared library ───────────────────────────────────
streamer_inc = include_directories('../api')

causallm_lib = shared_library('causallm',
    causallm_auto_src,
    streamer_src,
    include_directories: [streamer_inc],
    dependencies: [nntrainer_dep, openmp_dep, log_dep],
    cpp_args: causallm_inc_args,
    link_args: [tokenizer_path],
    install: true,
)

causallm_dep = declare_dependency(
    link_with: causallm_lib,
    compile_args: causallm_inc_args,
)

# ── Extension model sources ────────────────────────────────────────────
quick_dot_ai_src = []
quick_dot_ai_inc = []

subdir('models')

# ── Build model plugin (.so for LD_PRELOAD mode) ──────────────────────
quick_dot_ai_plugin = shared_library('quick_dot_ai',
    quick_dot_ai_src,
    include_directories: [quick_dot_ai_inc],
    dependencies: [nntrainer_dep, causallm_dep, openmp_dep, log_dep],
    install: true,
)

quick_dot_ai_dep = declare_dependency(
    link_with: quick_dot_ai_plugin,
    include_directories: [ext_inc, quick_dot_ai_inc],
)

# ── Static lib for linking into executable (ensures constructor runs) ──
quick_dot_ai_static = static_library('quick_dot_ai_static',
    quick_dot_ai_src,
    include_directories: [ext_inc, quick_dot_ai_inc],
    dependencies: [nntrainer_dep, causallm_dep, openmp_dep, log_dep],
)

# ── Build executable (original main.cpp + custom plugin) ──────────────
executable('quick_dot_ai',
    causallm_root / 'main.cpp',
    include_directories: [ext_inc, quick_dot_ai_inc],
    dependencies: [nntrainer_dep, causallm_dep, openmp_dep, log_dep],
    link_whole: quick_dot_ai_static,
    cpp_args: causallm_inc_args,
    install: true,
)

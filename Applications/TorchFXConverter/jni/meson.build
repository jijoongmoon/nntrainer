# TorchFXConverter model test build
#
# Builds test executables from auto-generated model code (converter.py output).
# Each test links against CausalLM's custom layer libraries and NNTrainer.

converter_test_inc = [
  include_directories('.'),
]

# Core custom layer dependencies (from CausalLM/layers)
converter_layer_deps = [
  causallm_rms_norm_dep,
  causallm_swiglu_dep,
  causallm_mha_core_dep,
  causallm_embedding_layer_dep,
  causallm_reshaped_rms_norm_dep,
  causallm_tie_word_embedding_dep,
]

# --- Qwen3 tiny model test ---
if fs.exists(meson.current_source_dir() / 'test_model.cpp')
  executable('converter_qwen3_test',
    [
      meson.current_source_dir() / 'main.cpp',
      meson.current_source_dir() / 'test_model.cpp',
    ],
    include_directories: converter_test_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep, converter_layer_deps],
    cpp_args: [
      '-DGENERATED_MODEL_CLASS=Qwen3CausalLM',
      '-DGENERATED_MODEL_HEADER="test_model.h"',
    ],
  )
endif

qwen3_5_moe_src = [
    meson.current_source_dir() / 'qwen3_5_moe_causallm.cpp',
]

qwen3_5_moe_inc = include_directories('.')

# The two custom layers this model adds. They live in their own shared library
# (the shape every other models/*/meson.build uses) and are published through
# causallm_layer_dependencies so libcausallm links their vtables -- a layer that
# is registered from a model but never linked in shows up as an undefined vtable
# at link time, which is the rms_reverse_norm failure documented in
# Applications/CausalLM/meson.build.
causallm_gdn_layer_src_abs = [
    meson.current_source_dir() / 'gated_delta_net_layer.cpp',
    meson.current_source_dir() / 'broadcast_mul_layer.cpp',
]

causallm_gdn_layer = shared_library(
    'gated_delta_net_layer',
    causallm_gdn_layer_src_abs,
    include_directories: [causallm_layer_inc, qwen3_5_moe_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_gdn_layer_dep = declare_dependency(
    link_with: causallm_gdn_layer,
    include_directories: qwen3_5_moe_inc
)

causallm_src += qwen3_5_moe_src
causallm_inc += qwen3_5_moe_inc
causallm_layer_dependencies += [causallm_gdn_layer_dep]

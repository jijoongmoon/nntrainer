causallm_layer_inc_abs = [meson.current_source_dir()]
causallm_layer_inc = [include_directories('.')]

causallm_rms_norm_src_abs = [meson.current_source_dir() / 'rms_norm.cpp']
causallm_rms_norm_gpu_src_abs = [meson.current_source_dir() / 'rms_norm_gpu.cpp']
# swiglu merged into core nntrainer::SwiGLULayer (nntrainer/layers/llm/) [T12]
# tie_word_embedding promoted to core nntrainer/layers/llm/ [T12]
# lm_head promoted to core nntrainer/layers/llm/ [T12 Wave B] — no app plugin
causallm_mha_core_abs = [meson.current_source_dir()/ 'mha_core.cpp']
causallm_embedding_src_abs = [meson.current_source_dir() / 'embedding_layer.cpp']
causallm_reshaped_rms_norm_src_abs = [meson.current_source_dir() / 'reshaped_rms_norm.cpp']
# qkv_layer promoted to core nntrainer/layers/llm/ [T12]
causallm_rms_reverse_norm_src_abs = [meson.current_source_dir() / 'rms_reverse_norm.cpp']
causallm_embedding_pooling_src_abs = [meson.current_source_dir() / 'embedding_pooling_layer.cpp']
causallm_embedding_normalize_src_abs = [meson.current_source_dir() / 'embedding_normalize_layer.cpp']
causallm_embedding_injection_src_abs = [meson.current_source_dir() / 'embedding_injection.cpp']
causallm_deberta_attention_src_abs = [meson.current_source_dir() / 'deberta_attention_layer.cpp']
causallm_shared_fully_connected_src_abs = [meson.current_source_dir() / 'shared_fully_connected_layer.cpp']
causallm_per_layer_slice_src_abs = [meson.current_source_dir() / 'per_layer_slice.cpp', meson.current_source_dir() / 'per_layer_slice_gpu.cpp']
# scalar_multiply (+gpu) promoted to core nntrainer/layers/llm/ [T12]
# logit_softcapping promoted to core nntrainer/layers/llm/ [T12]

causallm_embedding_injection = shared_library(
    'embedding_injection_layer',
    causallm_embedding_injection_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_embedding_injection_dep = declare_dependency(
    link_with: causallm_embedding_injection,
    include_directories: causallm_layer_inc
)

causallm_vision_rope_src_abs = [meson.current_source_dir() / 'vision_rope.cpp']
causallm_vision_rope = shared_library(
    'vision_rope_layer',
    causallm_vision_rope_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_vision_rope_dep = declare_dependency(
    link_with: causallm_vision_rope,
    include_directories: causallm_layer_inc
)

causallm_mrope_apply_src_abs = [meson.current_source_dir() / 'mrope_apply.cpp']
causallm_mrope_apply = shared_library(
    'mrope_apply_layer',
    causallm_mrope_apply_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_mrope_apply_dep = declare_dependency(
    link_with: causallm_mrope_apply,
    include_directories: causallm_layer_inc
)

causallm_snake_beta_src_abs = [meson.current_source_dir() / 'snake_beta.cpp']
causallm_snake_beta = shared_library(
    'snake_beta_layer',
    causallm_snake_beta_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_snake_beta_dep = declare_dependency(
    link_with: causallm_snake_beta,
    include_directories: causallm_layer_inc
)

causallm_conv1d_transpose_src_abs = [meson.current_source_dir() / 'conv1d_transpose.cpp']
causallm_conv1d_transpose = shared_library(
    'conv1d_transpose_layer',
    causallm_conv1d_transpose_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_conv1d_transpose_dep = declare_dependency(
    link_with: causallm_conv1d_transpose,
    include_directories: causallm_layer_inc
)

causallm_antialiased_snake_src_abs = [meson.current_source_dir() / 'antialiased_snake.cpp']
causallm_antialiased_snake = shared_library(
    'antialiased_snake_layer',
    causallm_antialiased_snake_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_antialiased_snake_dep = declare_dependency(
    link_with: causallm_antialiased_snake,
    include_directories: causallm_layer_inc
)

causallm_scale_src_abs = [meson.current_source_dir() / 'scale_layer.cpp']
causallm_scale = shared_library(
    'scale_layer',
    causallm_scale_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_scale_dep = declare_dependency(
    link_with: causallm_scale,
    include_directories: causallm_layer_inc
)

foreach _dl : ['dit_rope', 'dit_modulate', 'dit_gate']
    _lib = shared_library(_dl + '_layer',
        [meson.current_source_dir() / (_dl + '.cpp')],
        include_directories: causallm_layer_inc,
        dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
        install: true,
        install_dir: application_install_dir
    )
    set_variable('causallm_' + _dl + '_dep', declare_dependency(
        link_with: _lib,
        include_directories: causallm_layer_inc
    ))
endforeach

causallm_vision_attention_src_abs = [meson.current_source_dir() / 'vision_attention.cpp']
causallm_vision_attention = shared_library(
    'vision_attention_layer',
    causallm_vision_attention_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_vision_attention_dep = declare_dependency(
    link_with: causallm_vision_attention,
    include_directories: causallm_layer_inc
)

causallm_rms_norm = shared_library(
    'rms_norm_layer', 
    causallm_rms_norm_src_abs, 
    include_directories: causallm_layer_inc, 
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_rms_norm_dep = declare_dependency(
    link_with: causallm_rms_norm,
    include_directories: causallm_layer_inc
)

# GPU-routed RMSNorm (RMSNormLayerGPU). Referenced unconditionally by
# transformer.cpp; the GPU dispatch inside it only fires under enable-opencl.
causallm_rms_norm_gpu = shared_library(
    'rms_norm_gpu_layer',
    causallm_rms_norm_gpu_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_rms_norm_gpu_dep = declare_dependency(
    link_with: causallm_rms_norm_gpu,
    include_directories: causallm_layer_inc
)

# swiglu merged into core nntrainer::SwiGLULayer (nntrainer/layers/llm/) [T12]

# tie_word_embedding promoted to core nntrainer/layers/llm/ [T12] — no app plugin

# lm_head promoted to core libnntrainer [T12 Wave B]

causallm_mha_core = shared_library(
    'mha_core_layer',
    causallm_mha_core_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_mha_core_dep = declare_dependency(
    link_with: causallm_mha_core,
    include_directories: causallm_layer_inc
)

causallm_embedding_layer = shared_library(
    'embedding_layer',
    causallm_embedding_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_embedding_layer_dep = declare_dependency(
    link_with: causallm_embedding_layer,
    include_directories: causallm_layer_inc
)

causallm_embedding_pooling_layer = shared_library(
    'embedding_pooling_layer',
    causallm_embedding_pooling_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_embedding_pooling_layer_dep = declare_dependency(
    link_with: causallm_embedding_pooling_layer,
    include_directories: causallm_layer_inc
)

causallm_embedding_normalize_layer = shared_library(
    'embedding_normalize_layer',
    causallm_embedding_normalize_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_embedding_normalize_layer_dep = declare_dependency(
    link_with: causallm_embedding_normalize_layer,
    include_directories: causallm_layer_inc
)

causallm_reshaped_rms_norm = shared_library(
    'reshaped_rms_norm_layer', 
    causallm_reshaped_rms_norm_src_abs, 
    include_directories: causallm_layer_inc, 
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_reshaped_rms_norm_dep = declare_dependency(
    link_with: causallm_reshaped_rms_norm,
    include_directories: causallm_layer_inc
)

# qkv_layer promoted to core nntrainer/layers/llm/ [T12] — no app plugin

causallm_deberta_attention_layer = shared_library(
    'deberta_attention_layer',
    causallm_deberta_attention_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_deberta_attention_layer_dep = declare_dependency(
    link_with: causallm_deberta_attention_layer,
    include_directories: causallm_layer_inc
)

causallm_shared_fully_connected_layer = shared_library(
    'shared_fully_connected_layer',
    causallm_shared_fully_connected_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_shared_fully_connected_layer_dep = declare_dependency(
    link_with: causallm_shared_fully_connected_layer,
    include_directories: causallm_layer_inc
)

causallm_rms_reverse_norm = shared_library(
    'rms_reverse_norm_layer',
    causallm_rms_reverse_norm_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_rms_reverse_norm_dep = declare_dependency(
    link_with: causallm_rms_reverse_norm,
    include_directories: causallm_layer_inc
)

causallm_per_layer_slice = shared_library(
    'per_layer_slice_layer',
    causallm_per_layer_slice_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_per_layer_slice_dep = declare_dependency(
    link_with: causallm_per_layer_slice,
    include_directories: causallm_layer_inc
)

# scalar_multiply (+gpu) promoted to core nntrainer/layers/llm/ [T12] — no app plugin

# logit_softcapping promoted to core nntrainer/layers/llm/ [T12] — no app plugin

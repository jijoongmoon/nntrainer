causallm_layer_inc_abs = [meson.current_source_dir()]
causallm_layer_inc = [include_directories('.')]

causallm_rms_norm_src_abs = [meson.current_source_dir() / 'rms_norm.cpp']
causallm_rms_norm_gpu_src_abs = [meson.current_source_dir() / 'rms_norm_gpu.cpp']
causallm_swiglu_src_abs = [meson.current_source_dir() / 'swiglu.cpp']
# tie_word_embedding promoted to core nntrainer/layers/llm/ [T12]
causallm_lmhead_src_abs = [meson.current_source_dir() / 'lm_head.cpp']
causallm_mha_core_abs = [meson.current_source_dir()/ 'mha_core.cpp']
causallm_embedding_src_abs = [meson.current_source_dir() / 'embedding_layer.cpp']
causallm_reshaped_rms_norm_src_abs = [meson.current_source_dir() / 'reshaped_rms_norm.cpp']
# qkv_layer promoted to core nntrainer/layers/llm/ [T12]
causallm_rms_reverse_norm_src_abs = [meson.current_source_dir() / 'rms_reverse_norm.cpp']
causallm_embedding_pooling_src_abs = [meson.current_source_dir() / 'embedding_pooling_layer.cpp']
causallm_embedding_normalize_src_abs = [meson.current_source_dir() / 'embedding_normalize_layer.cpp']
causallm_deberta_attention_src_abs = [meson.current_source_dir() / 'deberta_attention_layer.cpp']
causallm_shared_fully_connected_src_abs = [meson.current_source_dir() / 'shared_fully_connected_layer.cpp']
causallm_per_layer_slice_src_abs = [meson.current_source_dir() / 'per_layer_slice.cpp', meson.current_source_dir() / 'per_layer_slice_gpu.cpp']
# scalar_multiply (+gpu) promoted to core nntrainer/layers/llm/ [T12]
# logit_softcapping promoted to core nntrainer/layers/llm/ [T12]

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

causallm_swiglu = shared_library(
    'swiglu_layer',
    causallm_swiglu_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_swiglu_dep = declare_dependency(
    link_with: causallm_swiglu,
    include_directories: causallm_layer_inc
)

# tie_word_embedding promoted to core nntrainer/layers/llm/ [T12] — no app plugin

causallm_lm_head = shared_library(
    'lm_head',
    causallm_lmhead_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_lm_head_dep = declare_dependency(
    link_with: causallm_lm_head,
    include_directories: causallm_layer_inc
)

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

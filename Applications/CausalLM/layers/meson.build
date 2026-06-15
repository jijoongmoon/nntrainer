causallm_layer_inc_abs = [meson.current_source_dir()]
causallm_layer_inc = [include_directories('.')]

causallm_rms_norm_src_abs = [meson.current_source_dir() / 'rms_norm.cpp']
causallm_swiglu_src_abs = [meson.current_source_dir() / 'swiglu.cpp']
causallm_tie_word_embedding_src_abs = [meson.current_source_dir() / 'tie_word_embedding.cpp']
causallm_lmhead_src_abs = [meson.current_source_dir() / 'lm_head.cpp']
causallm_mha_core_abs = [meson.current_source_dir()/ 'mha_core.cpp']
causallm_embedding_src_abs = [meson.current_source_dir() / 'embedding_layer.cpp']
causallm_reshaped_rms_norm_src_abs = [meson.current_source_dir() / 'reshaped_rms_norm.cpp']
causallm_qkv_layer_src_abs = [meson.current_source_dir() / 'qkv_layer.cpp']
causallm_embedding_pooling_src_abs = [meson.current_source_dir() / 'embedding_pooling_layer.cpp']
causallm_embedding_normalize_src_abs = [meson.current_source_dir() / 'embedding_normalize_layer.cpp']
causallm_embedding_injection_src_abs = [meson.current_source_dir() / 'embedding_injection.cpp']

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

causallm_tie_word_embedding = shared_library(
    'tie_word_embedding_layer',
    causallm_tie_word_embedding_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
causallm_tie_word_embedding_dep = declare_dependency(
    link_with: causallm_tie_word_embedding,
    include_directories: causallm_layer_inc
)

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

causallm_qkv_layer = shared_library(
    'qkv_layer',
    causallm_qkv_layer_src_abs,
    include_directories: causallm_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

causallm_qkv_layer_dep = declare_dependency(
    link_with: causallm_qkv_layer,
    include_directories: causallm_layer_inc
)

cuda_layer_sources = [
  'cuda_fc_layer.cpp',
  'cuda_geglu_layer.cpp',
  'cuda_rmsnorm_layer.cpp',
]

foreach s : cuda_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

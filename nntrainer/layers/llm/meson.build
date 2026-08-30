# Backend-neutral LLM layers. Compiled straight into libnntrainer (the loss/
# subdir pattern), not as per-layer shared_library plugins. Add sources
# incrementally, one per layer.
llm_layer_sources = [
  'geglu_layer.cpp',
]

llm_layer_headers = [
  'geglu_layer.h',
]

foreach s : llm_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : llm_layer_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

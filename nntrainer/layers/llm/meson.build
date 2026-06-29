# Param-general LLM layers promoted from Applications/CausalLM/layers/ [T12].
# Compiled straight into libnntrainer (the loss/ subdir pattern), NOT per-layer
# shared_library plugins. Add sources incrementally, one per promotion.
llm_layer_sources = [
  'logit_softcapping.cpp',
]

llm_layer_headers = [
  'logit_softcapping.h',
]

foreach s : llm_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : llm_layer_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

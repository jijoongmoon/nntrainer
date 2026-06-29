# Param-general LLM layers promoted from Applications/CausalLM/layers/ [T12].
# Compiled straight into libnntrainer (the loss/ subdir pattern), NOT per-layer
# shared_library plugins. Add sources incrementally, one per promotion.
llm_layer_sources = [
  'logit_softcapping.cpp',
  'scalar_multiply.cpp',
  'qkv_layer.cpp',
  'geglu_layer.cpp',
]

llm_layer_headers = [
  'logit_softcapping.h',
  'scalar_multiply.h',
  'scalar_multiply_gpu.h',
  'tie_word_embedding.h',
  'qkv_layer.h',
  'geglu_layer.h',
]

# scalar_multiply's GPU variant uses OpenCL directly (cl_mem / ClContext), so it
# compiles only when the OpenCL backend is on — mirroring the cl_context.cpp
# registration that consumes it (a CPU-only libnntrainer must not pull it in).
if get_option('enable-opencl')
  llm_layer_sources += 'scalar_multiply_gpu.cpp'
  # tie_word_embedding calls the OpenCL lm_head GEMV (lmhead_gemv_*_cl,
  # clmem_raise/lower) unguarded, so it too is OpenCL-only.
  llm_layer_sources += 'tie_word_embedding.cpp'
endif

foreach s : llm_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : llm_layer_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

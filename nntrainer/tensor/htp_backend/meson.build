htp_backend_sources = [
  'htp_backend.cpp',
  'htp_compute_ops.cpp',
  'htp_mem_allocator.cpp',
]

htp_backend_headers = [
  'htp_backend.h',
]

subdir('hmx_ops')
nntrainer_inc += include_directories('hmx_ops')
nntrainer_inc_abs += meson.current_source_dir() / 'hmx_ops'

foreach s : htp_backend_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : htp_backend_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

log_sources = [
  'Logger.cpp',
  'LogUtils.cpp',
]

log_headers = [
  'Logger.hpp',
  'LogUtils.hpp'
]

foreach s : log_sources
  qnn_sources += meson.current_source_dir() / s
endforeach

qnn_include_dir = quick_dot_ai_prefix / 'include' / 'nntrainer' / 'npu' / 'qnn'

install_subdir ('Log', install_dir : qnn_include_dir )

install_headers(log_headers, install_dir : qnn_include_dir / 'Log')

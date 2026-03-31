# Note: qnn_context.h and qnn_context.cpp have been moved to nntrainer/nntrainer/
# and are now included in the main nntrainer build
# Only qnn_rpc_manager files remain here

qnn_context_sources = [
  meson.current_source_dir() / 'qnn_rpc_manager.cpp',
]

qnn_context_headers = [
  meson.current_source_dir() / 'qnn_rpc_manager.h',
  meson.current_source_dir() / 'qnn_context_var.h',
]

# Add sources to nntrainer
foreach src : qnn_context_sources
  nntrainer_sources += src
endforeach

# Add headers to nntrainer
foreach hdr : qnn_context_headers
  nntrainer_headers += hdr
endforeach

# Include directories for QNN
nntrainer_inc += include_directories('.')
nntrainer_inc_abs += meson.current_source_dir()

# Add include paths so #include <IOTensor.hpp> works
nntrainer_inc += include_directories('qnn', 'qnn/Utils', 'qnn/Log', 'qnn/PAL/include')
nntrainer_inc_abs += meson.current_source_dir() / 'qnn'
nntrainer_inc_abs += meson.current_source_dir() / 'qnn' / 'Utils'
nntrainer_inc_abs += meson.current_source_dir() / 'qnn' / 'Log'
nntrainer_inc_abs += meson.current_source_dir() / 'qnn' / 'PAL' / 'include'

# Include QNN subdirectories
subdir('qnn')
subdir('qnn-api')

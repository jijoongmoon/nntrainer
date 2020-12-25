build_root = meson.build_root()
res_path = meson.current_source_dir() / '..' / 'res'

vgg_sources = [
  'main.cpp'
]

e = executable('vgg_tflite',
  mnist_sources,
  dependencies: [app_utils_dep, nntrainer_dep, tflite_dep],
  include_directories: include_directories('.'),
  install: get_option('install-app'),
  install_dir: application_install_dir
)

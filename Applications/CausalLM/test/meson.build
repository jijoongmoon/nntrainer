golden_test_src = meson.project_source_root() / 'test' / 'unittest' / 'layers' / 'layers_golden_tests.cpp'

test_target = [
  layer_common_test_standalone_files,
  golden_test_src,
  'unittest_rms_norm.cpp',
]

# Copy golden file to test resource directory
rms_norm_golden_src = meson.current_source_dir() / 'rms_norm_training.nnlayergolden'
rms_norm_golden_dst = nntrainer_resdir / 'test' / 'unittest_layers' / 'rms_norm_training.nnlayergolden'
run_command(['cp', '-f', rms_norm_golden_src, rms_norm_golden_dst], check: false)

exe = executable(
  'unittest_causallm_rms_norm',
  test_target,
  dependencies: [
    nntrainer_test_main_deps,
    causallm_rms_norm_dep,
  ],
  include_directories: layer_common_test_inc,
  install: get_option('enable-test'),
  install_dir: application_install_dir
)

test('unittest_causallm_rms_norm', exe,
  args: '--gtest_output=xml:@0@/@1@.xml'.format(meson.build_root(), 'unittest_causallm_rms_norm'),
  timeout: test_timeout,
  suite: 'unittests'
)

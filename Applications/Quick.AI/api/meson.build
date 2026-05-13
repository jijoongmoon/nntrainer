# api/meson.build — builds libquick_dot_ai_api.so

if not get_option('enable-api')
  subdir_done()
endif

api_inc = include_directories('.')

# Extension model headers
quick_dot_ai_inc_args = [
  '-I' + meson.current_source_dir() / '..' / 'src' / 'models' ,
]

if enable_qnn
quick_dot_ai_inc_args += [
  '-I' + meson.current_source_dir() / '..' / 'src' / 'models' / 'qnn',
  '-I' + meson.current_source_dir() / '..' / 'src' / 'models' / 'qnn' / 'gemma4-e2b-qnn',
]
endif

quick_dot_ai_api_lib = shared_library('quick_dot_ai_api',
    'quick_dot_ai_api.cpp',
    'model_config.cpp',
    'streamer.cpp',
    'callback_streamer.cpp',
    include_directories: [api_inc],
    dependencies: [nntrainer_dep, causallm_dep, quick_dot_ai_dep, openmp_dep, thread_dep, log_dep],
    cpp_args: quick_dot_ai_inc_args,
    install: true,
)

quick_dot_ai_api_dep = declare_dependency(
    link_with: quick_dot_ai_api_lib,
    include_directories: api_inc,
)

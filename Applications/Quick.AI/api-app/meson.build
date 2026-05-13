# api-app/meson.build — builds quick_dot_ai_test executable

if not get_option('enable-api-test')
  subdir_done()
endif

executable('quick_dot_ai_test',
    'test_api.cpp',
    dependencies: [quick_dot_ai_api_dep],
    install: true,
)

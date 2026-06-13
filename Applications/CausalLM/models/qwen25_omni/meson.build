qwen25_omni_src = [
    meson.current_source_dir() / 'qwen25_omni_causallm.cpp',
    meson.current_source_dir() / 'qwen25_omni_audio_encoder.cpp',
    meson.current_source_dir() / 'qwen25_omni_audio_causallm.cpp',
    meson.current_source_dir() / 'qwen25_omni_vision_encoder.cpp',
    meson.current_source_dir() / 'qwen25_omni_vision_causallm.cpp',
    meson.current_source_dir() / 'qwen25_omni_talker_causallm.cpp',
    meson.current_source_dir() / 'whisper_mel.cpp',
]

qwen25_omni_inc = include_directories('.')

causallm_inc += qwen25_omni_inc
causallm_src += qwen25_omni_src

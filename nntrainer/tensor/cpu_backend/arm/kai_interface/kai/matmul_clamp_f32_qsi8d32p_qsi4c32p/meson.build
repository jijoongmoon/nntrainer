matmul_clamp_f32_qsi8d32p_qsi4c32p_headers = [
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.h',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h',
    'kai_matmul_clamp_f32_qsi8d32p_qsi4c32p_interface.h'
]

matmul_clamp_f32_qsi8d32p_qsi4c32p_sources = [
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4x4_1x4_neon_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_16x4_neon_i8mm.c',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p4x8_8x4x32_neon_i8mm.c',
]

# SVE micro-kernels must only be added when the build actually targets SVE.
# cxx.has_argument() merely tests whether the compiler *accepts* the +sve flag
# (true on any modern gcc/clang) -- it does NOT reflect the project's effective
# -march, which only enables SVE for arm-arch=armv9.2-a (+sve2). On non-SVE
# aarch64 (e.g. Cortex-A78AE / Jetson Orin) these sources fail to assemble, and
# their consumers in kleidiai_interface_*.cpp are guarded by #ifdef
# __ARM_FEATURE_SVE anyway, so excluding them here keeps source inclusion in
# lock-step with that guard.
if get_option('arm-arch') == 'armv9.2-a'
  matmul_clamp_f32_qsi8d32p_qsi4c32p_headers += [
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.h',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.h',
  ]
  matmul_clamp_f32_qsi8d32p_qsi4c32p_sources += [
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p8x4_1x8_sve_dotprod_asm.S',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod.c',
    'kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p8x8_1x8_sve_dotprod_asm.S',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm.c',
    'kai_matmul_clamp_f32_qsi8d32p4x8_qsi4c32p8x8_16x8_sve_i8mm_asm.S'
  ]
endif

foreach s : matmul_clamp_f32_qsi8d32p_qsi4c32p_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : matmul_clamp_f32_qsi8d32p_qsi4c32p_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

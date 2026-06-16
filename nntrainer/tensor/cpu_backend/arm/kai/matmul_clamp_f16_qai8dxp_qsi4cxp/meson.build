# SPDX-License-Identifier: Apache-2.0

matmul_clamp_f16_qai8dxp_qsi4cxp_headers = [
    'kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.h',
]

matmul_clamp_f16_qai8dxp_qsi4cxp_sources = [
    'kai_matmul_clamp_f16_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm.c',
]

foreach s : matmul_clamp_f16_qai8dxp_qsi4cxp_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach

foreach h : matmul_clamp_f16_qai8dxp_qsi4cxp_headers
  nntrainer_headers += meson.current_source_dir() / h
endforeach

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_kalm_embedding_reference.py
## @brief Generate golden fixtures for the KaLM-Embedding differential test.
##
## KaLM-Embedding is backed by the same nntrainer Qwen2Embedding class as the
## plain Qwen2 embedding model; it only differs in its pooling mode (mean).
## This is a thin wrapper that drives generate_qwen2_embedding_reference.py with
## the "kalm" variant so the two fixtures stay in sync.
##
## Usage:
##   python3 generate_kalm_embedding_reference.py [--seed <int>]

import runpy
import sys
import pathlib

THIS_DIR = pathlib.Path(__file__).resolve().parent

if __name__ == "__main__":
    sys.argv = [str(THIS_DIR / "generate_qwen2_embedding_reference.py"),
                "--variant", "kalm", *sys.argv[1:]]
    runpy.run_path(str(THIS_DIR / "generate_qwen2_embedding_reference.py"),
                   run_name="__main__")

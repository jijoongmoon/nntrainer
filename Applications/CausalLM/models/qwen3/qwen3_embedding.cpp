// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_embedding.cpp
 * @date   07 Jan 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines Qwen3 Embedding model
 */

#include <qwen3_embedding.h>

namespace causallm {

void Qwen3Embedding::registerCustomLayers() {
  SentenceTransformer::registerCustomLayers();
  Qwen3Transformer::registerCustomLayers();
}

} // namespace causallm

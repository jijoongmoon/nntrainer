// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   function.h
 * @date   19 January 2026
 * @brief  This defines a chat format for FunctionGemma
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __GEMMA3_FUNCTION_H__
#define __GEMMA3_FUNCTION_H__

#include <json.hpp>
#include <string>

using json = nlohmann::json;

namespace causallm {
namespace gemma3 {

/**
 * @brief Apply the chat template for FunctionGemma
 * @param chat_input The input JSON containing "messages" and optionally "tools"
 * @return The formatted prompt string
 */
std::string apply_function_gemma_template(const json &chat_input);

} // namespace gemma3
} // namespace causallm

#endif // __GEMMA3_FUNCTION_H__

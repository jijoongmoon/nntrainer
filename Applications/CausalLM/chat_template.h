// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    chat_template.h
 * @date    10 Apr 2026
 * @brief   Chat template support using tokenizer_config.json
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#ifndef __CHAT_TEMPLATE_H__
#define __CHAT_TEMPLATE_H__

#include <string>
#include <vector>

#include "json.hpp"

namespace causallm {

using json = nlohmann::json;

/**
 * @brief Chat message structure for multi-turn conversations
 */
struct ChatMessage {
  std::string role;    // "system", "user", "assistant"
  std::string content; // message content
};

/**
 * @brief Chat template class that reads and applies HuggingFace chat templates
 *
 * Loads chat_template from tokenizer_config.json and renders it using a
 * minimal Jinja2 subset renderer. Supports common constructs used in
 * HuggingFace chat templates: for loops, if/elif/else, variable access,
 * string operations, loop variables, and filters.
 */
class ChatTemplate {
public:
  /**
   * @brief Default constructor (no template loaded)
   */
  ChatTemplate();

  /**
   * @brief Load chat template from tokenizer_config.json
   * @param tokenizer_config_path Path to tokenizer_config.json
   * @param template_name Template name to select when chat_template is an
   *        array (e.g., "default", "tool_use"). Ignored if chat_template
   *        is a string. Defaults to "default".
   * @return ChatTemplate instance
   */
  static ChatTemplate fromFile(const std::string &tokenizer_config_path,
                               const std::string &template_name = "default");

  /**
   * @brief Apply template to multi-turn messages
   * @param messages Vector of ChatMessage (role + content)
   * @param add_generation_prompt Whether to add generation prompt at end
   * @return Formatted prompt string
   */
  std::string apply(const std::vector<ChatMessage> &messages,
                    bool add_generation_prompt = true) const;

  /**
   * @brief Apply template to a single user input (convenience)
   * @param user_input Raw user input string
   * @param add_generation_prompt Whether to add generation prompt at end
   * @return Formatted prompt string
   */
  std::string apply(const std::string &user_input,
                    bool add_generation_prompt = true) const;

  /**
   * @brief Check if a chat template is loaded and available
   * @return true if template is available
   */
  bool isAvailable() const;

  /**
   * @brief Get BOS token
   */
  std::string getBosToken() const;

  /**
   * @brief Get EOS token
   */
  std::string getEosToken() const;

private:
  std::string template_str_;
  std::string bos_token_;
  std::string eos_token_;
  bool available_ = false;

  /**
   * @brief Render a Jinja2 template with the given context
   * @param tmpl Jinja2 template string
   * @param context JSON object with template variables
   * @return Rendered string
   */
  std::string render(const std::string &tmpl, const json &context) const;
};

} // namespace causallm

#endif // __CHAT_TEMPLATE_H__

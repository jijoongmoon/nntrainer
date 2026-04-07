// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   chat_template.h
 * @date   06 April 2026
 * @brief  High-level chat template wrapper. Loads HuggingFace
 *         tokenizer_config.json and applies the chat_template to messages.
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#ifndef __CHAT_TEMPLATE_H__
#define __CHAT_TEMPLATE_H__

#include "chat_template_engine.h"
#include <json.hpp>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace causallm {

/**
 * @brief Chat template wrapper that loads and applies HuggingFace-compatible
 *        chat templates from tokenizer_config.json.
 *
 * Usage:
 *   ChatTemplate ct;
 *   ct.load_from_tokenizer_config(model_path + "/tokenizer_config.json");
 *   // or
 *   ct.set_template(template_string);
 *
 *   json messages = json::array({
 *     {{"role", "user"}, {"content", "Hello"}}
 *   });
 *   std::string prompt = ct.apply(messages, true);
 */
class ChatTemplate {
public:
  ChatTemplate() = default;

  /**
   * @brief Load chat template from tokenizer_config.json
   * @param path Path to tokenizer_config.json
   * @return true if chat_template was found and loaded
   */
  bool load_from_tokenizer_config(const std::string &path);

  /**
   * @brief Load chat template from a config JSON object directly
   * @param config The parsed tokenizer_config.json
   * @return true if chat_template was found and loaded
   */
  bool load_from_config(const json &config);

  /**
   * @brief Set the template string directly
   * @param tmpl_str Jinja2 template string
   */
  void set_template(const std::string &tmpl_str);

  /**
   * @brief Check if a template has been loaded
   */
  bool has_template() const { return template_ != nullptr; }

  /**
   * @brief Apply the chat template to messages
   * @param messages JSON array of {role, content} message objects
   * @param add_generation_prompt If true, appends the assistant prompt prefix
   * @return Formatted prompt string
   */
  std::string apply(const json &messages, bool add_generation_prompt = true);

  /**
   * @brief Apply the chat template to a chat_input JSON (may contain
   *        messages, tools, etc.)
   * @param chat_input JSON object with "messages" key and optional extras
   * @param add_generation_prompt If true, appends the assistant prompt prefix
   * @return Formatted prompt string
   */
  std::string apply_chat_input(const json &chat_input,
                               bool add_generation_prompt = true);

  /**
   * @brief Get the BOS token
   */
  const std::string &bos_token() const { return bos_token_; }

  /**
   * @brief Get the EOS token
   */
  const std::string &eos_token() const { return eos_token_; }

private:
  std::unique_ptr<jinja::Template> template_;
  std::string bos_token_;
  std::string eos_token_;
  std::string template_str_;

  /**
   * @brief Convert nlohmann::json to jinja::Value
   */
  static jinja::Value json_to_value(const json &j);
};

} // namespace causallm

#endif // __CHAT_TEMPLATE_H__

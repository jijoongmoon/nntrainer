// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    chat_template.h
 * @date    10 Apr 2026
 * @brief   Hugging Face chat template adapter for OpenAI-style chat inputs.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jungwon-Lee <jungone.lee@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#ifndef __CHAT_TEMPLATE_H__
#define __CHAT_TEMPLATE_H__

#include "json.hpp"

#include <memory>
#include <string>

/**
 * @brief Namespace for CausalLM application components
 */
namespace causallm {

/**
 * @brief Applies Hugging Face chat templates to structured chat requests.
 */
class ChatTemplate {
public:
  /**
   * @brief Options controlling chat template rendering behavior.
   */
  struct Options {
    /**
     * @brief Controls whether a generation prompt is appended.
     */
    enum class GenerationPromptMode { Auto, Always, Never };

    /**
     * @brief Controls how developer messages are represented.
     */
    enum class DeveloperRolePolicy { Auto, Preserve, MergeIntoSystem };

    GenerationPromptMode generation_prompt = GenerationPromptMode::Auto;
    DeveloperRolePolicy developer_role_policy = DeveloperRolePolicy::Auto;
    bool continue_final_message = false;
    std::string template_name;
  };

  static bool Exists(const std::string &model_path);
  static ChatTemplate Load(const std::string &model_path);

  ChatTemplate(ChatTemplate &&) noexcept;
  ChatTemplate &operator=(ChatTemplate &&) noexcept;
  ChatTemplate(const ChatTemplate &) = delete;
  ChatTemplate &operator=(const ChatTemplate &) = delete;
  ~ChatTemplate();

  std::string apply(const nlohmann::json &request) const;
  std::string apply(const nlohmann::json &request,
                    const Options &options) const;

  const std::string &sourcePath() const;

private:
  struct Impl;

  explicit ChatTemplate(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

} // namespace causallm

#endif // __CHAT_TEMPLATE_H__

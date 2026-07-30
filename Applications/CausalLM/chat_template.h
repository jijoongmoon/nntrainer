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

/**
 * @brief Whether a caller's prompt should be run through the model's template.
 */
enum class PromptTemplateMode {
  Auto,  /**< apply the model package's template when it has one */
  Never, /**< feed the prompt exactly as given (already-templated input) */
};

/**
 * @brief The model package's default chat render context.
 *
 * Read from the optional "chat_template_context" object in nntr_config.json.
 * Some templates branch on a sibling render key rather than on the messages --
 * a thinking-channel switch is the common case -- and the right owner of that
 * value is the model package, not a table in the code and not every caller.
 *
 * @param nntr_cfg parsed nntr_config.json
 * @return the context object, or an empty object when the key is absent
 */
nlohmann::json chatTemplateContext(const nlohmann::json &nntr_cfg);

/**
 * @brief Wrap one user turn into the chat request a template expects.
 * @param user_text      the caller's prompt (UTF-8)
 * @param render_context extra top-level render keys merged into the request
 *                       (ignored when not an object); explicit request keys
 *                       always win, so this only supplies defaults
 */
nlohmann::json makeUserRequest(
  const std::string &user_text,
  const nlohmann::json &render_context = nlohmann::json::object());

/**
 * @brief Turn a chat request into the exact text handed to run().
 *
 * This is the single prompt-preparation seam. Every front end -- the CLI and
 * the SDK entry points alike -- goes through it, so one prompt plus one model
 * package yields one prompt string regardless of which front end the caller
 * used. There is deliberately no per-architecture template table anywhere: a
 * chat template is a property of the model package, not of the code.
 *
 * @param tmpl  template loaded from the model package, or nullptr when the
 *              package carries none
 * @param request chat request (messages array, or object with "messages")
 * @param mode  Auto = render when @a tmpl is present; Never = not rendered
 * @param render_context default render keys (see makeUserRequest)
 * @return the rendered prompt, or "" when nothing was rendered (mode Never or
 *         no template) -- callers in that case must feed their own raw text
 * @throw std::exception propagated from the renderer: a template that cannot
 *        render is reported, never silently downgraded to a raw prompt (a
 *        silently raw prompt makes an instruction model drift, which reads as
 *        a model bug rather than a configuration one)
 */
std::string
buildPrompt(const ChatTemplate *tmpl, const nlohmann::json &request,
            PromptTemplateMode mode = PromptTemplateMode::Auto,
            const nlohmann::json &render_context = nlohmann::json::object());

/**
 * @brief Seam for a single user turn -- the SDK convenience path.
 *
 * Deliberately a different name rather than an overload of buildPrompt(): a
 * string literal converts to nlohmann::json just as readily as to std::string,
 * so an overload pair would make every `buildPrompt(t, "text")` call site
 * ambiguous.
 *
 * @return the rendered prompt, or @a user_text unchanged when no template was
 *         applied (mode Never, or the package has none)
 * @see buildPrompt
 */
std::string buildUserPrompt(
  const ChatTemplate *tmpl, const std::string &user_text,
  PromptTemplateMode mode = PromptTemplateMode::Auto,
  const nlohmann::json &render_context = nlohmann::json::object());

} // namespace causallm

#endif // __CHAT_TEMPLATE_H__

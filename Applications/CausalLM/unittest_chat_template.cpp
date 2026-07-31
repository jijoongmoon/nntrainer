// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_chat_template.cpp
 * @date    30 Jul 2026
 * @brief   Unit tests for the shared prompt-preparation seam.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 *
 * These pin the contract every front end depends on: one prompt plus one model
 * package yields one prompt string, an opt-out passes bytes through untouched,
 * and a template that cannot render raises instead of quietly handing the model
 * an unmarked prompt. No model weights, tokenizer or backend are involved.
 */

#include "chat_template.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace {

namespace fs = std::filesystem;

/** @brief A throwaway model-package directory holding just a template. */
class TemplateDir {
public:
  explicit TemplateDir(const std::string &jinja, const std::string &name) {
    path_ = fs::temp_directory_path() / ("nntr_chat_template_test_" + name +
                                         "_" + std::to_string(::getpid()));
    fs::create_directories(path_);

    std::ofstream(path_ / "chat_template.jinja") << jinja;
    std::ofstream(path_ / "tokenizer_config.json")
      << R"({"bos_token": "<s>", "eos_token": "</s>"})";
  }

  ~TemplateDir() {
    std::error_code ec;
    fs::remove_all(path_, ec);
  }

  std::string str() const { return path_.string(); }

private:
  fs::path path_;
};

/**
 * @brief A template that marks turns and reacts to a sibling render key.
 *
 * Newline-free on purpose: jinja's whitespace-trimming markers would otherwise
 * make the expected strings below a puzzle about trimming rather than about the
 * seam under test.
 */
constexpr const char *kTurnTemplate =
  "{%- for m in messages -%}"
  "[{{ m['role'] }}:{{ m['content'] }}]"
  "{%- endfor -%}"
  "{%- if add_generation_prompt -%}"
  "[model{% if not (thinking | default(false)) %}:nothink{% endif %}]"
  "{%- endif -%}";

TEST(ChatTemplateSeam, PackageContextIsReadFromConfig) {
  EXPECT_TRUE(causallm::chatTemplateContext(nlohmann::json::object()).empty());

  nlohmann::json cfg = {{"chat_template_context", {{"thinking", false}}}};
  const auto context = causallm::chatTemplateContext(cfg);
  ASSERT_TRUE(context.is_object());
  EXPECT_EQ(context.at("thinking"), false);

  // A non-object value is ignored rather than propagated as junk.
  nlohmann::json bad = {{"chat_template_context", "yes please"}};
  EXPECT_TRUE(causallm::chatTemplateContext(bad).empty());
}

TEST(ChatTemplateSeam, UserRequestCarriesContextButOwnsMessages) {
  const auto request =
    causallm::makeUserRequest("hello", {{"thinking", true}, {"messages", 42}});

  ASSERT_TRUE(request.contains("messages"));
  ASSERT_TRUE(request["messages"].is_array());
  ASSERT_EQ(request["messages"].size(), 1u);
  EXPECT_EQ(request["messages"][0]["role"], "user");
  EXPECT_EQ(request["messages"][0]["content"], "hello");
  // The context supplied a "messages" key; the conversation still wins.
  EXPECT_EQ(request["thinking"], true);
}

TEST(ChatTemplateSeam, NoTemplateFeedsThePromptUnchanged) {
  const std::string prompt = "The capital of France is";
  EXPECT_EQ(causallm::buildUserPrompt(nullptr, prompt), prompt);
  EXPECT_TRUE(causallm::buildPrompt(nullptr, nlohmann::json::array()).empty());
}

TEST(ChatTemplateSeam, OptOutIsByteIdentical) {
  TemplateDir dir(kTurnTemplate, "optout");
  ASSERT_TRUE(causallm::ChatTemplate::Exists(dir.str()));
  auto tmpl = causallm::ChatTemplate::Load(dir.str());

  const std::string pre_templated = "<|turn>user\nalready wrapped<turn|>\n";
  EXPECT_EQ(causallm::buildUserPrompt(&tmpl, pre_templated,
                                      causallm::PromptTemplateMode::Never),
            pre_templated);
}

TEST(ChatTemplateSeam, TemplateIsAppliedAndIsDeterministic) {
  TemplateDir dir(kTurnTemplate, "apply");
  auto tmpl = causallm::ChatTemplate::Load(dir.str());

  const std::string first = causallm::buildUserPrompt(&tmpl, "hi");
  const std::string second = causallm::buildUserPrompt(&tmpl, "hi");
  EXPECT_EQ(first, second);
  EXPECT_EQ(first, "[user:hi][model:nothink]");
}

TEST(ChatTemplateSeam, PackageContextReachesTheTemplate) {
  TemplateDir dir(kTurnTemplate, "context");
  auto tmpl = causallm::ChatTemplate::Load(dir.str());

  // The package asks for the thinking channel: the empty-thought marker goes.
  const std::string with_context = causallm::buildUserPrompt(
    &tmpl, "hi", causallm::PromptTemplateMode::Auto, {{"thinking", true}});
  EXPECT_EQ(with_context, "[user:hi][model]");
}

TEST(ChatTemplateSeam, RequestKeysWinOverPackageContext) {
  TemplateDir dir(kTurnTemplate, "override");
  auto tmpl = causallm::ChatTemplate::Load(dir.str());

  nlohmann::json request = causallm::makeUserRequest("hi");
  request["thinking"] = false; // explicit request value
  const std::string rendered =
    causallm::buildPrompt(&tmpl, request, causallm::PromptTemplateMode::Auto,
                          {{"thinking", true}}); // package default loses
  EXPECT_EQ(rendered, "[user:hi][model:nothink]");
}

TEST(ChatTemplateSeam, AffixesFrameTheRenderedPrompt) {
  TemplateDir dir(kTurnTemplate, "affixes");
  auto tmpl = causallm::ChatTemplate::Load(dir.str());

  const std::string content = "a document long enough to need shortening";
  const auto request = causallm::makeUserRequest(content);
  const std::string rendered = causallm::buildPrompt(&tmpl, request);
  const auto affixes = causallm::promptAffixes(&tmpl, request);

  EXPECT_EQ(affixes.prefix_bytes, std::string("[user:").size());
  EXPECT_EQ(affixes.suffix_bytes, std::string("][model:nothink]").size());

  // The contract the runner relies on: the two spans really are the template's
  // own bytes, so what lies between them is the caller's content and nothing
  // else. Whatever is dropped from there leaves the frame intact.
  ASSERT_LE(affixes.prefix_bytes + affixes.suffix_bytes, rendered.size());
  EXPECT_EQ(rendered.substr(0, affixes.prefix_bytes) + content +
              rendered.substr(rendered.size() - affixes.suffix_bytes),
            rendered);
}

TEST(ChatTemplateSeam, EarlierTurnsBelongToTheProtectedPrefix) {
  TemplateDir dir(kTurnTemplate, "affixes_multi");
  auto tmpl = causallm::ChatTemplate::Load(dir.str());

  nlohmann::json request = nlohmann::json::object();
  request["messages"] =
    nlohmann::json::array({{{"role", "system"}, {"content", "S"}},
                           {{"role", "user"}, {"content", "X"}}});

  const auto affixes = causallm::promptAffixes(&tmpl, request);
  // Only the last message is content; the system turn ahead of it is frame.
  EXPECT_EQ(affixes.prefix_bytes, std::string("[system:S][user:").size());
  EXPECT_EQ(affixes.suffix_bytes, std::string("][model:nothink]").size());
}

TEST(ChatTemplateSeam, NoAffixClaimWithoutATemplateOrAMessage) {
  TemplateDir dir(kTurnTemplate, "affixes_none");
  auto tmpl = causallm::ChatTemplate::Load(dir.str());

  const auto request = causallm::makeUserRequest("hi");
  // {0, 0} is "no claim made", and every one of these is such a case.
  EXPECT_EQ(causallm::promptAffixes(nullptr, request).suffix_bytes, 0u);
  EXPECT_EQ(
    causallm::promptAffixes(&tmpl, request, causallm::PromptTemplateMode::Never)
      .suffix_bytes,
    0u);
  EXPECT_EQ(
    causallm::promptAffixes(&tmpl, nlohmann::json::array()).suffix_bytes, 0u);
  EXPECT_EQ(
    causallm::promptAffixes(&tmpl, nlohmann::json::object()).suffix_bytes, 0u);
}

TEST(ChatTemplateSeam, AContentSensitiveTemplateReportsNothing) {
  // The content is emitted twice, so no fixed pair of affix spans describes
  // this render. Reporting a pair anyway would have a caller keep bytes that
  // are not the frame -- better to say nothing and let it fall back.
  TemplateDir dir("{%- for m in messages -%}"
                  "[{{ m['role'] }}:{{ m['content'] }}:{{ m['content'] }}]"
                  "{%- endfor -%}",
                  "affixes_twice");
  auto tmpl = causallm::ChatTemplate::Load(dir.str());

  const auto affixes =
    causallm::promptAffixes(&tmpl, causallm::makeUserRequest("hi"));
  EXPECT_EQ(affixes.prefix_bytes, 0u);
  EXPECT_EQ(affixes.suffix_bytes, 0u);
}

TEST(ChatTemplateSeam, BrokenTemplateRaisesInsteadOfGoingRaw) {
  TemplateDir dir("{%- for m in messages -%}{{ m['role'] }}", "broken");

  // Whether the parse fails at Load or at the first render, the caller learns
  // about it: what must never happen is a silent fallback to the raw prompt.
  EXPECT_ANY_THROW({
    auto tmpl = causallm::ChatTemplate::Load(dir.str());
    (void)causallm::buildUserPrompt(&tmpl, "hi");
  });
}

} // namespace

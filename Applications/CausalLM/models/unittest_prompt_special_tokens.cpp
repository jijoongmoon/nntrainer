// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    unittest_prompt_special_tokens.cpp
 * @brief   Pins the special-token policy CausalLM::run() applies to a prompt:
 *          the tokenizer's special tokens belong at sequence position 0 and
 *          nowhere else, and no prefix length may be derived from an empty
 *          system prompt.
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include <tokenizers_cpp.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace {

/**
 * @brief A minimal BOS-prepending tokenizer, self-contained so the test needs
 * no model assets. It reproduces the only tokenizer property these tests care
 * about, and the one that makes the bugs below observable: a
 * `TemplateProcessing` post-processor whose `single` template puts a special
 * token in front of the sequence. Gemma2's tokenizer.json is shaped exactly
 * this way, with `<bos>` = id 2.
 */
constexpr const char *kBosTokenizerJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false,
     "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}},
               {"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [{"Sequence": {"id": "A", "type_id": 0}},
             {"Sequence": {"id": "B", "type_id": 0}}],
    "special_tokens": {
      "<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}
    }
  },
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"<unk>": 0, "<bos>": 2, "you": 3, "are": 4, "helpful": 5,
              "the": 6, "capital": 7, "of": 8, "france": 9, "is": 10,
              "and": 11, "germany": 12},
    "unk_token": "<unk>"
  }
})json";

constexpr int kBosTokenId = 2;

/**
 * @brief Mirror of the `prompt_starts_sequence` predicate in CausalLM::run()
 * (Applications/CausalLM/models/causal_lm.cpp).
 *
 * It is deliberately a copy and not a call into the model: the point of these
 * tests is to pin the POLICY, so changing the predicate in run() has to be a
 * conscious edit here as well.
 */
bool promptStartsSequence(bool save_kvcache, unsigned int sys_promp_len,
                          unsigned int global_token_len) {
  return save_kvcache || (sys_promp_len + global_token_len) == 0;
}

/**
 * @brief Mirror of the SYS_PROMP_LEN fallback in CausalLM::run(), which
 * re-derives the length of an already-cached system-prompt prefix when the
 * `sys_prompt_token_size` config key is absent.
 */
unsigned int resolveSysPrompLen(tokenizers::Tokenizer *tokenizer,
                                bool use_kvcache, bool save_kvcache,
                                unsigned int preset,
                                const std::string &system_prompt) {
  unsigned int sys_promp_len = preset;
  if (use_kvcache && !save_kvcache && sys_promp_len == 0 &&
      !system_prompt.empty())
    sys_promp_len = static_cast<unsigned int>(
      tokenizer->Encode(system_prompt, /*add_special_tokens=*/true).size());
  return sys_promp_len;
}

/** @brief Test fixture owning the tokenizer. */
class PromptSpecialTokens : public ::testing::Test {
protected:
  void SetUp() override {
    tokenizer = tokenizers::Tokenizer::FromBlobJSON(kBosTokenizerJson);
    ASSERT_NE(tokenizer, nullptr);
  }

  std::unique_ptr<tokenizers::Tokenizer> tokenizer;
};

/**
 * @brief The fixture really does prepend a BOS, and -- the reason the empty
 * prompt needs a guard -- it prepends one to the EMPTY string too, so
 * `Encode("", true).size()` is 1, not 0.
 */
TEST_F(PromptSpecialTokens, bos_tokenizer_fixture_p) {
  const auto with_specials =
    tokenizer->Encode(std::string("the capital"), true);
  const auto without = tokenizer->Encode(std::string("the capital"), false);

  ASSERT_FALSE(with_specials.empty());
  ASSERT_FALSE(without.empty());
  EXPECT_EQ(with_specials.front(), kBosTokenId);
  EXPECT_EQ(with_specials.size(), without.size() + 1);
  EXPECT_NE(without.front(), kBosTokenId);

  /// the empty string is not empty once the specials are added
  EXPECT_EQ(tokenizer->Encode(std::string(""), true).size(), 1u);
  EXPECT_EQ(tokenizer->Encode(std::string(""), false).size(), 0u);
}

/**
 * @brief No prefix length may be derived from an empty system prompt. Doing so
 * fabricates a one-row prefix that no cache ever described, which then restores
 * a stale KV row and shifts every later position by one.
 */
TEST_F(PromptSpecialTokens, empty_system_prompt_yields_no_prefix_length_p) {
  /// resume shape (use_kvcache && !save_kvcache), no preset, empty prompt
  EXPECT_EQ(resolveSysPrompLen(tokenizer.get(), true, false, 0, ""), 0u);

  /// a real system prompt is still counted, and counted WITH the specials so it
  /// agrees with what the save pass stored
  const std::string system_prompt = "you are helpful";
  const unsigned int save_pass_len = static_cast<unsigned int>(
    tokenizer->Encode(system_prompt, /*add_special_tokens=*/true).size());
  EXPECT_EQ(resolveSysPrompLen(tokenizer.get(), true, false, 0, system_prompt),
            save_pass_len);
  EXPECT_GT(
    save_pass_len,
    static_cast<unsigned int>(
      tokenizer->Encode(system_prompt, /*add_special_tokens=*/false).size()));

  /// an explicit sys_prompt_token_size always wins, empty prompt or not
  EXPECT_EQ(resolveSysPrompLen(tokenizer.get(), true, false, 7, ""), 7u);
  EXPECT_EQ(resolveSysPrompLen(tokenizer.get(), true, false, 7, system_prompt),
            7u);
}

/**
 * @brief The first prompt of a freshly constructed model opens the sequence and
 * must keep its BOS.
 */
TEST_F(PromptSpecialTokens, first_prompt_of_a_fresh_model_keeps_the_bos_p) {
  const unsigned int sys_promp_len = 0;    /// no kvcache configured
  const unsigned int global_token_len = 0; /// set by setupParameters()
  const bool starts =
    promptStartsSequence(false, sys_promp_len, global_token_len);
  EXPECT_TRUE(starts);

  const auto ids =
    tokenizer->Encode(std::string("the capital of france is"), starts);
  ASSERT_FALSE(ids.empty());
  EXPECT_EQ(ids.front(), kBosTokenId);
}

/**
 * @brief Two consecutive run()-equivalent encodings on ONE model object must
 * not both begin with the BOS: the second call continues a sequence whose rows
 * are already accounted for in global_token_len. No kvcache configuration is
 * needed to reach this -- the shipped C API `runModel()` can be called
 * repeatedly on one loaded model, and only `global_token_len` remembers the
 * earlier turn.
 */
TEST_F(PromptSpecialTokens, second_run_does_not_splice_a_second_bos_p) {
  const bool use_kvcache = false;
  const bool save_kvcache = false; /// USE_KVCACHE && ... -> false
  const unsigned int sys_promp_len =
    resolveSysPrompLen(tokenizer.get(), use_kvcache, save_kvcache, 0, "");
  unsigned int global_token_len = 0;

  /// turn 1: opens the sequence
  const bool starts_first =
    promptStartsSequence(save_kvcache, sys_promp_len, global_token_len);
  const auto first =
    tokenizer->Encode(std::string("the capital of france is"), starts_first);
  ASSERT_FALSE(first.empty());
  EXPECT_TRUE(starts_first);
  EXPECT_EQ(first.front(), kBosTokenId);

  /// end of run(): global_token_len += (generation_cnt + init_len)
  const unsigned int generated = 3;
  global_token_len += generated + static_cast<unsigned int>(first.size());
  ASSERT_GT(global_token_len, 0u);

  /// turn 2: continues it
  const bool starts_second =
    promptStartsSequence(save_kvcache, sys_promp_len, global_token_len);
  const auto second =
    tokenizer->Encode(std::string("and of germany"), starts_second);
  ASSERT_FALSE(second.empty());
  EXPECT_FALSE(starts_second);
  EXPECT_NE(second.front(), kBosTokenId);

  /// the property the model actually needs: exactly one BOS in the stream
  EXPECT_FALSE(first.front() == kBosTokenId && second.front() == kBosTokenId);
}

/**
 * @brief The KV-resume continuation must not carry specials, while the save
 * pass that builds the cache must.
 */
TEST_F(PromptSpecialTokens, kvcache_save_opens_and_resume_continues_p) {
  const std::string system_prompt = "you are helpful";
  const std::string continuation = "the capital of france is";

  /// save pass: SAVE_KVCACHE is true, prompt_ == system_prompt
  EXPECT_TRUE(promptStartsSequence(true, 0, 0));
  const auto saved = tokenizer->Encode(system_prompt, true);
  ASSERT_FALSE(saved.empty());
  EXPECT_EQ(saved.front(), kBosTokenId);
  const unsigned int cached_rows = static_cast<unsigned int>(saved.size());

  /// SAVE_KVCACHE stays true even when sys_prompt_token_size is preset, so the
  /// save run keeps its specials
  EXPECT_TRUE(promptStartsSequence(true, cached_rows, 0));

  /// resume pass: the cache supplies the first cached_rows rows
  const unsigned int resumed =
    resolveSysPrompLen(tokenizer.get(), true, false, 0, system_prompt);
  EXPECT_EQ(resumed, cached_rows);
  const bool starts = promptStartsSequence(false, resumed, 0);
  EXPECT_FALSE(starts);
  const auto tail = tokenizer->Encode(continuation, starts);
  ASSERT_FALSE(tail.empty());
  EXPECT_NE(tail.front(), kBosTokenId);
}

/**
 * @brief The predicate over the whole (SAVE_KVCACHE, SYS_PROMP_LEN,
 * global_token_len) space: specials exactly when this encode produces the
 * sequence's first tokens.
 */
TEST_F(PromptSpecialTokens, predicate_truth_table_p) {
  /// nothing written yet -> opens the sequence
  EXPECT_TRUE(promptStartsSequence(false, 0, 0));
  /// the save run always opens it, whatever the counters say
  EXPECT_TRUE(promptStartsSequence(true, 0, 0));
  EXPECT_TRUE(promptStartsSequence(true, 9, 0));
  EXPECT_TRUE(promptStartsSequence(true, 0, 9));
  /// a cached prefix, or an earlier turn, or both -> continuation
  EXPECT_FALSE(promptStartsSequence(false, 9, 0));
  EXPECT_FALSE(promptStartsSequence(false, 0, 9));
  EXPECT_FALSE(promptStartsSequence(false, 9, 9));
}

} // namespace

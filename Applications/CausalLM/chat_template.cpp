// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    chat_template.cpp
 * @date    10 Apr 2026
 * @brief   Hugging Face chat template adapter for OpenAI-style chat inputs.
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Jungwon-Lee <jungone.lee@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "chat_template.h"

#include <minja/chat-template.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

/**
 * @brief Namespace for CausalLM application components
 */
namespace causallm {

/**
 * @brief Anonymous namespace for chat template helpers
 */
namespace {

using OrderedJson = nlohmann::ordered_json;

bool fileExists(const std::string &path) {
  std::ifstream file(path);
  return file.good();
}

/**
 * @brief Read a whole file, or nullopt when it cannot be opened.
 *
 * [init-latency] Load() used to probe each file with fileExists() and then
 * open it a second time to read it -- three files, six opens. Only an OPEN
 * failure yields nullopt; a malformed file still throws out of the parse,
 * exactly as the fileExists()-guarded form did, so no error path changes.
 */
std::optional<std::string> tryReadTextFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
    return std::nullopt;

  std::ostringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

/** @copydoc tryReadTextFile */
std::optional<nlohmann::json> tryReadJsonFile(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open())
    return std::nullopt;

  nlohmann::json data;
  file >> data;
  return data;
}

OrderedJson toOrderedJson(const nlohmann::json &value) {
  if (value.is_object()) {
    OrderedJson out = OrderedJson::object();
    for (auto it = value.begin(); it != value.end(); ++it)
      out[it.key()] = toOrderedJson(it.value());
    return out;
  }

  if (value.is_array()) {
    OrderedJson out = OrderedJson::array();
    for (const auto &item : value)
      out.push_back(toOrderedJson(item));
    return out;
  }

  return OrderedJson::parse(value.dump());
}

std::string tokenContent(const nlohmann::json &value) {
  if (value.is_null())
    return "";

  if (value.is_string())
    return value.get<std::string>();

  if (value.is_object() && value.contains("content") &&
      value["content"].is_string())
    return value["content"].get<std::string>();

  return "";
}

std::string findToken(const nlohmann::json &tokenizer_config,
                      const nlohmann::json &special_tokens,
                      const std::string &name) {
  if (special_tokens.contains(name)) {
    std::string token = tokenContent(special_tokens[name]);
    if (!token.empty())
      return token;
  }

  if (tokenizer_config.contains(name))
    return tokenContent(tokenizer_config[name]);

  return "";
}

std::string textFromContentParts(const nlohmann::json &content) {
  std::string text;
  for (const auto &part : content) {
    if (!part.is_object() || !part.contains("type") ||
        !part["type"].is_string()) {
      throw std::runtime_error("Chat content parts must include a string type");
    }

    const std::string type = part["type"].get<std::string>();
    if (type == "text") {
      if (part.contains("text") && part["text"].is_string())
        text += part["text"].get<std::string>();
    } else {
      throw std::runtime_error("Unsupported non-text chat content part: " +
                               type);
    }
  }

  return text;
}

bool shouldAddGenerationPrompt(const OrderedJson &messages,
                               const nlohmann::json &request,
                               const ChatTemplate::Options &options) {
  using GenerationPromptMode = ChatTemplate::Options::GenerationPromptMode;

  if (options.generation_prompt == GenerationPromptMode::Always)
    return true;

  if (options.generation_prompt == GenerationPromptMode::Never)
    return false;

  if (request.is_object() && request.contains("add_generation_prompt") &&
      request["add_generation_prompt"].is_boolean())
    return request["add_generation_prompt"].get<bool>();

  if (messages.empty())
    throw std::runtime_error("chat_input.messages must not be empty");

  const auto &last = messages.back();
  if (!last.is_object() || !last.contains("role") || !last["role"].is_string())
    throw std::runtime_error("Each chat message must include a string role");

  return last["role"].get<std::string>() != "assistant";
}

} // namespace

/**
 * @brief Stores parsed template data and cached minja renderers.
 */
struct ChatTemplate::Impl {
  std::string model_path;
  std::string source_path;
  std::string template_source;
  nlohmann::json template_map = nlohmann::json::object();
  nlohmann::json tokenizer_config = nlohmann::json::object();
  nlohmann::json special_tokens = nlohmann::json::object();
  std::string bos_token;
  std::string eos_token;
  bool apply_polyfills = true;
  Options::DeveloperRolePolicy default_developer_role_policy =
    Options::DeveloperRolePolicy::MergeIntoSystem;
  mutable std::unordered_map<std::string, std::unique_ptr<minja::chat_template>>
    renderers;
  /** [init-latency L3] renderers is now touched by the warm-up thread too. */
  mutable std::mutex renderers_mtx;
  mutable std::thread warm_thread;
  /**
   * @brief Serialises the warm-up join, and ONLY the join.
   *
   * It cannot be `renderers_mtx`: the warm thread takes that lock to publish
   * its renderer, so joining while holding it deadlocks. That is why the join
   * was originally left unsynchronised -- but then two concurrent apply()
   * callers could both observe warm_thread.joinable() and both call join() on
   * the same thread, which is undefined. It has never fired only because the
   * one embedding caller in production serialises apply() behind its own run
   * mutex, which is a property of that caller and not of this class.
   *
   * A dedicated mutex is used rather than a std::once_flag because once_flag
   * would be consumed by a join attempt made BEFORE startWarmup() assigned the
   * thread; the flag would then be spent and the thread never joined, which is
   * std::terminate at destruction. The mutex has no such ordering trap: an
   * early call simply finds nothing to join and a later one still joins.
   * Losers block here until the winner's join returns, which is exactly the
   * "nobody reads renderers until the warm thread is done" contract.
   */
  mutable std::mutex warm_join_mtx;

  /**
   * @brief Join the warm-up thread before anyone reads `renderers`.
   *        Idempotent and safe to call concurrently.
   */
  void joinWarmup() const {
    std::lock_guard<std::mutex> lk(warm_join_mtx);
    if (warm_thread.joinable())
      warm_thread.join();
  }

  ~Impl() {
    // A destructor must not let an exception escape, and a std::thread that is
    // still joinable when destroyed is an unconditional std::terminate --
    // detaching is the only safe fallback if the join itself failed.
    try {
      joinWarmup();
    } catch (...) {
      if (warm_thread.joinable())
        warm_thread.detach();
    }
  }

  /**
   * @brief [init-latency L3] Build the renderer we are almost certainly going
   *   to need on a side thread, at template-LOAD time instead of at first
   *   apply().
   *
   * minja::chat_template's constructor parses the Jinja source and (for the
   * polyfill probe) renders it once with a synthetic message set; on the
   * an 18 KB chat_template.jinja that measured ~170 ms, paid serially inside
   * the first apply() -- i.e. straight onto the first-token latency, after the
   * whole model was already loaded. It depends on nothing but the source and
   * the bos/eos tokens, all of which are final by the end of Load(), so it
   * overlaps the graph compile + weight load for free (same trick as the
   * async tokenizer parse in Transformer's ctor).
   *
   * Only the key apply() will actually ask for is warmed: "__default__" for a
   * single-source template (chat_template.jinja or a string chat_template),
   * else the "default" entry of a template map. A tools/template_name request
   * simply misses the cache and constructs lazily as before.
   *
   * Failures are swallowed: apply() then constructs on its own and raises the
   * real error at the point the caller can attribute it.
   * NNTR_CHAT_TEMPLATE_WARM=0 disables the warm-up.
   */
  void startWarmup() {
    const char *e = std::getenv("NNTR_CHAT_TEMPLATE_WARM");
    if (e != nullptr && e[0] == '0')
      return;

    std::string key;
    std::string src;
    if (!template_source.empty()) {
      key = "__default__";
      src = template_source;
    } else if (template_map.contains("default") &&
               template_map["default"].is_string()) {
      key = "default";
      src = template_map["default"].get<std::string>();
    } else {
      return;
    }

    warm_thread = std::thread([this, key, src]() {
      try {
        auto renderer =
          std::make_unique<minja::chat_template>(src, bos_token, eos_token);
        std::lock_guard<std::mutex> lk(renderers_mtx);
        renderers.emplace(key, std::move(renderer));
      } catch (...) {
        // Leave the cache empty; apply() will construct and throw properly.
      }
      if (std::getenv("NNTR_INIT_TRACE")) {
        std::fprintf(stderr, "[init-trace] chat-template renderer warm done\n");
        std::fflush(stderr);
      }
    });
  }

  const std::string &selectTemplate(bool has_tools, const Options &options,
                                    std::string &cache_key) const {
    if (!template_source.empty()) {
      cache_key = "__default__";
      return template_source;
    }

    if (!options.template_name.empty()) {
      if (!template_map.contains(options.template_name) ||
          !template_map[options.template_name].is_string()) {
        throw std::runtime_error("Requested chat template is not available: " +
                                 options.template_name);
      }
      cache_key = options.template_name;
      return template_map[options.template_name].get_ref<const std::string &>();
    }

    if (has_tools && template_map.contains("tool_use") &&
        template_map["tool_use"].is_string()) {
      cache_key = "tool_use";
      return template_map["tool_use"].get_ref<const std::string &>();
    }

    if (template_map.contains("default") &&
        template_map["default"].is_string()) {
      cache_key = "default";
      return template_map["default"].get_ref<const std::string &>();
    }

    for (auto it = template_map.begin(); it != template_map.end(); ++it) {
      if (it.value().is_string()) {
        cache_key = it.key();
        return it.value().get_ref<const std::string &>();
      }
    }

    throw std::runtime_error("No usable chat template was found");
  }

  const minja::chat_template &rendererFor(const std::string &key,
                                          const std::string &source) const {
    // [init-latency L3] The warm-up thread may still be inserting; join it
    // before the lookup, then take renderers_mtx for the map itself (a later
    // apply() with a different key can race a concurrent caller on it).
    // Lock ORDER matters and is enforced by construction: joinWarmup() takes
    // and releases warm_join_mtx and returns before renderers_mtx is acquired
    // here, so the two are never held at once -- which is what keeps the join
    // from deadlocking against the warm thread's own renderers_mtx.
    joinWarmup();
    std::lock_guard<std::mutex> lk(renderers_mtx);
    auto it = renderers.find(key);
    if (it != renderers.end())
      return *it->second;

    auto renderer =
      std::make_unique<minja::chat_template>(source, bos_token, eos_token);
    auto inserted = renderers.emplace(key, std::move(renderer));
    return *inserted.first->second;
  }

  OrderedJson normalizeMessages(const nlohmann::json &request,
                                const Options &options) const {
    const nlohmann::json *messages_json = nullptr;
    if (request.is_array()) {
      messages_json = &request;
    } else if (request.is_object() && request.contains("messages")) {
      messages_json = &request["messages"];
    }

    if (messages_json == nullptr || !messages_json->is_array())
      throw std::runtime_error("chat_input must contain a messages array");

    OrderedJson messages = OrderedJson::array();
    std::unordered_map<std::string, std::string> tool_call_names;
    Options::DeveloperRolePolicy developer_role_policy =
      options.developer_role_policy == Options::DeveloperRolePolicy::Auto
        ? default_developer_role_policy
        : options.developer_role_policy;

    for (const auto &message_json : *messages_json) {
      if (!message_json.is_object() || !message_json.contains("role") ||
          !message_json["role"].is_string()) {
        throw std::runtime_error(
          "Each chat message must include a string role");
      }

      OrderedJson message = toOrderedJson(message_json);
      std::string role = message_json["role"].get<std::string>();
      if (role == "developer" &&
          developer_role_policy ==
            Options::DeveloperRolePolicy::MergeIntoSystem) {
        message["role"] = "system";
      } else if (role != "system" && role != "user" && role != "assistant" &&
                 role != "tool" && role != "developer") {
        throw std::runtime_error("Unsupported chat message role: " + role);
      }

      if (message_json.contains("content") &&
          message_json["content"].is_array())
        message["content"] = textFromContentParts(message_json["content"]);

      if (message_json.contains("function_call") &&
          !message_json.contains("tool_calls")) {
        message["tool_calls"] = OrderedJson::array(
          {{{"id", "function_call"},
            {"type", "function"},
            {"function", toOrderedJson(message_json["function_call"])}}});
      }

      if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
        for (const auto &tool_call : message["tool_calls"]) {
          if (tool_call.contains("id") && tool_call["id"].is_string() &&
              tool_call.contains("function") &&
              tool_call["function"].contains("name") &&
              tool_call["function"]["name"].is_string()) {
            tool_call_names[tool_call["id"].get<std::string>()] =
              tool_call["function"]["name"].get<std::string>();
          }
        }
      }

      if (message["role"] == "tool" && !message.contains("name") &&
          message.contains("tool_call_id") &&
          message["tool_call_id"].is_string()) {
        const auto it =
          tool_call_names.find(message["tool_call_id"].get<std::string>());
        if (it != tool_call_names.end())
          message["name"] = it->second;
      }

      messages.push_back(std::move(message));
    }

    return messages;
  }

  OrderedJson normalizeTools(const nlohmann::json &request) const {
    if (!request.is_object())
      return OrderedJson::array();

    const nlohmann::json *tools_json = nullptr;
    const char *tools_field = nullptr;
    if (request.contains("tools")) {
      tools_json = &request["tools"];
      tools_field = "tools";
    } else if (request.contains("functions")) {
      tools_json = &request["functions"];
      tools_field = "functions";
    }

    if (tools_json == nullptr)
      return OrderedJson::array();

    if (!tools_json->is_array())
      throw std::runtime_error(std::string("chat_input.") + tools_field +
                               " must be an array");

    OrderedJson tools = OrderedJson::array();
    for (const auto &tool_json : *tools_json) {
      if (!tool_json.is_object())
        throw std::runtime_error("Each tool must be an object");

      OrderedJson tool = OrderedJson::object();
      if (tool_json.contains("function")) {
        if (!tool_json["function"].is_object()) {
          throw std::runtime_error(
            "OpenAI function tools must include an object function field");
        }

        tool = toOrderedJson(tool_json);
        if (!tool.contains("type"))
          tool["type"] = "function";
      } else if (tool_json.contains("name")) {
        tool["type"] = "function";
        tool["function"] = toOrderedJson(tool_json);
      } else {
        throw std::runtime_error(
          "Function tools must be OpenAI tool objects or raw function schemas");
      }

      const auto &function = tool["function"];
      if (!function.contains("name") || !function["name"].is_string() ||
          function["name"].get<std::string>().empty()) {
        throw std::runtime_error(
          "Function tools must include a non-empty function name");
      }

      tools.push_back(std::move(tool));
    }

    return tools;
  }

  OrderedJson buildExtraContext(const nlohmann::json &request) const {
    OrderedJson extra_context = OrderedJson::object();

    for (const auto &name : {"pad_token", "unk_token"}) {
      std::string token = findToken(tokenizer_config, special_tokens, name);
      if (!token.empty())
        extra_context[name] = token;
    }

    if (special_tokens.contains("additional_special_tokens")) {
      extra_context["additional_special_tokens"] =
        toOrderedJson(special_tokens["additional_special_tokens"]);
    } else if (tokenizer_config.contains("additional_special_tokens")) {
      extra_context["additional_special_tokens"] =
        toOrderedJson(tokenizer_config["additional_special_tokens"]);
    }

    if (request.is_object()) {
      for (auto it = request.begin(); it != request.end(); ++it) {
        const std::string &key = it.key();
        if (key == "messages" || key == "tools" || key == "functions" ||
            key == "add_generation_prompt" || key == "continue_final_message") {
          continue;
        }
        extra_context[key] = toOrderedJson(it.value());
      }
    }

    return extra_context;
  }
};

ChatTemplate::ChatTemplate(std::unique_ptr<Impl> impl) :
  impl_(std::move(impl)) {}

ChatTemplate::ChatTemplate(ChatTemplate &&) noexcept = default;

ChatTemplate &ChatTemplate::operator=(ChatTemplate &&) noexcept = default;

ChatTemplate::~ChatTemplate() = default;

bool ChatTemplate::Exists(const std::string &model_path) {
  if (fileExists(model_path + "/chat_template.jinja"))
    return true;

  try {
    const auto tokenizer_config =
      tryReadJsonFile(model_path + "/tokenizer_config.json");
    return tokenizer_config.has_value() &&
           tokenizer_config->contains("chat_template") &&
           !(*tokenizer_config)["chat_template"].is_null();
  } catch (...) {
    return false;
  }
}

ChatTemplate ChatTemplate::Load(const std::string &model_path) {
  auto impl = std::unique_ptr<Impl>(new Impl());
  impl->model_path = model_path;

  const std::string tokenizer_config_path =
    model_path + "/tokenizer_config.json";
  if (auto tokenizer_config = tryReadJsonFile(tokenizer_config_path))
    impl->tokenizer_config = std::move(*tokenizer_config);

  const std::string special_tokens_path =
    model_path + "/special_tokens_map.json";
  if (auto special_tokens = tryReadJsonFile(special_tokens_path))
    impl->special_tokens = std::move(*special_tokens);

  impl->bos_token =
    findToken(impl->tokenizer_config, impl->special_tokens, "bos_token");
  impl->eos_token =
    findToken(impl->tokenizer_config, impl->special_tokens, "eos_token");

  const std::string template_file_path = model_path + "/chat_template.jinja";
  if (auto template_source = tryReadTextFile(template_file_path)) {
    impl->source_path = template_file_path;
    impl->template_source = std::move(*template_source);
    // [init-latency L3] Everything the renderer needs is final now.
    impl->startWarmup();
    return ChatTemplate(std::move(impl));
  }

  if (!impl->tokenizer_config.contains("chat_template") ||
      impl->tokenizer_config["chat_template"].is_null()) {
    throw std::runtime_error("No chat template found under: " + model_path);
  }

  impl->source_path = tokenizer_config_path + ":chat_template";
  const auto &chat_template = impl->tokenizer_config["chat_template"];
  if (chat_template.is_string()) {
    impl->template_source = chat_template.get<std::string>();
  } else if (chat_template.is_object()) {
    impl->template_map = chat_template;
  } else {
    throw std::runtime_error("tokenizer_config.chat_template must be string or "
                             "object");
  }

  // [init-latency L3] Everything the renderer needs is final now.
  impl->startWarmup();
  return ChatTemplate(std::move(impl));
}

std::string ChatTemplate::apply(const nlohmann::json &request) const {
  return apply(request, Options());
}

std::string ChatTemplate::apply(const nlohmann::json &request,
                                const Options &options) const {
  if (!impl_)
    throw std::runtime_error("ChatTemplate is not initialized");

  if (options.continue_final_message ||
      (request.is_object() && request.value("continue_final_message", false))) {
    throw std::runtime_error(
      "continue_final_message is not supported by this runner yet");
  }

  OrderedJson messages = impl_->normalizeMessages(request, options);
  OrderedJson tools = impl_->normalizeTools(request);

  minja::chat_template_inputs inputs;
  inputs.messages = messages;
  inputs.tools = tools;
  inputs.add_generation_prompt =
    shouldAddGenerationPrompt(messages, request, options);
  inputs.extra_context = impl_->buildExtraContext(request);

  std::string cache_key;
  const std::string &source =
    impl_->selectTemplate(!tools.empty(), options, cache_key);
  minja::chat_template_options render_options;
  render_options.apply_polyfills = impl_->apply_polyfills;
  return impl_->rendererFor(cache_key, source).apply(inputs, render_options);
}

const std::string &ChatTemplate::sourcePath() const {
  if (!impl_)
    throw std::runtime_error("ChatTemplate is not initialized");

  return impl_->source_path;
}

} // namespace causallm

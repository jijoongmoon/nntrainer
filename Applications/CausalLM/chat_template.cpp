// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   chat_template.cpp
 * @date   06 April 2026
 * @brief  Chat template wrapper implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include "chat_template.h"
#include <fstream>
#include <iostream>

namespace causallm {

bool ChatTemplate::load_from_tokenizer_config(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "[ChatTemplate] Cannot open: " << path << std::endl;
    return false;
  }

  json config;
  try {
    f >> config;
  } catch (const std::exception &e) {
    std::cerr << "[ChatTemplate] Failed to parse JSON: " << e.what()
              << std::endl;
    return false;
  }

  return load_from_config(config);
}

bool ChatTemplate::load_from_config(const json &config) {
  // Extract bos_token
  if (config.contains("bos_token")) {
    if (config["bos_token"].is_string()) {
      bos_token_ = config["bos_token"].get<std::string>();
    } else if (config["bos_token"].is_object() &&
               config["bos_token"].contains("content")) {
      bos_token_ = config["bos_token"]["content"].get<std::string>();
    }
  }

  // Extract eos_token
  if (config.contains("eos_token")) {
    if (config["eos_token"].is_string()) {
      eos_token_ = config["eos_token"].get<std::string>();
    } else if (config["eos_token"].is_object() &&
               config["eos_token"].contains("content")) {
      eos_token_ = config["eos_token"]["content"].get<std::string>();
    }
  }

  // Extract chat_template
  if (!config.contains("chat_template")) {
    std::cerr << "[ChatTemplate] No 'chat_template' found in config"
              << std::endl;
    return false;
  }

  std::string tmpl_str;
  if (config["chat_template"].is_string()) {
    tmpl_str = config["chat_template"].get<std::string>();
  } else if (config["chat_template"].is_array()) {
    // Some models have named templates: [{name: "default", template: "..."}]
    for (const auto &entry : config["chat_template"]) {
      if (entry.contains("name") && entry["name"] == "default") {
        tmpl_str = entry["template"].get<std::string>();
        break;
      }
    }
    // If no "default" found, use the first one
    if (tmpl_str.empty() && !config["chat_template"].empty()) {
      const auto &first = config["chat_template"][0];
      if (first.contains("template"))
        tmpl_str = first["template"].get<std::string>();
    }
  }

  if (tmpl_str.empty()) {
    std::cerr << "[ChatTemplate] Empty chat_template" << std::endl;
    return false;
  }

  set_template(tmpl_str);
  return true;
}

void ChatTemplate::set_template(const std::string &tmpl_str) {
  template_str_ = tmpl_str;
  template_ = std::make_unique<jinja::Template>(tmpl_str);
}

std::string ChatTemplate::apply(const json &messages,
                                bool add_generation_prompt) {
  if (!template_)
    throw std::runtime_error(
      "[ChatTemplate] No template loaded. Call load_from_tokenizer_config() "
      "or set_template() first.");

  std::map<std::string, jinja::Value> vars;
  vars["messages"] = json_to_value(messages);
  vars["add_generation_prompt"] = jinja::Value(add_generation_prompt);
  vars["bos_token"] = jinja::Value(bos_token_);
  vars["eos_token"] = jinja::Value(eos_token_);

  return template_->render(vars);
}

std::string ChatTemplate::apply_chat_input(const json &chat_input,
                                           bool add_generation_prompt) {
  if (!template_)
    throw std::runtime_error("[ChatTemplate] No template loaded.");

  std::map<std::string, jinja::Value> vars;

  // messages (required)
  if (chat_input.contains("messages"))
    vars["messages"] = json_to_value(chat_input["messages"]);
  else
    throw std::runtime_error(
      "[ChatTemplate] chat_input must contain 'messages'");

  vars["add_generation_prompt"] = jinja::Value(add_generation_prompt);
  vars["bos_token"] = jinja::Value(bos_token_);
  vars["eos_token"] = jinja::Value(eos_token_);

  // tools (optional)
  if (chat_input.contains("tools"))
    vars["tools"] = json_to_value(chat_input["tools"]);

  // Any other top-level keys
  for (auto it = chat_input.begin(); it != chat_input.end(); ++it) {
    if (it.key() != "messages" && it.key() != "tools") {
      vars[it.key()] = json_to_value(it.value());
    }
  }

  return template_->render(vars);
}

jinja::Value ChatTemplate::json_to_value(const json &j) {
  if (j.is_null())
    return jinja::Value(nullptr);
  if (j.is_boolean())
    return jinja::Value(j.get<bool>());
  if (j.is_number_integer())
    return jinja::Value(j.get<int64_t>());
  if (j.is_number_float())
    return jinja::Value(j.get<double>());
  if (j.is_string())
    return jinja::Value(j.get<std::string>());
  if (j.is_array()) {
    jinja::Value::Array arr;
    for (const auto &elem : j)
      arr.push_back(json_to_value(elem));
    return jinja::Value(arr);
  }
  if (j.is_object()) {
    jinja::Value::Object obj;
    for (auto it = j.begin(); it != j.end(); ++it)
      obj[it.key()] = json_to_value(it.value());
    return jinja::Value(obj);
  }
  return jinja::Value();
}

} // namespace causallm

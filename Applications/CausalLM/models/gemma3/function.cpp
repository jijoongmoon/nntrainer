// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   function.cpp
 * @date   19 January 2026
 * @brief  This defines a chat format for FunctionGemma
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "function.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

namespace causallm {
namespace gemma3 {

// Helper to escape string values
std::string escape_value(const std::string &value) {
  return "<escape>" + value + "<escape>";
}

// Helper to uppercase string
std::string to_upper(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::toupper);
  return s;
}

// Recursively format parameters
std::string format_parameters(const json &properties) {
  std::stringstream ss;
  bool first = true;
  for (auto &[key, val] : properties.items()) {
    if (!first)
      ss << ",";
    ss << key << ":{";

    // inner properties
    bool inner_first = true;
    if (val.contains("description")) {
      if (!inner_first)
        ss << ",";
      ss << "description:"
         << escape_value(val["description"].get<std::string>());
      inner_first = false;
    }

    if (val.contains("type")) {
      if (!inner_first)
        ss << ",";
      ss << "type:" << escape_value(to_upper(val["type"].get<std::string>()));
      inner_first = false;
    }

    // Recursion for nested objects
    if (val.contains("properties")) {
      if (!inner_first)
        ss << ",";
      ss << "properties:{" << format_parameters(val["properties"]) << "}";
      inner_first = false;
    }

    if (val.contains("required")) {
      if (!inner_first)
        ss << ",";
      ss << "required:[";
      bool req_first = true;
      for (const auto &item : val["required"]) {
        if (!req_first)
          ss << ",";
        ss << escape_value(item.get<std::string>());
        req_first = false;
      }
      ss << "]";
      inner_first = false;
    }

    ss << "}";
    first = false;
  }
  return ss.str();
}

std::string format_function_declaration(const json &tool) {
  std::stringstream ss;
  if (tool.contains("function")) {
    const auto &func = tool["function"];
    ss << "declaration:" << func.value("name", "") << ",";
    ss << "description:" << escape_value(func.value("description", "")) << ",";

    ss << "parameters:{";
    if (func.contains("parameters")) {
      const auto &params = func["parameters"];

      if (params.contains("properties")) {
        ss << "properties:{";
        ss << format_parameters(params["properties"]);
        ss << "},";
      }
      if (params.contains("required")) {
        ss << "required:[";
        bool first_req = true;
        for (const auto &req : params["required"]) {
          if (!first_req)
            ss << ",";
          ss << escape_value(req.get<std::string>());
          first_req = false;
        }
        ss << "],";
      }
      if (params.contains("type")) {
        ss << "type:" << escape_value(to_upper(params.value("type", "object")));
      }
    }
    ss << "}";
  }
  return ss.str();
}

// Helper to format a single argument value (for tool calls/responses)
std::string format_argument_value(const json &value) {
  if (value.is_string()) {
    return value.get<std::string>();
  } else {
    return value.dump();
  }
}

std::string apply_function_gemma_template(const json &chat_input) {
  std::stringstream prompt;

  prompt << "<bos>";
  const auto &messages = chat_input["messages"];
  bool tools_inserted = false;

  for (size_t i = 0; i < messages.size(); ++i) {
    const auto &message = messages[i];
    std::string role = message.value("role", "");
    if (role == "assistant")
      role = "model";

    // Open turn
    if (role != "tool") {
      prompt << "<start_of_turn>" << role << "\n";
    }

    // Content
    if (message.contains("content")) {
      if (message["content"].is_string()) {
        prompt << message["content"].get<std::string>();
      }
    }

    // Insert tools if this is the first message and it is developer/system
    if (!tools_inserted && chat_input.contains("tools") &&
        (role == "developer" || role == "system")) {
      for (const auto &tool : chat_input["tools"]) {
        prompt << "<start_function_declaration>";
        prompt << format_function_declaration(tool);
        prompt << "<end_function_declaration>";
      }
      tools_inserted = true;
    }

    // Tool calls
    if (message.contains("tool_calls")) {
      for (const auto &tool_call : message["tool_calls"]) {
        const auto &func = tool_call["function"];
        prompt << "<start_function_call>call:"
               << func["name"].get<std::string>() << "{";
        // Simplistic argument formatting
        if (func.contains("arguments")) {
          if (func["arguments"].is_object()) {
            bool first = true;
            for (auto &[key, val] : func["arguments"].items()) {
              if (!first)
                prompt << ",";
              prompt << key << ":" << format_argument_value(val);
              first = false;
            }
          } else if (func["arguments"].is_string()) {
            prompt << func["arguments"].get<std::string>();
          }
        }
        prompt << "}<end_function_call>";
      }
    }

    // End turn
    if (role != "tool") {
      prompt << "<end_of_turn>\n";
    } else {
      if (message.contains("content")) {
        std::string name = message.value("name", "");
        std::string content_str;
        if (message["content"].is_string())
          content_str = message["content"].get<std::string>();
        else
          content_str = message["content"].dump();

        prompt << "<start_function_response>response:" << name << "{"
               << "value:" << content_str << "}<end_function_response>";
      }
    }
  }

  // Add generation prompt
  prompt << "<start_of_turn>model\n";

  return prompt.str();
}

} // namespace gemma3
} // namespace causallm

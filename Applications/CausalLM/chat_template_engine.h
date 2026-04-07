// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   chat_template_engine.h
 * @date   06 April 2026
 * @brief  Lightweight Jinja2-compatible template engine for HuggingFace chat
 *         templates. This header provides the top-level Template class.
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#ifndef __CHAT_TEMPLATE_ENGINE_H__
#define __CHAT_TEMPLATE_ENGINE_H__

#include "jinja/evaluator.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/value.h"

#include <map>
#include <string>
#include <vector>

namespace causallm {
namespace jinja {

/**
 * @brief Top-level template class. Parses a Jinja2 template string once
 *        and can render it multiple times with different variable bindings.
 *
 * Usage:
 *   Template tmpl(chat_template_string);
 *   std::map<std::string, Value> vars;
 *   vars["messages"] = ...;
 *   vars["add_generation_prompt"] = Value(true);
 *   std::string result = tmpl.render(vars);
 */
class Template {
public:
  explicit Template(const std::string &source);

  std::string render(const std::map<std::string, Value> &variables) const;

private:
  std::string source_;
  std::vector<ASTNodePtr> ast_;
};

} // namespace jinja
} // namespace causallm

#endif // __CHAT_TEMPLATE_ENGINE_H__

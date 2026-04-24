// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   chat_template_engine.cpp
 * @date   06 April 2026
 * @brief  Template class implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include "chat_template_engine.h"

namespace causallm {
namespace jinja {

Template::Template(const std::string &source) : source_(source) {
  Lexer lexer(source);
  auto tokens = lexer.tokenize();
  Parser parser(tokens);
  ast_ = parser.parse();
}

std::string
Template::render(const std::map<std::string, Value> &variables) const {
  Context ctx(variables);
  Evaluator evaluator(ctx);
  return evaluator.render(ast_);
}

} // namespace jinja
} // namespace causallm

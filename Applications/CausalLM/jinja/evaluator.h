// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   evaluator.h
 * @date   06 April 2026
 * @brief  Jinja2 template evaluator — context and rendering
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#ifndef __JINJA_EVALUATOR_H__
#define __JINJA_EVALUATOR_H__

#include "parser.h"
#include "value.h"
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace causallm {
namespace jinja {

class Context {
public:
  Context();
  explicit Context(const std::map<std::string, Value> &vars);

  void set(const std::string &name, const Value &val);
  Value get(const std::string &name) const;
  bool has(const std::string &name) const;

  void push_scope();
  void pop_scope();

private:
  std::vector<std::map<std::string, Value>> scopes_;
};

class Evaluator {
public:
  explicit Evaluator(Context &ctx);

  std::string render(const std::vector<ASTNodePtr> &nodes);
  Value eval_expr(const ASTNodePtr &node);

private:
  Context &ctx_;

  // Macro storage
  std::map<std::string, ASTNodePtr> macros_;

  void render_node(const ASTNodePtr &node, std::ostringstream &out);
  void render_if(const ASTNodePtr &node, std::ostringstream &out);
  void render_for(const ASTNodePtr &node, std::ostringstream &out);
  void render_set(const ASTNodePtr &node);
  void render_macro(const ASTNodePtr &node);

  Value eval_binary(const ASTNodePtr &node);
  Value eval_unary(const ASTNodePtr &node);
  Value eval_getattr(const ASTNodePtr &node);
  Value eval_getitem(const ASTNodePtr &node);
  Value eval_filter(const ASTNodePtr &node);
  Value eval_call(const ASTNodePtr &node);
  Value eval_method_call(const ASTNodePtr &node);

  Value apply_filter(const std::string &name, const Value &val,
                     const std::vector<Value> &args,
                     const std::vector<std::pair<std::string, Value>> &kwargs);
};

} // namespace jinja
} // namespace causallm

#endif // __JINJA_EVALUATOR_H__

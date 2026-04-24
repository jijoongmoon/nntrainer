// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   parser.h
 * @date   06 April 2026
 * @brief  Jinja2 template parser — AST nodes and parser
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#ifndef __JINJA_PARSER_H__
#define __JINJA_PARSER_H__

#include "lexer.h"
#include "value.h"
#include <memory>
#include <string>
#include <vector>

namespace causallm {
namespace jinja {

enum class NodeType {
  Text,
  Output,
  If,
  For,
  Set,
  Literal,
  Variable,
  BinaryOp,
  UnaryOp,
  GetAttr,
  GetItem,
  Filter,
  Call,
  Array,
  Conditional,
  Macro,
  MethodCall,
};

struct ASTNode {
  NodeType type;
  std::string str_val;
  Value literal_val;

  std::shared_ptr<ASTNode> left;
  std::shared_ptr<ASTNode> right;
  std::shared_ptr<ASTNode> condition;

  std::vector<std::shared_ptr<ASTNode>> children;
  std::vector<std::shared_ptr<ASTNode>> else_children;
  std::vector<std::pair<std::shared_ptr<ASTNode>,
                        std::vector<std::shared_ptr<ASTNode>>>>
    elif_branches;

  // For loop
  std::string loop_var;
  std::string loop_var2;

  // Filter/call arguments
  std::vector<std::shared_ptr<ASTNode>> args;
  std::vector<std::pair<std::string, std::shared_ptr<ASTNode>>> kwargs;

  // Macro
  std::vector<std::string> macro_params;
};

using ASTNodePtr = std::shared_ptr<ASTNode>;

class Parser {
public:
  explicit Parser(const std::vector<Token> &tokens);
  std::vector<ASTNodePtr> parse();

private:
  std::vector<Token> tokens_;
  size_t pos_;

  const Token &current() const;
  bool at_end() const;
  Token consume();
  Token expect(TokenType type);
  bool check(TokenType type) const;
  bool check_value(const std::string &val) const;

  // Top-level
  ASTNodePtr parse_text();
  ASTNodePtr parse_output();
  ASTNodePtr parse_statement();
  ASTNodePtr parse_if();
  ASTNodePtr parse_for();
  ASTNodePtr parse_set();
  ASTNodePtr parse_macro();

  // Expressions (precedence climbing)
  ASTNodePtr parse_expr();
  ASTNodePtr parse_conditional();
  ASTNodePtr parse_or();
  ASTNodePtr parse_and();
  ASTNodePtr parse_not();
  ASTNodePtr parse_compare();
  ASTNodePtr parse_add_sub();
  ASTNodePtr parse_mul_div();
  ASTNodePtr parse_unary();
  ASTNodePtr parse_postfix();
  ASTNodePtr parse_primary();
  ASTNodePtr parse_filter(ASTNodePtr expr);

  // Helpers
  std::vector<ASTNodePtr> parse_body(
    const std::vector<std::string> &end_tags);
  bool is_end_tag(const std::vector<std::string> &tags) const;
};

} // namespace jinja
} // namespace causallm

#endif // __JINJA_PARSER_H__

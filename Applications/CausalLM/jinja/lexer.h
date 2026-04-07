// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   lexer.h
 * @date   06 April 2026
 * @brief  Jinja2 template lexer
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#ifndef __JINJA_LEXER_H__
#define __JINJA_LEXER_H__

#include <string>
#include <vector>

namespace causallm {
namespace jinja {

enum class TokenType {
  Text,
  ExprStart,    // {{
  ExprEnd,      // }}
  StmtStart,    // {%
  StmtEnd,      // %}
  Identifier,
  StringLiteral,
  IntLiteral,
  FloatLiteral,
  Dot,
  Comma,
  Pipe,
  LParen,
  RParen,
  LBracket,
  RBracket,
  Plus,
  Minus,
  Star,
  Slash,
  Percent,
  Eq,
  Neq,
  Lt,
  Gt,
  LtEq,
  GtEq,
  Assign,
  Tilde,
  Colon,
  Eof,
};

struct Token {
  TokenType type;
  std::string value;
};

class Lexer {
public:
  explicit Lexer(const std::string &source);
  std::vector<Token> tokenize();

private:
  std::string src_;
  size_t pos_;

  char peek() const;
  char peek_at(size_t offset) const;
  char advance();
  bool at_end() const;
  bool match(const std::string &s) const;
  void consume(const std::string &s);

  void scan_text(std::vector<Token> &tokens);
  void scan_inner_tokens(std::vector<Token> &tokens, const std::string &end);
  Token scan_string();
  Token scan_number();
  Token scan_identifier();
  void skip_comment();
  void skip_whitespace();
};

} // namespace jinja
} // namespace causallm

#endif // __JINJA_LEXER_H__

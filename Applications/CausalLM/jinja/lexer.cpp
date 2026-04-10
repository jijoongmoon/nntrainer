// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   lexer.cpp
 * @date   06 April 2026
 * @brief  Jinja2 template lexer implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include "lexer.h"
#include <stdexcept>

namespace causallm {
namespace jinja {

Lexer::Lexer(const std::string &source) : src_(source), pos_(0) {}

char Lexer::peek() const {
  if (at_end())
    return '\0';
  return src_[pos_];
}

char Lexer::peek_at(size_t offset) const {
  if (pos_ + offset >= src_.size())
    return '\0';
  return src_[pos_ + offset];
}

char Lexer::advance() {
  if (at_end())
    return '\0';
  return src_[pos_++];
}

bool Lexer::at_end() const { return pos_ >= src_.size(); }

bool Lexer::match(const std::string &s) const {
  if (pos_ + s.size() > src_.size())
    return false;
  return src_.compare(pos_, s.size(), s) == 0;
}

void Lexer::consume(const std::string &s) {
  if (!match(s))
    throw std::runtime_error("Expected '" + s + "' at position " +
                             std::to_string(pos_));
  pos_ += s.size();
}

std::vector<Token> Lexer::tokenize() {
  std::vector<Token> tokens;

  while (!at_end()) {
    if (match("{#")) {
      skip_comment();
    } else if (match("{{")) {
      // Handle whitespace trimming: {{-
      pos_ += 2;
      if (!at_end() && peek() == '-') {
        pos_++;
        // Trim trailing whitespace from previous text token
        if (!tokens.empty() && tokens.back().type == TokenType::Text) {
          auto &txt = tokens.back().value;
          size_t end = txt.find_last_not_of(" \t\n\r");
          if (end != std::string::npos)
            txt = txt.substr(0, end + 1);
          else
            txt.clear();
        }
      }
      tokens.push_back({TokenType::ExprStart, "{{"});
      scan_inner_tokens(tokens, "}}");
    } else if (match("{%")) {
      pos_ += 2;
      if (!at_end() && peek() == '-') {
        pos_++;
        if (!tokens.empty() && tokens.back().type == TokenType::Text) {
          auto &txt = tokens.back().value;
          size_t end = txt.find_last_not_of(" \t\n\r");
          if (end != std::string::npos)
            txt = txt.substr(0, end + 1);
          else
            txt.clear();
        }
      }
      tokens.push_back({TokenType::StmtStart, "{%"});
      scan_inner_tokens(tokens, "%}");
    } else {
      scan_text(tokens);
    }
  }

  tokens.push_back({TokenType::Eof, ""});
  return tokens;
}

void Lexer::scan_text(std::vector<Token> &tokens) {
  std::string text;
  while (!at_end()) {
    if (match("{{") || match("{%") || match("{#"))
      break;
    text += advance();
  }
  if (!text.empty())
    tokens.push_back({TokenType::Text, text});
}

void Lexer::scan_inner_tokens(std::vector<Token> &tokens,
                              const std::string &end) {
  bool rtrim = false;

  while (!at_end()) {
    skip_whitespace();

    if (at_end())
      throw std::runtime_error("Unexpected end of template, expected '" + end +
                               "'");

    // Check for right-trim marker before end tag
    if (peek() == '-' && pos_ + 1 < src_.size()) {
      std::string check = std::string(1, src_[pos_ + 1]);
      if (pos_ + 2 < src_.size())
        check += src_[pos_ + 2];
      if (check == end) {
        rtrim = true;
        pos_++; // skip '-'
      }
    }

    if (match(end)) {
      pos_ += end.size();
      if (end == "}}")
        tokens.push_back({TokenType::ExprEnd, "}}"});
      else
        tokens.push_back({TokenType::StmtEnd, "%}"});

      // Handle right-trim: remove leading whitespace from next text
      if (rtrim) {
        // We'll mark that next text should be left-trimmed
        // Skip whitespace including one newline
        while (!at_end() && (peek() == ' ' || peek() == '\t'))
          pos_++;
        if (!at_end() && peek() == '\n')
          pos_++;
        else if (!at_end() && peek() == '\r') {
          pos_++;
          if (!at_end() && peek() == '\n')
            pos_++;
        }
      }
      return;
    }

    // Scan individual tokens
    char c = peek();
    if (c == '"' || c == '\'') {
      tokens.push_back(scan_string());
    } else if (std::isdigit(c)) {
      tokens.push_back(scan_number());
    } else if (std::isalpha(c) || c == '_') {
      tokens.push_back(scan_identifier());
    } else if (c == '.') {
      advance();
      tokens.push_back({TokenType::Dot, "."});
    } else if (c == ',') {
      advance();
      tokens.push_back({TokenType::Comma, ","});
    } else if (c == '|') {
      advance();
      tokens.push_back({TokenType::Pipe, "|"});
    } else if (c == '(') {
      advance();
      tokens.push_back({TokenType::LParen, "("});
    } else if (c == ')') {
      advance();
      tokens.push_back({TokenType::RParen, ")"});
    } else if (c == '[') {
      advance();
      tokens.push_back({TokenType::LBracket, "["});
    } else if (c == ']') {
      advance();
      tokens.push_back({TokenType::RBracket, "]"});
    } else if (c == '+') {
      advance();
      tokens.push_back({TokenType::Plus, "+"});
    } else if (c == '-') {
      advance();
      tokens.push_back({TokenType::Minus, "-"});
    } else if (c == '*') {
      advance();
      tokens.push_back({TokenType::Star, "*"});
    } else if (c == '/') {
      advance();
      tokens.push_back({TokenType::Slash, "/"});
    } else if (c == '%') {
      advance();
      tokens.push_back({TokenType::Percent, "%"});
    } else if (c == '~') {
      advance();
      tokens.push_back({TokenType::Tilde, "~"});
    } else if (c == ':') {
      advance();
      tokens.push_back({TokenType::Colon, ":"});
    } else if (c == '=' && peek_at(1) == '=') {
      pos_ += 2;
      tokens.push_back({TokenType::Eq, "=="});
    } else if (c == '!' && peek_at(1) == '=') {
      pos_ += 2;
      tokens.push_back({TokenType::Neq, "!="});
    } else if (c == '<' && peek_at(1) == '=') {
      pos_ += 2;
      tokens.push_back({TokenType::LtEq, "<="});
    } else if (c == '>' && peek_at(1) == '=') {
      pos_ += 2;
      tokens.push_back({TokenType::GtEq, ">="});
    } else if (c == '<') {
      advance();
      tokens.push_back({TokenType::Lt, "<"});
    } else if (c == '>') {
      advance();
      tokens.push_back({TokenType::Gt, ">"});
    } else if (c == '=') {
      advance();
      tokens.push_back({TokenType::Assign, "="});
    } else {
      throw std::runtime_error(std::string("Unexpected character '") + c +
                               "' at position " + std::to_string(pos_));
    }
  }

  throw std::runtime_error("Unexpected end of template, expected '" + end +
                           "'");
}

Token Lexer::scan_string() {
  char quote = advance(); // consume opening quote
  std::string value;

  while (!at_end() && peek() != quote) {
    if (peek() == '\\') {
      advance(); // skip backslash
      if (at_end())
        throw std::runtime_error("Unexpected end in string escape");
      char esc = advance();
      switch (esc) {
      case 'n':
        value += '\n';
        break;
      case 't':
        value += '\t';
        break;
      case 'r':
        value += '\r';
        break;
      case '\\':
        value += '\\';
        break;
      case '\'':
        value += '\'';
        break;
      case '"':
        value += '"';
        break;
      default:
        value += '\\';
        value += esc;
        break;
      }
    } else {
      value += advance();
    }
  }

  if (at_end())
    throw std::runtime_error("Unterminated string literal");
  advance(); // consume closing quote
  return {TokenType::StringLiteral, value};
}

Token Lexer::scan_number() {
  std::string num;
  bool has_dot = false;

  while (!at_end() && (std::isdigit(peek()) || peek() == '.')) {
    if (peek() == '.') {
      if (has_dot)
        break;
      has_dot = true;
    }
    num += advance();
  }

  return {has_dot ? TokenType::FloatLiteral : TokenType::IntLiteral, num};
}

Token Lexer::scan_identifier() {
  std::string id;
  while (!at_end() && (std::isalnum(peek()) || peek() == '_'))
    id += advance();
  return {TokenType::Identifier, id};
}

void Lexer::skip_comment() {
  consume("{#");
  while (!at_end()) {
    if (match("#}")) {
      pos_ += 2;
      return;
    }
    advance();
  }
  throw std::runtime_error("Unterminated comment");
}

void Lexer::skip_whitespace() {
  while (!at_end() && (peek() == ' ' || peek() == '\t' || peek() == '\n' ||
                       peek() == '\r'))
    advance();
}

} // namespace jinja
} // namespace causallm

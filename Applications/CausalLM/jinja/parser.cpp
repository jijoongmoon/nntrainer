// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   parser.cpp
 * @date   06 April 2026
 * @brief  Jinja2 template parser implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include "parser.h"
#include <stdexcept>

namespace causallm {
namespace jinja {

Parser::Parser(const std::vector<Token> &tokens) : tokens_(tokens), pos_(0) {}

const Token &Parser::current() const { return tokens_[pos_]; }

bool Parser::at_end() const {
  return pos_ >= tokens_.size() || current().type == TokenType::Eof;
}

Token Parser::consume() {
  if (at_end())
    throw std::runtime_error("Unexpected end of tokens");
  return tokens_[pos_++];
}

Token Parser::expect(TokenType type) {
  if (at_end() || current().type != type)
    throw std::runtime_error("Expected token type " +
                             std::to_string(static_cast<int>(type)) +
                             " but got " +
                             std::to_string(static_cast<int>(current().type)) +
                             " ('" + current().value + "')");
  return consume();
}

bool Parser::check(TokenType type) const {
  return !at_end() && current().type == type;
}

bool Parser::check_value(const std::string &val) const {
  return !at_end() && current().value == val;
}

std::vector<ASTNodePtr> Parser::parse() {
  std::vector<ASTNodePtr> nodes;
  while (!at_end()) {
    if (check(TokenType::Text)) {
      nodes.push_back(parse_text());
    } else if (check(TokenType::ExprStart)) {
      nodes.push_back(parse_output());
    } else if (check(TokenType::StmtStart)) {
      nodes.push_back(parse_statement());
    } else {
      break;
    }
  }
  return nodes;
}

ASTNodePtr Parser::parse_text() {
  auto node = std::make_shared<ASTNode>();
  node->type = NodeType::Text;
  node->str_val = consume().value;
  return node;
}

ASTNodePtr Parser::parse_output() {
  expect(TokenType::ExprStart);
  auto node = std::make_shared<ASTNode>();
  node->type = NodeType::Output;
  node->left = parse_expr();
  expect(TokenType::ExprEnd);
  return node;
}

ASTNodePtr Parser::parse_statement() {
  expect(TokenType::StmtStart);

  if (check_value("if")) {
    return parse_if();
  } else if (check_value("for")) {
    return parse_for();
  } else if (check_value("set")) {
    return parse_set();
  } else if (check_value("macro")) {
    return parse_macro();
  } else {
    throw std::runtime_error("Unknown statement: '" + current().value + "'");
  }
}

ASTNodePtr Parser::parse_if() {
  consume(); // 'if'
  auto node = std::make_shared<ASTNode>();
  node->type = NodeType::If;
  node->condition = parse_expr();
  expect(TokenType::StmtEnd);

  node->children = parse_body({"endif", "elif", "else"});

  while (check(TokenType::StmtStart)) {
    // Peek at the keyword after {%
    size_t saved = pos_;
    consume(); // StmtStart

    if (check_value("elif")) {
      consume(); // 'elif'
      auto elif_cond = parse_expr();
      expect(TokenType::StmtEnd);
      auto elif_body = parse_body({"endif", "elif", "else"});
      node->elif_branches.push_back({elif_cond, elif_body});
    } else if (check_value("else")) {
      consume(); // 'else'
      expect(TokenType::StmtEnd);
      node->else_children = parse_body({"endif"});
    } else if (check_value("endif")) {
      consume(); // 'endif'
      expect(TokenType::StmtEnd);
      return node;
    } else {
      pos_ = saved;
      break;
    }
  }

  return node;
}

ASTNodePtr Parser::parse_for() {
  consume(); // 'for'
  auto node = std::make_shared<ASTNode>();
  node->type = NodeType::For;
  node->loop_var = expect(TokenType::Identifier).value;

  // Check for "key, value" pattern
  if (check(TokenType::Comma)) {
    consume(); // ','
    node->loop_var2 = node->loop_var;
    node->loop_var = expect(TokenType::Identifier).value;
    // Now loop_var2 = key, loop_var = value
    // Swap so loop_var = first, loop_var2 = second
    std::swap(node->loop_var, node->loop_var2);
  }

  if (!check_value("in"))
    throw std::runtime_error("Expected 'in' in for loop");
  consume(); // 'in'

  node->left = parse_or(); // Use parse_or to avoid consuming 'if' as ternary

  // Optional 'if' filter
  if (check_value("if")) {
    consume(); // 'if'
    node->condition = parse_or();
  }

  // Optional 'recursive'
  if (check_value("recursive")) {
    consume();
  }

  expect(TokenType::StmtEnd);

  node->children = parse_body({"endfor", "else"});

  // Check for else clause (empty loop)
  if (check(TokenType::StmtStart)) {
    size_t saved = pos_;
    consume();
    if (check_value("else")) {
      consume();
      expect(TokenType::StmtEnd);
      node->else_children = parse_body({"endfor"});
      // consume endfor
      expect(TokenType::StmtStart);
      consume(); // 'endfor'
      expect(TokenType::StmtEnd);
    } else if (check_value("endfor")) {
      consume();
      expect(TokenType::StmtEnd);
    } else {
      pos_ = saved;
    }
  }

  return node;
}

ASTNodePtr Parser::parse_set() {
  consume(); // 'set'
  auto node = std::make_shared<ASTNode>();
  node->type = NodeType::Set;
  node->str_val = expect(TokenType::Identifier).value;

  // Support dotted names: {% set ns.attr = value %}
  while (check(TokenType::Dot)) {
    consume(); // '.'
    node->str_val += "." + expect(TokenType::Identifier).value;
  }

  expect(TokenType::Assign);
  node->left = parse_expr();
  expect(TokenType::StmtEnd);
  return node;
}

ASTNodePtr Parser::parse_macro() {
  consume(); // 'macro'
  auto node = std::make_shared<ASTNode>();
  node->type = NodeType::Macro;
  node->str_val = expect(TokenType::Identifier).value;

  expect(TokenType::LParen);
  while (!check(TokenType::RParen)) {
    if (!node->macro_params.empty())
      expect(TokenType::Comma);
    node->macro_params.push_back(expect(TokenType::Identifier).value);
  }
  expect(TokenType::RParen);
  expect(TokenType::StmtEnd);

  node->children = parse_body({"endmacro"});

  // consume endmacro
  expect(TokenType::StmtStart);
  consume(); // 'endmacro'
  expect(TokenType::StmtEnd);

  return node;
}

// Expression parsing with precedence climbing

ASTNodePtr Parser::parse_expr() { return parse_conditional(); }

ASTNodePtr Parser::parse_conditional() {
  auto expr = parse_or();

  // Ternary: value if condition else other
  if (check_value("if")) {
    consume(); // 'if'
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::Conditional;
    node->left = expr;
    node->condition = parse_or();
    if (check_value("else")) {
      consume(); // 'else'
      node->right = parse_or();
    } else {
      node->right = std::make_shared<ASTNode>();
      node->right->type = NodeType::Literal;
      node->right->literal_val = Value("");
    }
    return node;
  }

  return expr;
}

ASTNodePtr Parser::parse_or() {
  auto left = parse_and();
  while (check_value("or")) {
    consume();
    auto right = parse_and();
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::BinaryOp;
    node->str_val = "or";
    node->left = left;
    node->right = right;
    left = node;
  }
  return left;
}

ASTNodePtr Parser::parse_and() {
  auto left = parse_not();
  while (check_value("and")) {
    consume();
    auto right = parse_not();
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::BinaryOp;
    node->str_val = "and";
    node->left = left;
    node->right = right;
    left = node;
  }
  return left;
}

ASTNodePtr Parser::parse_not() {
  if (check_value("not")) {
    consume();
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::UnaryOp;
    node->str_val = "not";
    node->left = parse_not();
    return node;
  }
  return parse_compare();
}

ASTNodePtr Parser::parse_compare() {
  auto left = parse_add_sub();

  while (check(TokenType::Eq) || check(TokenType::Neq) ||
         check(TokenType::Lt) || check(TokenType::Gt) ||
         check(TokenType::LtEq) || check(TokenType::GtEq) ||
         check_value("in") || check_value("not") ||
         check_value("is")) {

    if (check_value("is")) {
      consume(); // 'is'
      bool negated = false;
      if (check_value("not")) {
        negated = true;
        consume(); // 'not'
      }
      // 'is defined', 'is none', 'is true', 'is false', etc.
      std::string test_name = expect(TokenType::Identifier).value;
      auto node = std::make_shared<ASTNode>();
      node->type = NodeType::BinaryOp;
      node->str_val = negated ? "is not " + test_name : "is " + test_name;
      node->left = left;
      left = node;
      continue;
    }

    if (check_value("not")) {
      // 'not in'
      consume(); // 'not'
      if (!check_value("in"))
        throw std::runtime_error("Expected 'in' after 'not'");
      consume(); // 'in'
      auto right = parse_add_sub();
      auto node = std::make_shared<ASTNode>();
      node->type = NodeType::BinaryOp;
      node->str_val = "not in";
      node->left = left;
      node->right = right;
      left = node;
      continue;
    }

    if (check_value("in")) {
      consume(); // 'in'
      auto right = parse_add_sub();
      auto node = std::make_shared<ASTNode>();
      node->type = NodeType::BinaryOp;
      node->str_val = "in";
      node->left = left;
      node->right = right;
      left = node;
      continue;
    }

    std::string op = consume().value;
    auto right = parse_add_sub();
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::BinaryOp;
    node->str_val = op;
    node->left = left;
    node->right = right;
    left = node;
  }

  return left;
}

ASTNodePtr Parser::parse_add_sub() {
  auto left = parse_mul_div();
  while (check(TokenType::Plus) || check(TokenType::Minus) ||
         check(TokenType::Tilde)) {
    std::string op = consume().value;
    if (op == "~")
      op = "~"; // string concat
    auto right = parse_mul_div();
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::BinaryOp;
    node->str_val = op;
    node->left = left;
    node->right = right;
    left = node;
  }
  return left;
}

ASTNodePtr Parser::parse_mul_div() {
  auto left = parse_unary();
  while (check(TokenType::Star) || check(TokenType::Slash) ||
         check(TokenType::Percent)) {
    std::string op = consume().value;
    auto right = parse_unary();
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::BinaryOp;
    node->str_val = op;
    node->left = left;
    node->right = right;
    left = node;
  }
  return left;
}

ASTNodePtr Parser::parse_unary() {
  if (check(TokenType::Minus)) {
    consume();
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::UnaryOp;
    node->str_val = "-";
    node->left = parse_unary();
    return node;
  }
  return parse_postfix();
}

ASTNodePtr Parser::parse_postfix() {
  auto expr = parse_primary();

  while (true) {
    if (check(TokenType::Dot)) {
      consume(); // '.'
      std::string attr = expect(TokenType::Identifier).value;

      // Check if it's a method call: expr.method(args)
      if (check(TokenType::LParen)) {
        consume(); // '('
        auto node = std::make_shared<ASTNode>();
        node->type = NodeType::MethodCall;
        node->str_val = attr;
        node->left = expr;
        while (!check(TokenType::RParen)) {
          if (!node->args.empty())
            expect(TokenType::Comma);
          // Check for keyword arg
          if (check(TokenType::Identifier)) {
            size_t saved = pos_;
            std::string name = consume().value;
            if (check(TokenType::Assign)) {
              consume(); // '='
              node->kwargs.push_back({name, parse_expr()});
            } else {
              pos_ = saved;
              node->args.push_back(parse_expr());
            }
          } else {
            node->args.push_back(parse_expr());
          }
        }
        expect(TokenType::RParen);
        expr = node;
      } else {
        auto node = std::make_shared<ASTNode>();
        node->type = NodeType::GetAttr;
        node->str_val = attr;
        node->left = expr;
        expr = node;
      }
    } else if (check(TokenType::LBracket)) {
      consume(); // '['
      auto node = std::make_shared<ASTNode>();
      node->type = NodeType::GetItem;
      node->left = expr;
      // Support slice notation [start:end]
      if (check(TokenType::Colon)) {
        consume(); // ':'
        node->str_val = "slice";
        node->right = std::make_shared<ASTNode>();
        node->right->type = NodeType::Literal;
        node->right->literal_val = Value(0); // start = 0
        if (!check(TokenType::RBracket)) {
          node->condition = parse_expr(); // end
        }
      } else {
        node->right = parse_expr();
        if (check(TokenType::Colon)) {
          consume(); // ':'
          node->str_val = "slice";
          if (!check(TokenType::RBracket)) {
            node->condition = parse_expr(); // end
          }
        }
      }
      expect(TokenType::RBracket);
      expr = node;
    } else if (check(TokenType::Pipe)) {
      expr = parse_filter(expr);
    } else {
      break;
    }
  }

  return expr;
}

ASTNodePtr Parser::parse_filter(ASTNodePtr expr) {
  consume(); // '|'
  auto node = std::make_shared<ASTNode>();
  node->type = NodeType::Filter;
  node->str_val = expect(TokenType::Identifier).value;
  node->left = expr;

  // Optional arguments
  if (check(TokenType::LParen)) {
    consume();
    while (!check(TokenType::RParen)) {
      if (!node->args.empty())
        expect(TokenType::Comma);
      // Check for keyword arg
      if (check(TokenType::Identifier)) {
        size_t saved = pos_;
        std::string name = consume().value;
        if (check(TokenType::Assign)) {
          consume();
          node->kwargs.push_back({name, parse_expr()});
        } else {
          pos_ = saved;
          node->args.push_back(parse_expr());
        }
      } else {
        node->args.push_back(parse_expr());
      }
    }
    expect(TokenType::RParen);
  }

  // Chain filters
  if (check(TokenType::Pipe))
    return parse_filter(node);

  return node;
}

ASTNodePtr Parser::parse_primary() {
  if (check(TokenType::StringLiteral)) {
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::Literal;
    node->literal_val = Value(consume().value);
    return node;
  }

  if (check(TokenType::IntLiteral)) {
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::Literal;
    node->literal_val = Value(static_cast<int64_t>(std::stoll(consume().value)));
    return node;
  }

  if (check(TokenType::FloatLiteral)) {
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::Literal;
    node->literal_val = Value(std::stod(consume().value));
    return node;
  }

  if (check_value("true") || check_value("True")) {
    consume();
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::Literal;
    node->literal_val = Value(true);
    return node;
  }

  if (check_value("false") || check_value("False")) {
    consume();
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::Literal;
    node->literal_val = Value(false);
    return node;
  }

  if (check_value("none") || check_value("None")) {
    consume();
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::Literal;
    node->literal_val = Value(nullptr);
    return node;
  }

  if (check(TokenType::LBracket)) {
    consume(); // '['
    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::Array;
    while (!check(TokenType::RBracket)) {
      if (!node->children.empty())
        expect(TokenType::Comma);
      if (check(TokenType::RBracket))
        break; // trailing comma
      node->children.push_back(parse_expr());
    }
    expect(TokenType::RBracket);
    return node;
  }

  if (check(TokenType::LParen)) {
    consume(); // '('
    auto expr = parse_expr();
    expect(TokenType::RParen);
    return expr;
  }

  if (check(TokenType::Identifier)) {
    std::string name = consume().value;

    // Namespace constructor: namespace(key=val, ...)
    if (name == "namespace" && check(TokenType::LParen)) {
      consume(); // '('
      auto node = std::make_shared<ASTNode>();
      node->type = NodeType::Call;
      node->str_val = "namespace";
      while (!check(TokenType::RParen)) {
        if (!node->kwargs.empty())
          expect(TokenType::Comma);
        std::string key = expect(TokenType::Identifier).value;
        expect(TokenType::Assign);
        node->kwargs.push_back({key, parse_expr()});
      }
      expect(TokenType::RParen);
      return node;
    }

    // Function call: name(args)
    if (check(TokenType::LParen)) {
      consume(); // '('
      auto node = std::make_shared<ASTNode>();
      node->type = NodeType::Call;
      node->str_val = name;
      while (!check(TokenType::RParen)) {
        if (!node->args.empty() || !node->kwargs.empty())
          expect(TokenType::Comma);
        // Check for keyword arg
        if (check(TokenType::Identifier)) {
          size_t saved = pos_;
          std::string argname = consume().value;
          if (check(TokenType::Assign)) {
            consume();
            node->kwargs.push_back({argname, parse_expr()});
          } else {
            pos_ = saved;
            node->args.push_back(parse_expr());
          }
        } else {
          node->args.push_back(parse_expr());
        }
      }
      expect(TokenType::RParen);
      return node;
    }

    auto node = std::make_shared<ASTNode>();
    node->type = NodeType::Variable;
    node->str_val = name;
    return node;
  }

  throw std::runtime_error("Unexpected token: '" + current().value +
                           "' (type=" +
                           std::to_string(static_cast<int>(current().type)) +
                           ")");
}

std::vector<ASTNodePtr>
Parser::parse_body(const std::vector<std::string> &end_tags) {
  std::vector<ASTNodePtr> body;

  while (!at_end()) {
    if (is_end_tag(end_tags))
      break;

    if (check(TokenType::Text)) {
      body.push_back(parse_text());
    } else if (check(TokenType::ExprStart)) {
      body.push_back(parse_output());
    } else if (check(TokenType::StmtStart)) {
      body.push_back(parse_statement());
    } else {
      break;
    }
  }

  return body;
}

bool Parser::is_end_tag(const std::vector<std::string> &tags) const {
  if (!check(TokenType::StmtStart))
    return false;
  if (pos_ + 1 >= tokens_.size())
    return false;

  const auto &next = tokens_[pos_ + 1];
  if (next.type != TokenType::Identifier)
    return false;

  for (const auto &tag : tags) {
    if (next.value == tag)
      return true;
  }
  return false;
}

} // namespace jinja
} // namespace causallm

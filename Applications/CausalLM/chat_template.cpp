// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    chat_template.cpp
 * @date    10 Apr 2026
 * @brief   Chat template implementation with mini Jinja2 renderer
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */

#include "chat_template.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace causallm {

// ============================================================================
// Token types for the Jinja2 lexer
// ============================================================================
/** @brief TokenType - enum class for Jinja2 template processing */
enum class TokenType {
  TEXT,
  EXPRESSION_START, // {{
  EXPRESSION_END,   // }}
  STATEMENT_START,  // {%
  STATEMENT_END,    // %}
  STRING,
  INTEGER,
  FLOAT,
  IDENTIFIER,
  DOT,
  LBRACKET,
  RBRACKET,
  LPAREN,
  RPAREN,
  PLUS,
  MINUS,
  PERCENT,
  PIPE,
  COMMA,
  EQ,     // ==
  NEQ,    // !=
  ASSIGN, // =
  NOT,    // not
  AND,    // and
  OR,     // or
  TRUE_LIT,
  FALSE_LIT,
  NONE_LIT,
  IF,
  ELIF,
  ELSE,
  ENDIF,
  FOR,
  IN,
  ENDFOR,
  SET,
  IS,
  TILDE, // ~
  COLON, // :
  GT,    // >
  LT,    // <
  GTE,   // >=
  LTE,   // <=
  END_OF_INPUT,
};

/** @brief Lexer token with type, value, and whitespace control flags */
struct Token {
  TokenType type;
  std::string value;
  bool strip_before = false; // {%- or {{-
  bool strip_after = false;  // -%} or -}}
};

// ============================================================================
// Lexer
// ============================================================================
/** @brief Tokenizes Jinja2 template strings into token sequences */
class Lexer {
public:
  /** @brief Construct lexer with input template string */
  explicit Lexer(const std::string &input) : input_(input), pos_(0) {}

  /** @brief Tokenize the input template into a sequence of tokens */
  std::vector<Token> tokenize() {
    std::vector<Token> tokens;

    while (pos_ < input_.size()) {
      if (match("{{")) {
        bool strip = false;
        if (pos_ < input_.size() && input_[pos_] == '-') {
          strip = true;
          pos_++;
        }
        Token start;
        start.type = TokenType::EXPRESSION_START;
        start.strip_before = strip;
        tokens.push_back(start);
        skipWhitespace();
        tokenizeInside(tokens, TokenType::EXPRESSION_END);
      } else if (match("{%")) {
        bool strip = false;
        if (pos_ < input_.size() && input_[pos_] == '-') {
          strip = true;
          pos_++;
        }
        Token start;
        start.type = TokenType::STATEMENT_START;
        start.strip_before = strip;
        tokens.push_back(start);
        skipWhitespace();
        tokenizeInside(tokens, TokenType::STATEMENT_END);
      } else {
        std::string text;
        while (pos_ < input_.size()) {
          if ((pos_ + 1 < input_.size()) &&
              ((input_[pos_] == '{' &&
                (input_[pos_ + 1] == '{' || input_[pos_ + 1] == '%')))) {
            break;
          }
          text += input_[pos_++];
        }
        if (!text.empty()) {
          Token t;
          t.type = TokenType::TEXT;
          t.value = text;
          tokens.push_back(t);
        }
      }
    }

    Token eof;
    eof.type = TokenType::END_OF_INPUT;
    tokens.push_back(eof);
    return tokens;
  }

private:
  /** @brief Try to match a string at current position and advance */
  bool match(const std::string &s) {
    if (pos_ + s.size() <= input_.size() &&
        input_.substr(pos_, s.size()) == s) {
      pos_ += s.size();
      return true;
    }
    return false;
  }

  /** @brief Skip whitespace characters at current position */
  void skipWhitespace() {
    while (pos_ < input_.size() && std::isspace(input_[pos_]))
      pos_++;
  }

  /** @brief Tokenize content inside expression or statement tags */
  void tokenizeInside(std::vector<Token> &tokens, TokenType end_type) {
    while (pos_ < input_.size()) {
      skipWhitespace();
      if (pos_ >= input_.size())
        break;

      // Check for closing tag
      if (end_type == TokenType::EXPRESSION_END) {
        if (pos_ + 1 < input_.size() && input_[pos_] == '-' &&
            input_[pos_ + 1] == '}' && pos_ + 2 < input_.size() &&
            input_[pos_ + 2] == '}') {
          pos_ += 3;
          Token end;
          end.type = end_type;
          end.strip_after = true;
          tokens.push_back(end);
          return;
        }
        if (match("}}")) {
          Token end;
          end.type = end_type;
          tokens.push_back(end);
          return;
        }
      } else if (end_type == TokenType::STATEMENT_END) {
        if (pos_ + 1 < input_.size() && input_[pos_] == '-' &&
            input_[pos_ + 1] == '%' && pos_ + 2 < input_.size() &&
            input_[pos_ + 2] == '}') {
          pos_ += 3;
          Token end;
          end.type = end_type;
          end.strip_after = true;
          tokens.push_back(end);
          return;
        }
        if (match("%}")) {
          Token end;
          end.type = end_type;
          tokens.push_back(end);
          return;
        }
      }

      // String literal
      if (input_[pos_] == '\'' || input_[pos_] == '"') {
        tokens.push_back(readString());
        continue;
      }

      // Number
      if (std::isdigit(input_[pos_])) {
        tokens.push_back(readNumber());
        continue;
      }

      // Identifier or keyword
      if (std::isalpha(input_[pos_]) || input_[pos_] == '_') {
        tokens.push_back(readIdentifier());
        continue;
      }

      // Operators and punctuation
      Token t;
      switch (input_[pos_]) {
      case '.':
        t.type = TokenType::DOT;
        t.value = ".";
        pos_++;
        break;
      case '[':
        t.type = TokenType::LBRACKET;
        t.value = "[";
        pos_++;
        break;
      case ']':
        t.type = TokenType::RBRACKET;
        t.value = "]";
        pos_++;
        break;
      case '(':
        t.type = TokenType::LPAREN;
        t.value = "(";
        pos_++;
        break;
      case ')':
        t.type = TokenType::RPAREN;
        t.value = ")";
        pos_++;
        break;
      case '+':
        t.type = TokenType::PLUS;
        t.value = "+";
        pos_++;
        break;
      case '-':
        t.type = TokenType::MINUS;
        t.value = "-";
        pos_++;
        break;
      case '%':
        t.type = TokenType::PERCENT;
        t.value = "%";
        pos_++;
        break;
      case '|':
        t.type = TokenType::PIPE;
        t.value = "|";
        pos_++;
        break;
      case ',':
        t.type = TokenType::COMMA;
        t.value = ",";
        pos_++;
        break;
      case '=':
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '=') {
          t.type = TokenType::EQ;
          t.value = "==";
          pos_ += 2;
        } else {
          t.type = TokenType::ASSIGN;
          t.value = "=";
          pos_++;
        }
        break;
      case '!':
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '=') {
          t.type = TokenType::NEQ;
          t.value = "!=";
          pos_ += 2;
        } else {
          pos_++;
          continue;
        }
        break;
      case '~':
        t.type = TokenType::TILDE;
        t.value = "~";
        pos_++;
        break;
      case ':':
        t.type = TokenType::COLON;
        t.value = ":";
        pos_++;
        break;
      case '>':
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '=') {
          t.type = TokenType::GTE;
          t.value = ">=";
          pos_ += 2;
        } else {
          t.type = TokenType::GT;
          t.value = ">";
          pos_++;
        }
        break;
      case '<':
        if (pos_ + 1 < input_.size() && input_[pos_ + 1] == '=') {
          t.type = TokenType::LTE;
          t.value = "<=";
          pos_ += 2;
        } else {
          t.type = TokenType::LT;
          t.value = "<";
          pos_++;
        }
        break;
      default:
        pos_++;
        continue;
      }
      tokens.push_back(t);
    }
  }

  /** @brief Read a string literal token */
  Token readString() {
    char quote = input_[pos_++];
    std::string value;
    while (pos_ < input_.size() && input_[pos_] != quote) {
      if (input_[pos_] == '\\' && pos_ + 1 < input_.size()) {
        pos_++;
        switch (input_[pos_]) {
        case 'n':
          value += '\n';
          break;
        case 't':
          value += '\t';
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
          value += input_[pos_];
          break;
        }
      } else {
        value += input_[pos_];
      }
      pos_++;
    }
    if (pos_ < input_.size())
      pos_++; // skip closing quote

    Token t;
    t.type = TokenType::STRING;
    t.value = value;
    return t;
  }

  /** @brief Read a numeric literal token */
  Token readNumber() {
    std::string value;
    bool has_dot = false;
    while (pos_ < input_.size() &&
           (std::isdigit(input_[pos_]) || input_[pos_] == '.')) {
      if (input_[pos_] == '.') {
        if (has_dot)
          break;
        has_dot = true;
      }
      value += input_[pos_++];
    }
    Token t;
    t.type = has_dot ? TokenType::FLOAT : TokenType::INTEGER;
    t.value = value;
    return t;
  }

  /** @brief Read an identifier or keyword token */
  Token readIdentifier() {
    std::string value;
    while (pos_ < input_.size() &&
           (std::isalnum(input_[pos_]) || input_[pos_] == '_')) {
      value += input_[pos_++];
    }

    Token t;
    t.value = value;

    // Check for keywords
    if (value == "if")
      t.type = TokenType::IF;
    else if (value == "elif")
      t.type = TokenType::ELIF;
    else if (value == "else")
      t.type = TokenType::ELSE;
    else if (value == "endif")
      t.type = TokenType::ENDIF;
    else if (value == "for")
      t.type = TokenType::FOR;
    else if (value == "in")
      t.type = TokenType::IN;
    else if (value == "endfor")
      t.type = TokenType::ENDFOR;
    else if (value == "set")
      t.type = TokenType::SET;
    else if (value == "not")
      t.type = TokenType::NOT;
    else if (value == "and")
      t.type = TokenType::AND;
    else if (value == "or")
      t.type = TokenType::OR;
    else if (value == "is")
      t.type = TokenType::IS;
    else if (value == "true" || value == "True")
      t.type = TokenType::TRUE_LIT;
    else if (value == "false" || value == "False")
      t.type = TokenType::FALSE_LIT;
    else if (value == "none" || value == "None")
      t.type = TokenType::NONE_LIT;
    else
      t.type = TokenType::IDENTIFIER;

    return t;
  }

  std::string input_;
  size_t pos_;
};

// ============================================================================
// AST Nodes
// ============================================================================
/** @brief Base AST node for template parsing */
struct ASTNode {
  virtual ~ASTNode() = default;
};

using ASTNodePtr = std::shared_ptr<ASTNode>;

/** @brief Base expression AST node */
struct ExprNode : ASTNode {};
using ExprNodePtr = std::shared_ptr<ExprNode>;

/** @brief AST node for literal text output */
struct TextNode : ASTNode {
  std::string text;
};

/** @brief AST node for expression output ({{ expr }}) */
struct OutputNode : ASTNode {
  ExprNodePtr expr;
  bool strip_before = false;
  bool strip_after = false;
};

/** @brief Single branch of an if/elif/else block */
struct IfBranch {
  ExprNodePtr condition; // nullptr for else branch
  std::vector<ASTNodePtr> body;
};

/** @brief AST node for if/elif/else conditional blocks */
struct IfNode : ASTNode {
  std::vector<IfBranch> branches;
  bool strip_before = false;
  bool strip_after = false;
};

/** @brief AST node for for-loop iteration blocks */
struct ForNode : ASTNode {
  std::string var_name;
  ExprNodePtr iterable;
  std::vector<ASTNodePtr> body;
  bool strip_before = false;
  bool strip_after = false;
};

/** @brief AST node for variable assignment (set statement) */
struct SetNode : ASTNode {
  std::string var_name;
  std::string attr_name; // for "set ns.attr = val" (empty if simple set)
  ExprNodePtr value;
  bool strip_before = false;
  bool strip_after = false;
};

// Expression nodes
/** @brief Expression node for string literal values */
struct StringLiteral : ExprNode {
  std::string value;
};

/** @brief Expression node for integer literal values */
struct IntegerLiteral : ExprNode {
  int value;
};

/** @brief Expression node for boolean literal values */
struct BoolLiteral : ExprNode {
  bool value;
};

/** @brief Expression node for None/null literal values */
struct NoneLiteral : ExprNode {};

/** @brief Expression node for variable references */
struct VariableExpr : ExprNode {
  std::string name;
};

/** @brief Expression node for attribute access (obj.attr) */
struct AttributeExpr : ExprNode {
  ExprNodePtr object;
  std::string attribute;
};

/** @brief Expression node for index access (obj[key]) */
struct IndexExpr : ExprNode {
  ExprNodePtr object;
  ExprNodePtr index;
};

/** @brief Expression node for binary operations (+, ==, and, etc.) */
struct BinaryExpr : ExprNode {
  std::string op; // "+", "==", "!=", "and", "or", "%"
  ExprNodePtr left;
  ExprNodePtr right;
};

/** @brief Expression node for unary operations (not) */
struct UnaryExpr : ExprNode {
  std::string op; // "not"
  ExprNodePtr operand;
};

/** @brief Expression node for filter application (val | filter) */
struct FilterExpr : ExprNode {
  ExprNodePtr value;
  std::string filter_name;
};

/** @brief Expression node for "is defined" test */
struct IsDefinedExpr : ExprNode {
  ExprNodePtr value;
};

/** @brief Expression node for function calls */
struct FunctionCallExpr : ExprNode {
  std::string name;
  std::vector<ExprNodePtr> args;
};

/** @brief Expression node for method calls (obj.method()) */
struct MethodCallExpr : ExprNode {
  ExprNodePtr object;
  std::string method;
  std::vector<ExprNodePtr> args;
};

/** @brief Expression node for slice operations (obj[start:stop:step]) */
struct SliceExpr : ExprNode {
  ExprNodePtr object;
  ExprNodePtr start; // nullable
  ExprNodePtr stop;  // nullable
  ExprNodePtr step;  // nullable
};

/** @brief Expression node for "in" containment test */
struct ContainsExpr : ExprNode {
  ExprNodePtr item;
  ExprNodePtr container;
};

// ============================================================================
// Parser
// ============================================================================
/** @brief Parses token sequences into an AST for template rendering */
class Parser {
public:
  explicit Parser(const std::vector<Token> &tokens) :
    tokens_(tokens), pos_(0) {}

  /** @brief Parse tokens into AST node list */
  std::vector<ASTNodePtr> parse() {
    std::vector<ASTNodePtr> nodes;
    parseBody(nodes, {});
    return nodes;
  }

private:
  /** @brief Get the current token without advancing */
  const Token &current() const { return tokens_[pos_]; }

  /** @brief Advance to next token and return the current one */
  const Token &advance() { return tokens_[pos_++]; }

  /** @brief Peek at the next token without advancing */
  const Token &peek() const { return tokens_[pos_ + 1]; }

  /** @brief Check if current token matches the given type */
  bool check(TokenType type) const { return current().type == type; }

  /** @brief Match current token type and advance if matched */
  bool matchToken(TokenType type) {
    if (check(type)) {
      advance();
      return true;
    }
    return false;
  }

  /** @brief Expect current token to match type, throw on mismatch */
  Token expect(TokenType type) {
    if (!check(type)) {
      throw std::runtime_error("ChatTemplate parser: unexpected token '" +
                               current().value + "', expected type " +
                               std::to_string(static_cast<int>(type)));
    }
    return advance();
  }

  /** @brief Parse template body until a stop keyword is found */
  void parseBody(std::vector<ASTNodePtr> &nodes,
                 const std::vector<TokenType> &stop_keywords) {
    while (pos_ < tokens_.size() && current().type != TokenType::END_OF_INPUT) {
      if (current().type == TokenType::TEXT) {
        auto node = std::make_shared<TextNode>();
        node->text = current().value;
        nodes.push_back(node);
        advance();
      } else if (current().type == TokenType::EXPRESSION_START) {
        nodes.push_back(parseOutput());
      } else if (current().type == TokenType::STATEMENT_START) {
        // Peek at the keyword after {%
        size_t save = pos_;
        advance(); // skip STATEMENT_START

        // Check if this is a stop keyword
        bool is_stop = false;
        for (auto sk : stop_keywords) {
          if (check(sk)) {
            is_stop = true;
            break;
          }
        }

        if (is_stop) {
          pos_ = save; // rewind
          return;
        }

        pos_ = save; // rewind
        parseStatement(nodes);
      } else {
        advance(); // skip unexpected
      }
    }
  }

  /** @brief Parse an output expression block ({{ expr }}) */
  ASTNodePtr parseOutput() {
    auto node = std::make_shared<OutputNode>();
    Token start = expect(TokenType::EXPRESSION_START);
    node->strip_before = start.strip_before;
    node->expr = parseExpression();
    Token end = expect(TokenType::EXPRESSION_END);
    node->strip_after = end.strip_after;
    return node;
  }

  /** @brief Parse a statement block ({% ... %}) */
  void parseStatement(std::vector<ASTNodePtr> &nodes) {
    Token start = expect(TokenType::STATEMENT_START);
    bool strip_before = start.strip_before;

    if (check(TokenType::IF)) {
      nodes.push_back(parseIf(strip_before));
    } else if (check(TokenType::FOR)) {
      nodes.push_back(parseFor(strip_before));
    } else if (check(TokenType::SET)) {
      nodes.push_back(parseSet(strip_before));
    } else {
      // Unknown statement - skip to end
      while (pos_ < tokens_.size() &&
             current().type != TokenType::STATEMENT_END) {
        advance();
      }
      if (check(TokenType::STATEMENT_END))
        advance();
    }
  }

  /** @brief Parse an if/elif/else/endif block */
  ASTNodePtr parseIf(bool strip_before) {
    auto node = std::make_shared<IfNode>();
    node->strip_before = strip_before;

    // Parse: if <expr> %}
    expect(TokenType::IF);
    IfBranch branch;
    branch.condition = parseExpression();
    Token end = expect(TokenType::STATEMENT_END);
    node->strip_after = end.strip_after;

    // Parse body until elif/else/endif
    parseBody(branch.body,
              {TokenType::ELIF, TokenType::ELSE, TokenType::ENDIF});
    node->branches.push_back(branch);

    // Parse elif/else branches
    while (pos_ < tokens_.size() &&
           current().type == TokenType::STATEMENT_START) {
      advance(); // skip {%

      if (check(TokenType::ELIF)) {
        advance(); // skip elif
        IfBranch elif_branch;
        elif_branch.condition = parseExpression();
        expect(TokenType::STATEMENT_END);
        parseBody(elif_branch.body,
                  {TokenType::ELIF, TokenType::ELSE, TokenType::ENDIF});
        node->branches.push_back(elif_branch);
      } else if (check(TokenType::ELSE)) {
        advance(); // skip else
        expect(TokenType::STATEMENT_END);
        IfBranch else_branch;
        else_branch.condition = nullptr; // no condition = else
        parseBody(else_branch.body, {TokenType::ENDIF});
        node->branches.push_back(else_branch);
      } else if (check(TokenType::ENDIF)) {
        advance(); // skip endif
        expect(TokenType::STATEMENT_END);
        break;
      } else {
        break;
      }
    }

    return node;
  }

  /** @brief Parse a for/endfor loop block */
  ASTNodePtr parseFor(bool strip_before) {
    auto node = std::make_shared<ForNode>();
    node->strip_before = strip_before;

    expect(TokenType::FOR);
    node->var_name = expect(TokenType::IDENTIFIER).value;
    expect(TokenType::IN);
    node->iterable = parseExpression();
    Token end = expect(TokenType::STATEMENT_END);
    node->strip_after = end.strip_after;

    parseBody(node->body, {TokenType::ENDFOR});

    // Consume endfor
    expect(TokenType::STATEMENT_START);
    expect(TokenType::ENDFOR);
    expect(TokenType::STATEMENT_END);

    return node;
  }

  /** @brief Parse a set variable assignment statement */
  ASTNodePtr parseSet(bool strip_before) {
    auto node = std::make_shared<SetNode>();
    node->strip_before = strip_before;

    expect(TokenType::SET);
    node->var_name = expect(TokenType::IDENTIFIER).value;

    // Handle dotted assignment: "set ns.attr = val"
    if (check(TokenType::DOT)) {
      advance();
      node->attr_name = expect(TokenType::IDENTIFIER).value;
    }

    expect(TokenType::ASSIGN);
    node->value = parseExpression();
    Token end = expect(TokenType::STATEMENT_END);
    node->strip_after = end.strip_after;

    return node;
  }

  // Expression parsing with precedence
  /** @brief Parse a complete expression with precedence */
  ExprNodePtr parseExpression() { return parseOr(); }

  /** @brief Parse OR boolean expression */
  ExprNodePtr parseOr() {
    auto left = parseAnd();
    while (check(TokenType::OR)) {
      advance();
      auto right = parseAnd();
      auto node = std::make_shared<BinaryExpr>();
      node->op = "or";
      node->left = left;
      node->right = right;
      left = node;
    }
    return left;
  }

  /** @brief Parse AND boolean expression */
  ExprNodePtr parseAnd() {
    auto left = parseNot();
    while (check(TokenType::AND)) {
      advance();
      auto right = parseNot();
      auto node = std::make_shared<BinaryExpr>();
      node->op = "and";
      node->left = left;
      node->right = right;
      left = node;
    }
    return left;
  }

  /** @brief Parse NOT unary boolean expression */
  ExprNodePtr parseNot() {
    if (check(TokenType::NOT)) {
      advance();
      auto node = std::make_shared<UnaryExpr>();
      node->op = "not";
      node->operand = parseNot();
      return node;
    }
    return parseComparison();
  }

  /** @brief Parse comparison and "is" test expressions */
  ExprNodePtr parseComparison() {
    auto left = parseContains();

    if (check(TokenType::EQ) || check(TokenType::NEQ) || check(TokenType::GT) ||
        check(TokenType::LT) || check(TokenType::GTE) ||
        check(TokenType::LTE)) {
      std::string op = advance().value;
      auto right = parseContains();
      auto node = std::make_shared<BinaryExpr>();
      node->op = op;
      node->left = left;
      node->right = right;
      return node;
    }

    // "is" tests: "is defined", "is not defined", "is string", "is false", etc.
    if (check(TokenType::IS)) {
      advance();
      bool negated = false;
      if (check(TokenType::NOT)) {
        negated = true;
        advance();
      }
      if (check(TokenType::IDENTIFIER)) {
        std::string test_name = current().value;
        advance();
        ExprNodePtr result;
        if (test_name == "defined") {
          auto node = std::make_shared<IsDefinedExpr>();
          node->value = left;
          result = node;
        } else if (test_name == "string") {
          // "is string" -> check if value is a string type
          auto call = std::make_shared<FunctionCallExpr>();
          call->name = "__is_string";
          call->args.push_back(left);
          result = call;
        } else if (test_name == "none") {
          auto call = std::make_shared<FunctionCallExpr>();
          call->name = "__is_none";
          call->args.push_back(left);
          result = call;
        } else if (test_name == "number") {
          auto call = std::make_shared<FunctionCallExpr>();
          call->name = "__is_number";
          call->args.push_back(left);
          result = call;
        } else {
          // Fallback: treat unknown test as comparison
          result = left;
        }
        if (negated) {
          auto not_node = std::make_shared<UnaryExpr>();
          not_node->op = "not";
          not_node->operand = result;
          return not_node;
        }
        return result;
      }
      // "is true" / "is false" / "is none" (keyword forms)
      if (check(TokenType::TRUE_LIT)) {
        advance();
        auto node = std::make_shared<BinaryExpr>();
        node->op = "==";
        node->left = left;
        auto lit = std::make_shared<BoolLiteral>();
        lit->value = true;
        node->right = lit;
        ExprNodePtr result = node;
        if (negated) {
          auto not_node = std::make_shared<UnaryExpr>();
          not_node->op = "not";
          not_node->operand = result;
          return not_node;
        }
        return result;
      }
      if (check(TokenType::FALSE_LIT)) {
        advance();
        auto node = std::make_shared<BinaryExpr>();
        node->op = "==";
        node->left = left;
        auto lit = std::make_shared<BoolLiteral>();
        lit->value = false;
        node->right = lit;
        ExprNodePtr result = node;
        if (negated) {
          auto not_node = std::make_shared<UnaryExpr>();
          not_node->op = "not";
          not_node->operand = result;
          return not_node;
        }
        return result;
      }
      if (check(TokenType::NONE_LIT)) {
        advance();
        auto node = std::make_shared<BinaryExpr>();
        node->op = "==";
        node->left = left;
        node->right = std::make_shared<NoneLiteral>();
        ExprNodePtr result = node;
        if (negated) {
          auto not_node = std::make_shared<UnaryExpr>();
          not_node->op = "not";
          not_node->operand = result;
          return not_node;
        }
        return result;
      }
    }

    return left;
  }

  // "in" / "not in" containment
  /** @brief Parse "in" and "not in" containment expressions */
  ExprNodePtr parseContains() {
    auto left = parseAddition();

    bool negated = false;
    if (check(TokenType::NOT) && pos_ + 1 < tokens_.size() &&
        peek().type == TokenType::IN) {
      advance(); // skip 'not'
      negated = true;
    }

    if (check(TokenType::IN)) {
      advance();
      auto right = parseAddition();
      auto node = std::make_shared<ContainsExpr>();
      node->item = left;
      node->container = right;
      if (negated) {
        auto not_node = std::make_shared<UnaryExpr>();
        not_node->op = "not";
        not_node->operand = node;
        return not_node;
      }
      return node;
    }

    return left;
  }

  /** @brief Parse addition, subtraction, and tilde concat */
  ExprNodePtr parseAddition() {
    auto left = parseModulo();
    while (check(TokenType::PLUS) || check(TokenType::MINUS) ||
           check(TokenType::TILDE)) {
      std::string op = advance().value;
      auto right = parseModulo();
      auto node = std::make_shared<BinaryExpr>();
      node->op = op;
      node->left = left;
      node->right = right;
      left = node;
    }
    return left;
  }

  /** @brief Parse modulo arithmetic expression */
  ExprNodePtr parseModulo() {
    auto left = parseFilter();
    while (check(TokenType::PERCENT)) {
      advance();
      auto right = parseFilter();
      auto node = std::make_shared<BinaryExpr>();
      node->op = "%";
      node->left = left;
      node->right = right;
      left = node;
    }
    return left;
  }

  /** @brief Parse pipe filter expression (val | filter) */
  ExprNodePtr parseFilter() {
    auto left = parsePostfix();
    while (check(TokenType::PIPE)) {
      advance();
      std::string filter_name = expect(TokenType::IDENTIFIER).value;
      auto node = std::make_shared<FilterExpr>();
      node->value = left;
      node->filter_name = filter_name;
      left = node;
    }
    return left;
  }

  /** @brief Parse postfix operations (dot, index, method, slice) */
  ExprNodePtr parsePostfix() {
    auto node = parsePrimary();
    while (true) {
      if (check(TokenType::DOT)) {
        advance();
        std::string attr = expect(TokenType::IDENTIFIER).value;
        // Check for method call: obj.method(args)
        if (check(TokenType::LPAREN)) {
          advance();
          auto call = std::make_shared<MethodCallExpr>();
          call->object = node;
          call->method = attr;
          if (!check(TokenType::RPAREN)) {
            call->args.push_back(parseExpression());
            while (check(TokenType::COMMA)) {
              advance();
              call->args.push_back(parseExpression());
            }
          }
          expect(TokenType::RPAREN);
          node = call;
        } else {
          auto access = std::make_shared<AttributeExpr>();
          access->object = node;
          access->attribute = attr;
          node = access;
        }
      } else if (check(TokenType::LBRACKET)) {
        advance();
        // Check for slice: obj[start:stop:step] or obj[::step]
        if (check(TokenType::COLON)) {
          // [:stop] or [::step]
          auto slice = std::make_shared<SliceExpr>();
          slice->object = node;
          slice->start = nullptr;
          advance(); // skip first ':'
          if (check(TokenType::COLON)) {
            // [::step]
            advance();
            slice->stop = nullptr;
            if (!check(TokenType::RBRACKET)) {
              slice->step = parseExpression();
            }
          } else if (!check(TokenType::RBRACKET)) {
            slice->stop = parseExpression();
            if (check(TokenType::COLON)) {
              advance();
              if (!check(TokenType::RBRACKET)) {
                slice->step = parseExpression();
              }
            }
          }
          expect(TokenType::RBRACKET);
          node = slice;
        } else {
          auto index = parseExpression();
          if (check(TokenType::COLON)) {
            // [start:stop] or [start:stop:step]
            advance();
            auto slice = std::make_shared<SliceExpr>();
            slice->object = node;
            slice->start = index;
            if (check(TokenType::COLON)) {
              // [start::step]
              advance();
              slice->stop = nullptr;
              if (!check(TokenType::RBRACKET)) {
                slice->step = parseExpression();
              }
            } else if (!check(TokenType::RBRACKET)) {
              slice->stop = parseExpression();
              if (check(TokenType::COLON)) {
                advance();
                if (!check(TokenType::RBRACKET)) {
                  slice->step = parseExpression();
                }
              }
            }
            expect(TokenType::RBRACKET);
            node = slice;
          } else {
            expect(TokenType::RBRACKET);
            auto access = std::make_shared<IndexExpr>();
            access->object = node;
            access->index = index;
            node = access;
          }
        }
      } else {
        break;
      }
    }
    return node;
  }

  /** @brief Parse primary expression (literals, variables, parens) */
  ExprNodePtr parsePrimary() {
    // Unary minus
    if (check(TokenType::MINUS)) {
      advance();
      auto operand = parsePrimary();
      if (auto *intLit = dynamic_cast<IntegerLiteral *>(operand.get())) {
        intLit->value = -intLit->value;
        return operand;
      }
      auto node = std::make_shared<BinaryExpr>();
      node->op = "-";
      auto zero = std::make_shared<IntegerLiteral>();
      zero->value = 0;
      node->left = zero;
      node->right = operand;
      return node;
    }
    if (check(TokenType::STRING)) {
      auto node = std::make_shared<StringLiteral>();
      node->value = advance().value;
      return node;
    }
    if (check(TokenType::INTEGER)) {
      auto node = std::make_shared<IntegerLiteral>();
      node->value = std::stoi(advance().value);
      return node;
    }
    if (check(TokenType::TRUE_LIT)) {
      advance();
      auto node = std::make_shared<BoolLiteral>();
      node->value = true;
      return node;
    }
    if (check(TokenType::FALSE_LIT)) {
      advance();
      auto node = std::make_shared<BoolLiteral>();
      node->value = false;
      return node;
    }
    if (check(TokenType::NONE_LIT)) {
      advance();
      return std::make_shared<NoneLiteral>();
    }
    if (check(TokenType::IDENTIFIER)) {
      std::string name = advance().value;

      // Check for function call
      if (check(TokenType::LPAREN)) {
        advance();
        // Special case: namespace(key=val, ...) with keyword arguments
        if (name == "namespace") {
          // Parse keyword arguments and build a JSON object initializer
          // We create a FunctionCallExpr where args[0] is a JSON-like init
          auto call = std::make_shared<FunctionCallExpr>();
          call->name = name;
          // Parse key=value pairs and store as string literal pairs
          json init_pairs = json::object();
          while (!check(TokenType::RPAREN) && pos_ < tokens_.size()) {
            if (check(TokenType::IDENTIFIER)) {
              std::string key = advance().value;
              if (check(TokenType::ASSIGN)) {
                advance();
                // We can't fully evaluate here, so skip for now
                // Just consume until comma or rparen
                auto val_expr = parseExpression();
                // Store key for later reference
              }
            }
            if (check(TokenType::COMMA))
              advance();
            else
              break;
          }
          expect(TokenType::RPAREN);
          return call;
        }
        auto call = std::make_shared<FunctionCallExpr>();
        call->name = name;
        if (!check(TokenType::RPAREN)) {
          call->args.push_back(parseExpression());
          while (check(TokenType::COMMA)) {
            advance();
            call->args.push_back(parseExpression());
          }
        }
        expect(TokenType::RPAREN);
        return call;
      }

      auto node = std::make_shared<VariableExpr>();
      node->name = name;
      return node;
    }
    if (check(TokenType::LPAREN)) {
      advance();
      auto expr = parseExpression();
      expect(TokenType::RPAREN);
      return expr;
    }

    // Fallback: return an empty string literal
    return std::make_shared<StringLiteral>();
  }

  const std::vector<Token> &tokens_;
  size_t pos_;
};

// ============================================================================
// Evaluator
// ============================================================================
/** @brief Evaluates AST nodes to produce rendered template output */
class Evaluator {
public:
  /** @brief Construct evaluator with template context variables */
  explicit Evaluator(const json &context) { scopes_.push_back(context); }

  /** @brief Evaluate AST node list and return rendered string */
  std::string evaluate(const std::vector<ASTNodePtr> &nodes) {
    std::string result;
    for (size_t i = 0; i < nodes.size(); ++i) {
      std::string chunk = evalNode(nodes[i].get());

      // Handle whitespace stripping
      if (shouldStripBefore(nodes[i].get())) {
        // Strip trailing whitespace from result
        while (!result.empty() &&
               (result.back() == ' ' || result.back() == '\t' ||
                result.back() == '\n' || result.back() == '\r')) {
          result.pop_back();
        }
      }

      result += chunk;

      // Handle strip_after: strip leading whitespace of next text
      if (shouldStripAfter(nodes[i].get()) && i + 1 < nodes.size()) {
        auto *text = dynamic_cast<TextNode *>(nodes[i + 1].get());
        if (text) {
          size_t start = 0;
          while (start < text->text.size() &&
                 (text->text[start] == ' ' || text->text[start] == '\t' ||
                  text->text[start] == '\n' || text->text[start] == '\r')) {
            start++;
          }
          text->text = text->text.substr(start);
        }
      }
    }
    return result;
  }

private:
  /** @brief Check if node has strip-before whitespace control */
  bool shouldStripBefore(ASTNode *node) {
    if (auto *o = dynamic_cast<OutputNode *>(node))
      return o->strip_before;
    if (auto *i = dynamic_cast<IfNode *>(node))
      return i->strip_before;
    if (auto *f = dynamic_cast<ForNode *>(node))
      return f->strip_before;
    if (auto *s = dynamic_cast<SetNode *>(node))
      return s->strip_before;
    return false;
  }

  /** @brief Check if node has strip-after whitespace control */
  bool shouldStripAfter(ASTNode *node) {
    if (auto *o = dynamic_cast<OutputNode *>(node))
      return o->strip_after;
    if (auto *i = dynamic_cast<IfNode *>(node))
      return i->strip_after;
    if (auto *f = dynamic_cast<ForNode *>(node))
      return f->strip_after;
    if (auto *s = dynamic_cast<SetNode *>(node))
      return s->strip_after;
    return false;
  }

  /** @brief Evaluate a single AST node and return its string result */
  std::string evalNode(ASTNode *node) {
    if (auto *text = dynamic_cast<TextNode *>(node)) {
      return text->text;
    }
    if (auto *output = dynamic_cast<OutputNode *>(node)) {
      json val = evalExpr(output->expr.get());
      return jsonToString(val);
    }
    if (auto *if_node = dynamic_cast<IfNode *>(node)) {
      return evalIf(if_node);
    }
    if (auto *for_node = dynamic_cast<ForNode *>(node)) {
      return evalFor(for_node);
    }
    if (auto *set_node = dynamic_cast<SetNode *>(node)) {
      json val = evalExpr(set_node->value.get());
      if (!set_node->attr_name.empty()) {
        // Namespace attribute mutation: "set ns.attr = val"
        // Find the namespace object and mutate it in place
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
          if (it->contains(set_node->var_name)) {
            (*it)[set_node->var_name][set_node->attr_name] = val;
            break;
          }
        }
      } else {
        setVariable(set_node->var_name, val);
      }
      return "";
    }
    return "";
  }

  /** @brief Evaluate an if/elif/else conditional node */
  std::string evalIf(IfNode *node) {
    for (auto &branch : node->branches) {
      if (!branch.condition || isTruthy(evalExpr(branch.condition.get()))) {
        return evaluate(branch.body);
      }
    }
    return "";
  }

  /** @brief Evaluate a for-loop node over an iterable */
  std::string evalFor(ForNode *node) {
    json iterable = evalExpr(node->iterable.get());
    if (!iterable.is_array())
      return "";

    std::string result;
    size_t size = iterable.size();

    for (size_t i = 0; i < size; ++i) {
      // Push new scope with loop variable
      json scope;
      scope[node->var_name] = iterable[i];

      // Loop context
      json loop;
      loop["index"] = static_cast<int>(i + 1);
      loop["index0"] = static_cast<int>(i);
      loop["first"] = (i == 0);
      loop["last"] = (i == size - 1);
      loop["length"] = static_cast<int>(size);
      scope["loop"] = loop;

      scopes_.push_back(scope);
      result += evaluate(node->body);
      scopes_.pop_back();
    }

    return result;
  }

  /** @brief Evaluate an expression node and return JSON value */
  json evalExpr(ExprNode *node) {
    if (auto *str = dynamic_cast<StringLiteral *>(node)) {
      return str->value;
    }
    if (auto *num = dynamic_cast<IntegerLiteral *>(node)) {
      return num->value;
    }
    if (auto *b = dynamic_cast<BoolLiteral *>(node)) {
      return b->value;
    }
    if (dynamic_cast<NoneLiteral *>(node)) {
      return nullptr;
    }
    if (auto *var = dynamic_cast<VariableExpr *>(node)) {
      return lookupVariable(var->name);
    }
    if (auto *attr = dynamic_cast<AttributeExpr *>(node)) {
      json obj = evalExpr(attr->object.get());
      if (obj.is_object() && obj.contains(attr->attribute)) {
        return obj[attr->attribute];
      }
      return nullptr;
    }
    if (auto *idx = dynamic_cast<IndexExpr *>(node)) {
      json obj = evalExpr(idx->object.get());
      json index = evalExpr(idx->index.get());
      if (obj.is_array() && index.is_number_integer()) {
        int i = index.get<int>();
        int sz = static_cast<int>(obj.size());
        if (i < 0)
          i += sz;
        if (i >= 0 && i < sz)
          return obj[i];
      } else if (obj.is_object() && index.is_string()) {
        std::string key = index.get<std::string>();
        if (obj.contains(key))
          return obj[key];
      }
      return nullptr;
    }
    if (auto *bin = dynamic_cast<BinaryExpr *>(node)) {
      return evalBinary(bin);
    }
    if (auto *unary = dynamic_cast<UnaryExpr *>(node)) {
      if (unary->op == "not") {
        return !isTruthy(evalExpr(unary->operand.get()));
      }
    }
    if (auto *filter = dynamic_cast<FilterExpr *>(node)) {
      json val = evalExpr(filter->value.get());
      if (filter->filter_name == "trim" && val.is_string()) {
        std::string s = val.get<std::string>();
        // Trim whitespace
        size_t start = s.find_first_not_of(" \t\n\r");
        size_t end = s.find_last_not_of(" \t\n\r");
        if (start == std::string::npos)
          return "";
        return s.substr(start, end - start + 1);
      }
      if (filter->filter_name == "length") {
        if (val.is_array())
          return static_cast<int>(val.size());
        if (val.is_string())
          return static_cast<int>(val.get<std::string>().size());
        return 0;
      }
      if (filter->filter_name == "tojson") {
        return val.dump();
      }
      return val; // unknown filter, passthrough
    }
    if (auto *def = dynamic_cast<IsDefinedExpr *>(node)) {
      // Check if the variable exists in any scope
      if (auto *var = dynamic_cast<VariableExpr *>(def->value.get())) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
          if (it->contains(var->name))
            return true;
        }
        return false;
      }
      // For attribute access, check if the parent exists and has the attr
      if (auto *attr = dynamic_cast<AttributeExpr *>(def->value.get())) {
        json obj = evalExpr(attr->object.get());
        return obj.is_object() && obj.contains(attr->attribute);
      }
      return false;
    }
    if (auto *call = dynamic_cast<FunctionCallExpr *>(node)) {
      if (call->name == "raise_exception") {
        std::string msg = "Template error";
        if (!call->args.empty()) {
          json arg = evalExpr(call->args[0].get());
          if (arg.is_string())
            msg = arg.get<std::string>();
        }
        throw std::runtime_error("ChatTemplate: " + msg);
      }
      if (call->name == "namespace") {
        // namespace() creates a mutable object that persists across scopes
        // keyword args are not parsed at AST level, so we return empty object
        // The actual initialization happens via {% set ns.attr = val %}
        return json::object();
      }
      // Type-checking built-in tests
      if (call->name == "__is_string") {
        if (!call->args.empty()) {
          json val = evalExpr(call->args[0].get());
          return val.is_string();
        }
        return false;
      }
      if (call->name == "__is_none") {
        if (!call->args.empty()) {
          json val = evalExpr(call->args[0].get());
          return val.is_null();
        }
        return false;
      }
      if (call->name == "__is_number") {
        if (!call->args.empty()) {
          json val = evalExpr(call->args[0].get());
          return val.is_number();
        }
        return false;
      }
      return nullptr;
    }
    if (auto *method = dynamic_cast<MethodCallExpr *>(node)) {
      return evalMethodCall(method);
    }
    if (auto *slice = dynamic_cast<SliceExpr *>(node)) {
      return evalSlice(slice);
    }
    if (auto *contains = dynamic_cast<ContainsExpr *>(node)) {
      json item = evalExpr(contains->item.get());
      json container = evalExpr(contains->container.get());
      // String containment: 'x' in 'xyz'
      if (item.is_string() && container.is_string()) {
        return container.get<std::string>().find(item.get<std::string>()) !=
               std::string::npos;
      }
      // Array containment
      if (container.is_array()) {
        for (const auto &elem : container) {
          if (elem == item)
            return true;
        }
        return false;
      }
      // Object key containment
      if (container.is_object() && item.is_string()) {
        return container.contains(item.get<std::string>());
      }
      return false;
    }
    return nullptr;
  }

  /** @brief Evaluate a method call on a string object */
  json evalMethodCall(MethodCallExpr *node) {
    json obj = evalExpr(node->object.get());
    const std::string &method = node->method;

    if (obj.is_string()) {
      std::string s = obj.get<std::string>();

      if (method == "startswith" && !node->args.empty()) {
        json arg = evalExpr(node->args[0].get());
        if (arg.is_string()) {
          std::string prefix = arg.get<std::string>();
          return s.size() >= prefix.size() &&
                 s.compare(0, prefix.size(), prefix) == 0;
        }
        return false;
      }
      if (method == "endswith" && !node->args.empty()) {
        json arg = evalExpr(node->args[0].get());
        if (arg.is_string()) {
          std::string suffix = arg.get<std::string>();
          return s.size() >= suffix.size() &&
                 s.compare(s.size() - suffix.size(), suffix.size(), suffix) ==
                   0;
        }
        return false;
      }
      if (method == "strip") {
        std::string chars = " \t\n\r";
        if (!node->args.empty()) {
          json arg = evalExpr(node->args[0].get());
          if (arg.is_string())
            chars = arg.get<std::string>();
        }
        size_t start = s.find_first_not_of(chars);
        if (start == std::string::npos)
          return std::string("");
        size_t end = s.find_last_not_of(chars);
        return s.substr(start, end - start + 1);
      }
      if (method == "lstrip") {
        std::string chars = " \t\n\r";
        if (!node->args.empty()) {
          json arg = evalExpr(node->args[0].get());
          if (arg.is_string())
            chars = arg.get<std::string>();
        }
        size_t start = s.find_first_not_of(chars);
        if (start == std::string::npos)
          return std::string("");
        return s.substr(start);
      }
      if (method == "rstrip") {
        std::string chars = " \t\n\r";
        if (!node->args.empty()) {
          json arg = evalExpr(node->args[0].get());
          if (arg.is_string())
            chars = arg.get<std::string>();
        }
        size_t end = s.find_last_not_of(chars);
        if (end == std::string::npos)
          return std::string("");
        return s.substr(0, end + 1);
      }
      if (method == "split" && !node->args.empty()) {
        json arg = evalExpr(node->args[0].get());
        if (arg.is_string()) {
          std::string delimiter = arg.get<std::string>();
          json result = json::array();
          size_t pos = 0;
          size_t found;
          while ((found = s.find(delimiter, pos)) != std::string::npos) {
            result.push_back(s.substr(pos, found - pos));
            pos = found + delimiter.size();
          }
          result.push_back(s.substr(pos));
          return result;
        }
      }
      if (method == "upper") {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(), ::toupper);
        return result;
      }
      if (method == "lower") {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
      }
    }

    return nullptr;
  }

  /** @brief Evaluate an array slice expression */
  json evalSlice(SliceExpr *node) {
    json obj = evalExpr(node->object.get());
    if (!obj.is_array())
      return json::array();

    int size = static_cast<int>(obj.size());
    int start = 0, stop = size, step = 1;

    if (node->start)
      start = evalExpr(node->start.get()).get<int>();
    if (node->stop)
      stop = evalExpr(node->stop.get()).get<int>();
    if (node->step)
      step = evalExpr(node->step.get()).get<int>();

    // Handle negative indices
    if (start < 0)
      start = std::max(0, size + start);
    if (stop < 0)
      stop = std::max(0, size + stop);

    // Clamp
    start = std::max(0, std::min(start, size));
    stop = std::max(0, std::min(stop, size));

    json result = json::array();
    if (step > 0) {
      for (int i = start; i < stop; i += step)
        result.push_back(obj[i]);
    } else if (step < 0) {
      // Reverse iteration: e.g., [::-1]
      if (!node->start)
        start = size - 1;
      if (!node->stop)
        stop = -1;
      else if (stop < 0)
        stop = std::max(-1, size + stop);
      // Re-clamp for reverse
      if (start >= size)
        start = size - 1;
      for (int i = start; i > stop; i += step) {
        if (i >= 0 && i < size)
          result.push_back(obj[i]);
      }
    }

    return result;
  }

  /** @brief Evaluate a binary operation expression */
  json evalBinary(BinaryExpr *node) {
    json left = evalExpr(node->left.get());
    json right = evalExpr(node->right.get());

    if (node->op == "+" || node->op == "~") {
      if (node->op == "~") {
        // Tilde always does string concat
        return jsonToString(left) + jsonToString(right);
      }
      if (left.is_string() && right.is_string()) {
        return left.get<std::string>() + right.get<std::string>();
      }
      if (left.is_number() && right.is_number()) {
        return left.get<int>() + right.get<int>();
      }
      // String + non-string: convert to string
      return jsonToString(left) + jsonToString(right);
    }
    if (node->op == "-") {
      if (left.is_number() && right.is_number()) {
        return left.get<int>() - right.get<int>();
      }
      return 0;
    }
    if (node->op == "==") {
      return left == right;
    }
    if (node->op == "!=") {
      return left != right;
    }
    if (node->op == ">") {
      if (left.is_number() && right.is_number())
        return left.get<int>() > right.get<int>();
      return false;
    }
    if (node->op == "<") {
      if (left.is_number() && right.is_number())
        return left.get<int>() < right.get<int>();
      return false;
    }
    if (node->op == ">=") {
      if (left.is_number() && right.is_number())
        return left.get<int>() >= right.get<int>();
      return false;
    }
    if (node->op == "<=") {
      if (left.is_number() && right.is_number())
        return left.get<int>() <= right.get<int>();
      return false;
    }
    if (node->op == "%") {
      if (left.is_number_integer() && right.is_number_integer()) {
        int r = right.get<int>();
        if (r != 0)
          return left.get<int>() % r;
      }
      return 0;
    }
    if (node->op == "and") {
      return isTruthy(left) && isTruthy(right);
    }
    if (node->op == "or") {
      return isTruthy(left) || isTruthy(right);
    }
    return nullptr;
  }

  /** @brief Check if a JSON value is truthy */
  bool isTruthy(const json &val) {
    if (val.is_null())
      return false;
    if (val.is_boolean())
      return val.get<bool>();
    if (val.is_number_integer())
      return val.get<int>() != 0;
    if (val.is_string())
      return !val.get<std::string>().empty();
    if (val.is_array())
      return !val.empty();
    if (val.is_object())
      return !val.empty();
    return false;
  }

  /** @brief Convert a JSON value to its string representation */
  std::string jsonToString(const json &val) {
    if (val.is_string())
      return val.get<std::string>();
    if (val.is_null())
      return "";
    if (val.is_boolean())
      return val.get<bool>() ? "True" : "False";
    if (val.is_number_integer())
      return std::to_string(val.get<int>());
    if (val.is_number_float())
      return std::to_string(val.get<double>());
    return val.dump();
  }

  /** @brief Look up a variable name in scope chain */
  json lookupVariable(const std::string &name) {
    // Search from innermost scope to outermost
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      if (it->contains(name))
        return (*it)[name];
    }
    return nullptr;
  }

  /** @brief Set a variable in the current innermost scope */
  void setVariable(const std::string &name, const json &value) {
    // Set in the current (innermost) scope
    if (!scopes_.empty()) {
      scopes_.back()[name] = value;
    }
  }

  std::vector<json> scopes_;
};

// ============================================================================
// ChatTemplate Implementation
// ============================================================================

ChatTemplate::ChatTemplate() : available_(false) {}

ChatTemplate ChatTemplate::fromFile(const std::string &tokenizer_config_path,
                                    const std::string &template_name) {
  ChatTemplate tmpl;

  std::ifstream file(tokenizer_config_path);
  if (!file.is_open()) {
    std::cerr << "[ChatTemplate] Warning: cannot open " << tokenizer_config_path
              << std::endl;
    return tmpl;
  }

  json config;
  try {
    file >> config;
  } catch (const json::parse_error &e) {
    std::cerr << "[ChatTemplate] Warning: JSON parse error in "
              << tokenizer_config_path << ": " << e.what() << std::endl;
    return tmpl;
  }

  // Extract chat_template
  if (config.contains("chat_template")) {
    if (config["chat_template"].is_string()) {
      tmpl.template_str_ = config["chat_template"].get<std::string>();
    } else if (config["chat_template"].is_array()) {
      // Array of named templates: select by template_name
      // First pass: find matching name
      for (const auto &entry : config["chat_template"]) {
        if (entry.is_object() && entry.contains("name") &&
            entry.contains("template") &&
            entry["name"].get<std::string>() == template_name) {
          tmpl.template_str_ = entry["template"].get<std::string>();
          break;
        }
      }
      // Fallback: if requested name not found, use first template
      if (tmpl.template_str_.empty()) {
        for (const auto &entry : config["chat_template"]) {
          if (entry.is_object() && entry.contains("template")) {
            std::string fallback_name = entry.contains("name")
                                          ? entry["name"].get<std::string>()
                                          : "(unnamed)";
            std::cerr << "[ChatTemplate] Template '" << template_name
                      << "' not found. Using '" << fallback_name << "'."
                      << std::endl;
            tmpl.template_str_ = entry["template"].get<std::string>();
            break;
          }
        }
      }
    }
  }

  if (tmpl.template_str_.empty()) {
    std::cerr << "[ChatTemplate] Warning: no 'chat_template' field found in "
              << tokenizer_config_path << std::endl;
    return tmpl;
  }

  // Extract bos_token (can be string or object with "content" field)
  if (config.contains("bos_token")) {
    if (config["bos_token"].is_string()) {
      tmpl.bos_token_ = config["bos_token"].get<std::string>();
    } else if (config["bos_token"].is_object() &&
               config["bos_token"].contains("content")) {
      tmpl.bos_token_ = config["bos_token"]["content"].get<std::string>();
    }
  }

  // Extract eos_token
  if (config.contains("eos_token")) {
    if (config["eos_token"].is_string()) {
      tmpl.eos_token_ = config["eos_token"].get<std::string>();
    } else if (config["eos_token"].is_object() &&
               config["eos_token"].contains("content")) {
      tmpl.eos_token_ = config["eos_token"]["content"].get<std::string>();
    }
  }

  tmpl.available_ = true;
  return tmpl;
}

std::string ChatTemplate::apply(const std::vector<ChatMessage> &messages,
                                bool add_generation_prompt) const {
  if (!available_)
    return "";

  // Build context
  json context;
  json msgs = json::array();
  for (const auto &msg : messages) {
    json m;
    m["role"] = msg.role;
    m["content"] = msg.content;
    msgs.push_back(m);
  }
  context["messages"] = msgs;
  context["bos_token"] = bos_token_;
  context["eos_token"] = eos_token_;
  context["add_generation_prompt"] = add_generation_prompt;

  return render(template_str_, context);
}

std::string ChatTemplate::apply(const std::string &user_input,
                                bool add_generation_prompt) const {
  std::vector<ChatMessage> messages = {{"user", user_input}};
  return apply(messages, add_generation_prompt);
}

bool ChatTemplate::isAvailable() const { return available_; }

std::string ChatTemplate::getBosToken() const { return bos_token_; }

std::string ChatTemplate::getEosToken() const { return eos_token_; }

std::string ChatTemplate::render(const std::string &tmpl,
                                 const json &context) const {
  try {
    Lexer lexer(tmpl);
    auto tokens = lexer.tokenize();

    Parser parser(tokens);
    auto ast = parser.parse();

    Evaluator evaluator(context);
    return evaluator.evaluate(ast);
  } catch (const std::exception &e) {
    std::cerr << "[ChatTemplate] Render error: " << e.what() << std::endl;
    return "";
  }
}

} // namespace causallm

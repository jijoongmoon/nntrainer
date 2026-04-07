// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   evaluator.cpp
 * @date   06 April 2026
 * @brief  Jinja2 template evaluator implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include "evaluator.h"
#include <algorithm>
#include <stdexcept>

namespace causallm {
namespace jinja {

// ============================================================================
// Context
// ============================================================================

Context::Context() { scopes_.push_back({}); }

Context::Context(const std::map<std::string, Value> &vars) {
  scopes_.push_back(vars);
}

void Context::set(const std::string &name, const Value &val) {
  // Set in the innermost scope
  scopes_.back()[name] = val;
}

void Context::set_in_defining_scope(const std::string &name,
                                    const Value &val) {
  // Find the scope where this variable is defined and update it there
  for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
    auto found = it->find(name);
    if (found != it->end()) {
      found->second = val;
      return;
    }
  }
  // Not found anywhere, set in current scope
  scopes_.back()[name] = val;
}

Value Context::get(const std::string &name) const {
  // Search from innermost scope outward
  for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
    auto found = it->find(name);
    if (found != it->end())
      return found->second;
  }
  return Value(); // None / undefined
}

bool Context::has(const std::string &name) const {
  for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
    if (it->count(name))
      return true;
  }
  return false;
}

void Context::push_scope() { scopes_.push_back({}); }

void Context::pop_scope() {
  if (scopes_.size() > 1)
    scopes_.pop_back();
}

// ============================================================================
// Evaluator
// ============================================================================

Evaluator::Evaluator(Context &ctx) : ctx_(ctx) {}

std::string Evaluator::render(const std::vector<ASTNodePtr> &nodes) {
  std::ostringstream out;
  for (const auto &node : nodes)
    render_node(node, out);
  return out.str();
}

Value Evaluator::eval_expr(const ASTNodePtr &node) {
  if (!node)
    return Value();

  switch (node->type) {
  case NodeType::Literal:
    return node->literal_val;
  case NodeType::Variable:
    return ctx_.get(node->str_val);
  case NodeType::BinaryOp:
    return eval_binary(node);
  case NodeType::UnaryOp:
    return eval_unary(node);
  case NodeType::GetAttr:
    return eval_getattr(node);
  case NodeType::GetItem:
    return eval_getitem(node);
  case NodeType::Filter:
    return eval_filter(node);
  case NodeType::Call:
    return eval_call(node);
  case NodeType::MethodCall:
    return eval_method_call(node);
  case NodeType::Array: {
    Value::Array arr;
    for (const auto &child : node->children)
      arr.push_back(eval_expr(child));
    return Value(arr);
  }
  case NodeType::Conditional: {
    Value cond = eval_expr(node->condition);
    return cond.truthy() ? eval_expr(node->left) : eval_expr(node->right);
  }
  default:
    throw std::runtime_error("Cannot evaluate node type " +
                             std::to_string(static_cast<int>(node->type)));
  }
}

void Evaluator::render_node(const ASTNodePtr &node, std::ostringstream &out) {
  switch (node->type) {
  case NodeType::Text:
    out << node->str_val;
    break;
  case NodeType::Output: {
    Value val = eval_expr(node->left);
    out << val.to_string();
    break;
  }
  case NodeType::If:
    render_if(node, out);
    break;
  case NodeType::For:
    render_for(node, out);
    break;
  case NodeType::Set:
    render_set(node);
    break;
  case NodeType::Macro:
    render_macro(node);
    break;
  default:
    throw std::runtime_error("Cannot render node type " +
                             std::to_string(static_cast<int>(node->type)));
  }
}

void Evaluator::render_if(const ASTNodePtr &node, std::ostringstream &out) {
  Value cond = eval_expr(node->condition);
  if (cond.truthy()) {
    for (const auto &child : node->children)
      render_node(child, out);
    return;
  }

  // Check elif branches
  for (const auto &[elif_cond, elif_body] : node->elif_branches) {
    Value elif_val = eval_expr(elif_cond);
    if (elif_val.truthy()) {
      for (const auto &child : elif_body)
        render_node(child, out);
      return;
    }
  }

  // Else
  for (const auto &child : node->else_children)
    render_node(child, out);
}

void Evaluator::render_for(const ASTNodePtr &node, std::ostringstream &out) {
  Value iterable = eval_expr(node->left);
  ctx_.push_scope();

  if (iterable.is_array()) {
    const auto &arr = iterable.as_array();
    if (arr.empty() && !node->else_children.empty()) {
      for (const auto &child : node->else_children)
        render_node(child, out);
      ctx_.pop_scope();
      return;
    }

    for (size_t i = 0; i < arr.size(); i++) {
      // Support tuple unpacking: {% for k, v in items %}
      if (!node->loop_var2.empty() && arr[i].is_array() &&
          arr[i].size() >= 2) {
        ctx_.set(node->loop_var, arr[i].get(size_t(0)));
        ctx_.set(node->loop_var2, arr[i].get(size_t(1)));
      } else {
        ctx_.set(node->loop_var, arr[i]);
      }

      // Set loop variable
      Value::Object loop;
      loop["index"] = Value(static_cast<int64_t>(i + 1));
      loop["index0"] = Value(static_cast<int64_t>(i));
      loop["first"] = Value(i == 0);
      loop["last"] = Value(i == arr.size() - 1);
      loop["length"] = Value(static_cast<int64_t>(arr.size()));
      loop["revindex"] = Value(static_cast<int64_t>(arr.size() - i));
      loop["revindex0"] = Value(static_cast<int64_t>(arr.size() - i - 1));
      ctx_.set("loop", Value(loop));

      // Apply filter condition if present
      if (node->condition) {
        Value filter_result = eval_expr(node->condition);
        if (!filter_result.truthy())
          continue;
      }

      for (const auto &child : node->children)
        render_node(child, out);
    }
  } else if (iterable.is_object()) {
    const auto &obj = iterable.as_object();
    if (obj.empty() && !node->else_children.empty()) {
      for (const auto &child : node->else_children)
        render_node(child, out);
      ctx_.pop_scope();
      return;
    }

    size_t i = 0;
    size_t total = obj.size();
    for (const auto &[key, val] : obj) {
      if (node->loop_var2.empty()) {
        // Iterating over keys only
        ctx_.set(node->loop_var, Value(key));
      } else {
        // key, value unpacking
        ctx_.set(node->loop_var, Value(key));
        ctx_.set(node->loop_var2, val);
      }

      Value::Object loop;
      loop["index"] = Value(static_cast<int64_t>(i + 1));
      loop["index0"] = Value(static_cast<int64_t>(i));
      loop["first"] = Value(i == 0);
      loop["last"] = Value(i == total - 1);
      loop["length"] = Value(static_cast<int64_t>(total));
      ctx_.set("loop", Value(loop));

      for (const auto &child : node->children)
        render_node(child, out);

      i++;
    }
  } else if (iterable.is_string()) {
    // Iterate over characters
    const auto &s = iterable.as_string();
    for (size_t i = 0; i < s.size(); i++) {
      ctx_.set(node->loop_var, Value(std::string(1, s[i])));

      Value::Object loop;
      loop["index"] = Value(static_cast<int64_t>(i + 1));
      loop["index0"] = Value(static_cast<int64_t>(i));
      loop["first"] = Value(i == 0);
      loop["last"] = Value(i == s.size() - 1);
      loop["length"] = Value(static_cast<int64_t>(s.size()));
      ctx_.set("loop", Value(loop));

      for (const auto &child : node->children)
        render_node(child, out);
    }
  }

  ctx_.pop_scope();
}

void Evaluator::render_set(const ASTNodePtr &node) {
  Value val = eval_expr(node->left);

  // Check if it's a namespace attribute set: "ns.attr"
  size_t dot = node->str_val.find('.');
  if (dot != std::string::npos) {
    std::string ns_name = node->str_val.substr(0, dot);
    std::string attr = node->str_val.substr(dot + 1);
    Value ns = ctx_.get(ns_name);
    if (ns.is_object()) {
      auto obj = ns.as_object();
      obj[attr] = val;
      // Update in all scopes where ns exists (namespace spans scopes)
      ctx_.set_in_defining_scope(ns_name, Value(obj));
    }
  } else {
    ctx_.set(node->str_val, val);
  }
}

void Evaluator::render_macro(const ASTNodePtr &node) {
  macros_[node->str_val] = node;
}

// ============================================================================
// Expression evaluation
// ============================================================================

Value Evaluator::eval_binary(const ASTNodePtr &node) {
  const std::string &op = node->str_val;

  // Short-circuit for logical operators
  if (op == "or") {
    Value left = eval_expr(node->left);
    return left.truthy() ? left : eval_expr(node->right);
  }
  if (op == "and") {
    Value left = eval_expr(node->left);
    return !left.truthy() ? left : eval_expr(node->right);
  }

  // 'is' tests
  if (op.substr(0, 2) == "is") {
    Value left = eval_expr(node->left);
    bool negated = op.find("not") != std::string::npos;
    std::string test = op.substr(op.find_last_of(' ') + 1);

    bool result = false;
    if (test == "defined")
      result = !left.is_none();
    else if (test == "undefined")
      result = left.is_none();
    else if (test == "none")
      result = left.is_none();
    else if (test == "true")
      result = left.is_bool() && left.as_bool();
    else if (test == "false")
      result = left.is_bool() && !left.as_bool();
    else if (test == "string")
      result = left.is_string();
    else if (test == "number")
      result = left.is_number();
    else if (test == "integer")
      result = left.is_int();
    else if (test == "sequence")
      result = left.is_array();
    else if (test == "mapping")
      result = left.is_object();
    else
      throw std::runtime_error("Unknown test: " + test);

    return Value(negated ? !result : result);
  }

  // 'in' operator
  if (op == "in") {
    Value left = eval_expr(node->left);
    Value right = eval_expr(node->right);
    if (right.is_array()) {
      for (const auto &item : right.as_array()) {
        if (left == item)
          return Value(true);
      }
      return Value(false);
    }
    if (right.is_string() && left.is_string()) {
      return Value(right.as_string().find(left.as_string()) !=
                   std::string::npos);
    }
    if (right.is_object() && left.is_string()) {
      return Value(right.contains(left.as_string()));
    }
    return Value(false);
  }
  if (op == "not in") {
    Value left = eval_expr(node->left);
    Value right = eval_expr(node->right);
    if (right.is_array()) {
      for (const auto &item : right.as_array()) {
        if (left == item)
          return Value(false);
      }
      return Value(true);
    }
    if (right.is_string() && left.is_string()) {
      return Value(right.as_string().find(left.as_string()) ==
                   std::string::npos);
    }
    if (right.is_object() && left.is_string()) {
      return Value(!right.contains(left.as_string()));
    }
    return Value(true);
  }

  Value left = eval_expr(node->left);
  Value right = eval_expr(node->right);

  if (op == "+")
    return left + right;
  if (op == "-")
    return left - right;
  if (op == "*")
    return left * right;
  if (op == "/")
    return left / right;
  if (op == "%")
    return left % right;
  if (op == "~")
    return Value(left.to_string() + right.to_string());
  if (op == "==")
    return Value(left == right);
  if (op == "!=")
    return Value(left != right);
  if (op == "<")
    return Value(left < right);
  if (op == ">")
    return Value(left > right);
  if (op == "<=")
    return Value(left <= right);
  if (op == ">=")
    return Value(left >= right);

  throw std::runtime_error("Unknown binary operator: " + op);
}

Value Evaluator::eval_unary(const ASTNodePtr &node) {
  Value val = eval_expr(node->left);
  if (node->str_val == "not")
    return Value(!val.truthy());
  if (node->str_val == "-") {
    if (val.is_int())
      return Value(-val.as_int());
    if (val.is_double())
      return Value(-val.as_double());
  }
  throw std::runtime_error("Unknown unary operator: " + node->str_val);
}

Value Evaluator::eval_getattr(const ASTNodePtr &node) {
  Value obj = eval_expr(node->left);
  const std::string &attr = node->str_val;

  // Object attribute
  if (obj.is_object()) {
    if (obj.contains(attr))
      return obj.get(attr);
  }

  // Built-in properties
  if (attr == "length" || attr == "size") {
    return Value(static_cast<int64_t>(obj.size()));
  }

  return Value(); // None for missing attributes
}

Value Evaluator::eval_getitem(const ASTNodePtr &node) {
  Value obj = eval_expr(node->left);

  if (node->str_val == "slice") {
    // Handle slice [start:end]
    Value start_val = eval_expr(node->right);
    int64_t start = start_val.is_int() ? start_val.as_int() : 0;

    if (obj.is_array()) {
      const auto &arr = obj.as_array();
      int64_t len = static_cast<int64_t>(arr.size());
      if (start < 0)
        start = std::max(int64_t(0), len + start);
      int64_t end = len;
      if (node->condition) {
        Value end_val = eval_expr(node->condition);
        if (end_val.is_int()) {
          end = end_val.as_int();
          if (end < 0)
            end = std::max(int64_t(0), len + end);
        }
      }
      Value::Array result;
      for (int64_t i = start; i < end && i < len; i++)
        result.push_back(arr[i]);
      return Value(result);
    }
    if (obj.is_string()) {
      const auto &s = obj.as_string();
      int64_t len = static_cast<int64_t>(s.size());
      if (start < 0)
        start = std::max(int64_t(0), len + start);
      int64_t end = len;
      if (node->condition) {
        Value end_val = eval_expr(node->condition);
        if (end_val.is_int()) {
          end = end_val.as_int();
          if (end < 0)
            end = std::max(int64_t(0), len + end);
        }
      }
      return Value(s.substr(start, end - start));
    }
  }

  Value key = eval_expr(node->right);
  return obj.get(key);
}

Value Evaluator::eval_filter(const ASTNodePtr &node) {
  Value val = eval_expr(node->left);

  std::vector<Value> args;
  for (const auto &arg : node->args)
    args.push_back(eval_expr(arg));

  std::vector<std::pair<std::string, Value>> kwargs;
  for (const auto &[name, expr] : node->kwargs)
    kwargs.push_back({name, eval_expr(expr)});

  return apply_filter(node->str_val, val, args, kwargs);
}

Value Evaluator::eval_call(const ASTNodePtr &node) {
  const std::string &name = node->str_val;

  // namespace() constructor
  if (name == "namespace") {
    Value::Object obj;
    for (const auto &[key, expr] : node->kwargs)
      obj[key] = eval_expr(expr);
    return Value(obj);
  }

  // range() builtin
  if (name == "range") {
    std::vector<Value> args;
    for (const auto &arg : node->args)
      args.push_back(eval_expr(arg));

    int64_t start = 0, stop = 0, step = 1;
    if (args.size() == 1) {
      stop = args[0].as_int();
    } else if (args.size() == 2) {
      start = args[0].as_int();
      stop = args[1].as_int();
    } else if (args.size() >= 3) {
      start = args[0].as_int();
      stop = args[1].as_int();
      step = args[2].as_int();
    }

    Value::Array result;
    if (step > 0) {
      for (int64_t i = start; i < stop; i += step)
        result.push_back(Value(i));
    } else if (step < 0) {
      for (int64_t i = start; i > stop; i += step)
        result.push_back(Value(i));
    }
    return Value(result);
  }

  // dict() builtin
  if (name == "dict") {
    Value::Object obj;
    for (const auto &[key, expr] : node->kwargs)
      obj[key] = eval_expr(expr);
    return Value(obj);
  }

  // Check for macros
  auto macro_it = macros_.find(name);
  if (macro_it != macros_.end()) {
    const auto &macro = macro_it->second;
    ctx_.push_scope();

    // Bind positional arguments
    for (size_t i = 0; i < node->args.size() && i < macro->macro_params.size();
         i++) {
      ctx_.set(macro->macro_params[i], eval_expr(node->args[i]));
    }

    std::ostringstream out;
    for (const auto &child : macro->children)
      render_node(child, out);

    ctx_.pop_scope();
    return Value(out.str());
  }

  // Check if it's a callable variable (e.g., raise_exception)
  Value func_val = ctx_.get(name);
  if (!func_val.is_none()) {
    // If it's a string, just return it
    return func_val;
  }

  throw std::runtime_error("Unknown function: " + name);
}

Value Evaluator::eval_method_call(const ASTNodePtr &node) {
  Value obj = eval_expr(node->left);
  const std::string &method = node->str_val;

  std::vector<Value> args;
  for (const auto &arg : node->args)
    args.push_back(eval_expr(arg));

  // String methods
  if (obj.is_string()) {
    const std::string &s = obj.as_string();

    if (method == "strip" || method == "trim") {
      std::string result = s;
      size_t start = result.find_first_not_of(" \t\n\r");
      size_t end = result.find_last_not_of(" \t\n\r");
      if (start == std::string::npos)
        return Value(std::string(""));
      return Value(result.substr(start, end - start + 1));
    }
    if (method == "lstrip") {
      size_t start = s.find_first_not_of(" \t\n\r");
      if (start == std::string::npos)
        return Value(std::string(""));
      return Value(s.substr(start));
    }
    if (method == "rstrip") {
      size_t end = s.find_last_not_of(" \t\n\r");
      if (end == std::string::npos)
        return Value(std::string(""));
      return Value(s.substr(0, end + 1));
    }
    if (method == "upper") {
      std::string result = s;
      std::transform(result.begin(), result.end(), result.begin(), ::toupper);
      return Value(result);
    }
    if (method == "lower") {
      std::string result = s;
      std::transform(result.begin(), result.end(), result.begin(), ::tolower);
      return Value(result);
    }
    if (method == "startswith") {
      if (args.empty())
        throw std::runtime_error("startswith requires an argument");
      return Value(s.find(args[0].as_string()) == 0);
    }
    if (method == "endswith") {
      if (args.empty())
        throw std::runtime_error("endswith requires an argument");
      const std::string &suffix = args[0].as_string();
      if (suffix.size() > s.size())
        return Value(false);
      return Value(s.compare(s.size() - suffix.size(), suffix.size(), suffix) ==
                   0);
    }
    if (method == "split") {
      std::string delim = " ";
      if (!args.empty())
        delim = args[0].as_string();
      Value::Array parts;
      size_t start = 0;
      size_t found;
      while ((found = s.find(delim, start)) != std::string::npos) {
        parts.push_back(Value(s.substr(start, found - start)));
        start = found + delim.size();
      }
      parts.push_back(Value(s.substr(start)));
      return Value(parts);
    }
    if (method == "replace") {
      if (args.size() < 2)
        throw std::runtime_error("replace requires 2 arguments");
      std::string result = s;
      const std::string &from = args[0].as_string();
      const std::string &to = args[1].as_string();
      size_t pos = 0;
      while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.size(), to);
        pos += to.size();
      }
      return Value(result);
    }
    if (method == "find") {
      if (args.empty())
        throw std::runtime_error("find requires an argument");
      size_t pos = s.find(args[0].as_string());
      return Value(pos == std::string::npos ? int64_t(-1)
                                            : static_cast<int64_t>(pos));
    }
    if (method == "count") {
      if (args.empty())
        throw std::runtime_error("count requires an argument");
      const std::string &sub = args[0].as_string();
      int64_t count = 0;
      size_t pos = 0;
      while ((pos = s.find(sub, pos)) != std::string::npos) {
        count++;
        pos += sub.size();
      }
      return Value(count);
    }
  }

  // Array methods
  if (obj.is_array()) {
    if (method == "append") {
      // Note: arrays are immutable in our Value, so return new array
      auto arr = obj.as_array();
      if (!args.empty())
        arr.push_back(args[0]);
      return Value(arr);
    }
  }

  // Object methods
  if (obj.is_object()) {
    if (method == "items") {
      Value::Array items;
      for (const auto &[k, v] : obj.as_object()) {
        Value::Array pair;
        pair.push_back(Value(k));
        pair.push_back(v);
        items.push_back(Value(pair));
      }
      return Value(items);
    }
    if (method == "keys") {
      Value::Array keys;
      for (const auto &[k, v] : obj.as_object())
        keys.push_back(Value(k));
      return Value(keys);
    }
    if (method == "values") {
      Value::Array values;
      for (const auto &[k, v] : obj.as_object())
        values.push_back(v);
      return Value(values);
    }
    if (method == "get") {
      if (args.empty())
        throw std::runtime_error("dict.get requires an argument");
      std::string key = args[0].as_string();
      Value default_val = args.size() > 1 ? args[1] : Value();
      if (obj.contains(key))
        return obj.get(key);
      return default_val;
    }
    if (method == "update") {
      auto result = obj.as_object();
      if (!args.empty() && args[0].is_object()) {
        for (const auto &[k, v] : args[0].as_object())
          result[k] = v;
      }
      return Value(result);
    }
  }

  throw std::runtime_error("Unknown method '" + method + "' on " +
                           obj.to_string());
}

// ============================================================================
// Built-in filters
// ============================================================================

Value Evaluator::apply_filter(
  const std::string &name, const Value &val,
  const std::vector<Value> &args,
  const std::vector<std::pair<std::string, Value>> &kwargs) {

  if (name == "trim" || name == "strip") {
    std::string s = val.to_string();
    size_t start = s.find_first_not_of(" \t\n\r");
    size_t end = s.find_last_not_of(" \t\n\r");
    if (start == std::string::npos)
      return Value(std::string(""));
    return Value(s.substr(start, end - start + 1));
  }

  if (name == "upper") {
    std::string s = val.to_string();
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return Value(s);
  }

  if (name == "lower") {
    std::string s = val.to_string();
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return Value(s);
  }

  if (name == "title") {
    std::string s = val.to_string();
    bool capitalize = true;
    for (auto &c : s) {
      if (std::isspace(c)) {
        capitalize = true;
      } else if (capitalize) {
        c = std::toupper(c);
        capitalize = false;
      } else {
        c = std::tolower(c);
      }
    }
    return Value(s);
  }

  if (name == "capitalize") {
    std::string s = val.to_string();
    if (!s.empty())
      s[0] = std::toupper(s[0]);
    return Value(s);
  }

  if (name == "length" || name == "count") {
    return Value(static_cast<int64_t>(val.size()));
  }

  if (name == "int" || name == "integer") {
    if (val.is_int())
      return val;
    if (val.is_double())
      return Value(static_cast<int64_t>(val.as_double()));
    if (val.is_string()) {
      try {
        return Value(static_cast<int64_t>(std::stoll(val.as_string())));
      } catch (...) {
        return Value(int64_t(0));
      }
    }
    return Value(int64_t(0));
  }

  if (name == "float") {
    if (val.is_double())
      return val;
    if (val.is_int())
      return Value(static_cast<double>(val.as_int()));
    if (val.is_string()) {
      try {
        return Value(std::stod(val.as_string()));
      } catch (...) {
        return Value(0.0);
      }
    }
    return Value(0.0);
  }

  if (name == "string") {
    return Value(val.to_string());
  }

  if (name == "default" || name == "d") {
    if (val.is_none() || (val.is_string() && val.as_string().empty())) {
      if (!args.empty())
        return args[0];
      return Value(std::string(""));
    }
    return val;
  }

  if (name == "first") {
    if (val.is_array() && val.size() > 0)
      return val.get(size_t(0));
    if (val.is_string() && val.size() > 0)
      return Value(std::string(1, val.as_string()[0]));
    return Value();
  }

  if (name == "last") {
    if (val.is_array() && val.size() > 0)
      return val.get(val.size() - 1);
    if (val.is_string() && val.size() > 0)
      return Value(std::string(1, val.as_string().back()));
    return Value();
  }

  if (name == "join") {
    if (!val.is_array())
      return val;
    std::string sep = "";
    if (!args.empty())
      sep = args[0].to_string();
    std::string result;
    const auto &arr = val.as_array();
    for (size_t i = 0; i < arr.size(); i++) {
      if (i > 0)
        result += sep;
      result += arr[i].to_string();
    }
    return Value(result);
  }

  if (name == "list") {
    if (val.is_array())
      return val;
    if (val.is_string()) {
      Value::Array arr;
      for (char c : val.as_string())
        arr.push_back(Value(std::string(1, c)));
      return Value(arr);
    }
    return Value(Value::Array{});
  }

  if (name == "reverse") {
    if (val.is_array()) {
      auto arr = val.as_array();
      std::reverse(arr.begin(), arr.end());
      return Value(arr);
    }
    if (val.is_string()) {
      std::string s = val.as_string();
      std::reverse(s.begin(), s.end());
      return Value(s);
    }
    return val;
  }

  if (name == "sort") {
    if (!val.is_array())
      return val;
    auto arr = val.as_array();
    std::sort(arr.begin(), arr.end(),
              [](const Value &a, const Value &b) { return a < b; });
    return Value(arr);
  }

  if (name == "map") {
    if (!val.is_array())
      return val;
    Value::Array result;
    if (!args.empty()) {
      // map(attribute=name) or map('attr_name')
      std::string attr;
      if (!args.empty())
        attr = args[0].as_string();
      for (const auto &item : val.as_array()) {
        if (item.is_object() && item.contains(attr))
          result.push_back(item.get(attr));
        else
          result.push_back(Value());
      }
    }
    // Check kwargs for attribute=
    for (const auto &[k, v] : kwargs) {
      if (k == "attribute") {
        std::string attr = v.as_string();
        for (const auto &item : val.as_array()) {
          if (item.is_object() && item.contains(attr))
            result.push_back(item.get(attr));
          else
            result.push_back(Value());
        }
        break;
      }
    }
    return Value(result);
  }

  if (name == "select" || name == "selectattr") {
    if (!val.is_array())
      return val;
    Value::Array result;
    for (const auto &item : val.as_array()) {
      if (item.truthy())
        result.push_back(item);
    }
    return Value(result);
  }

  if (name == "reject" || name == "rejectattr") {
    if (!val.is_array())
      return val;
    Value::Array result;
    for (const auto &item : val.as_array()) {
      if (!item.truthy())
        result.push_back(item);
    }
    return Value(result);
  }

  if (name == "replace") {
    if (!val.is_string() || args.size() < 2)
      return val;
    std::string s = val.as_string();
    const std::string &from = args[0].as_string();
    const std::string &to = args[1].as_string();
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
      s.replace(pos, from.size(), to);
      pos += to.size();
    }
    return Value(s);
  }

  if (name == "tojson") {
    // Simple JSON serialization
    if (val.is_string())
      return Value("\"" + val.as_string() + "\"");
    return Value(val.to_string());
  }

  if (name == "items") {
    if (!val.is_object())
      return val;
    Value::Array items;
    for (const auto &[k, v] : val.as_object()) {
      Value::Array pair;
      pair.push_back(Value(k));
      pair.push_back(v);
      items.push_back(Value(pair));
    }
    return Value(items);
  }

  if (name == "batch") {
    if (!val.is_array() || args.empty())
      return val;
    int64_t size = args[0].as_int();
    Value::Array result;
    const auto &arr = val.as_array();
    for (size_t i = 0; i < arr.size(); i += size) {
      Value::Array batch;
      for (size_t j = i; j < i + size && j < arr.size(); j++)
        batch.push_back(arr[j]);
      result.push_back(Value(batch));
    }
    return Value(result);
  }

  if (name == "indent") {
    std::string s = val.to_string();
    int64_t width = 4;
    if (!args.empty())
      width = args[0].as_int();
    bool first_line = false;
    for (const auto &[k, v] : kwargs) {
      if (k == "first" && v.truthy())
        first_line = true;
    }

    std::string indent_str(width, ' ');
    std::string result;
    bool is_first = true;
    size_t start = 0;
    size_t found;
    while ((found = s.find('\n', start)) != std::string::npos) {
      if (is_first && !first_line) {
        result += s.substr(start, found - start + 1);
      } else {
        result += indent_str + s.substr(start, found - start + 1);
      }
      start = found + 1;
      is_first = false;
    }
    if (start < s.size()) {
      if (is_first && !first_line)
        result += s.substr(start);
      else
        result += indent_str + s.substr(start);
    }
    return Value(result);
  }

  throw std::runtime_error("Unknown filter: " + name);
}

} // namespace jinja
} // namespace causallm

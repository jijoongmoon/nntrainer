// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   value.h
 * @date   06 April 2026
 * @brief  Dynamic value type for the Jinja2 template engine
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#ifndef __JINJA_VALUE_H__
#define __JINJA_VALUE_H__

#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace causallm {
namespace jinja {

class Value {
public:
  using Array = std::vector<Value>;
  using Object = std::map<std::string, Value>;
  using None = std::monostate;

  Value() : data_(None{}) {}
  Value(std::nullptr_t) : data_(None{}) {}
  Value(bool b) : data_(b) {}
  Value(int i) : data_(static_cast<int64_t>(i)) {}
  Value(int64_t i) : data_(i) {}
  Value(size_t i) : data_(static_cast<int64_t>(i)) {}
  Value(double d) : data_(d) {}
  Value(const std::string &s) : data_(s) {}
  Value(const char *s) : data_(std::string(s)) {}
  Value(const Array &a) : data_(a) {}
  Value(const Object &o) : data_(o) {}

  bool is_none() const { return std::holds_alternative<None>(data_); }
  bool is_bool() const { return std::holds_alternative<bool>(data_); }
  bool is_int() const { return std::holds_alternative<int64_t>(data_); }
  bool is_double() const { return std::holds_alternative<double>(data_); }
  bool is_string() const { return std::holds_alternative<std::string>(data_); }
  bool is_array() const { return std::holds_alternative<Array>(data_); }
  bool is_object() const { return std::holds_alternative<Object>(data_); }
  bool is_number() const { return is_int() || is_double(); }
  bool is_undefined() const { return is_none(); }

  bool as_bool() const;
  int64_t as_int() const;
  double as_double() const;
  const std::string &as_string() const;
  const Array &as_array() const;
  const Object &as_object() const;

  bool truthy() const;
  std::string to_string() const;

  Value operator+(const Value &rhs) const;
  Value operator-(const Value &rhs) const;
  Value operator*(const Value &rhs) const;
  Value operator/(const Value &rhs) const;
  Value operator%(const Value &rhs) const;
  bool operator==(const Value &rhs) const;
  bool operator!=(const Value &rhs) const;
  bool operator<(const Value &rhs) const;
  bool operator<=(const Value &rhs) const;
  bool operator>(const Value &rhs) const;
  bool operator>=(const Value &rhs) const;

  Value get(const std::string &key) const;
  Value get(size_t index) const;
  Value get(const Value &key) const;

  size_t size() const;
  bool contains(const std::string &key) const;

private:
  std::variant<None, bool, int64_t, double, std::string, Array, Object> data_;
};

} // namespace jinja
} // namespace causallm

#endif // __JINJA_VALUE_H__

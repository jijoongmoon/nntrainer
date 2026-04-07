// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd.
 *
 * @file   value.cpp
 * @date   06 April 2026
 * @brief  Dynamic value type implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include "value.h"
#include <stdexcept>

namespace causallm {
namespace jinja {

bool Value::as_bool() const {
  if (is_bool())
    return std::get<bool>(data_);
  throw std::runtime_error("Value is not a bool");
}

int64_t Value::as_int() const {
  if (is_int())
    return std::get<int64_t>(data_);
  if (is_double())
    return static_cast<int64_t>(std::get<double>(data_));
  throw std::runtime_error("Value is not an int");
}

double Value::as_double() const {
  if (is_double())
    return std::get<double>(data_);
  if (is_int())
    return static_cast<double>(std::get<int64_t>(data_));
  throw std::runtime_error("Value is not a double");
}

const std::string &Value::as_string() const {
  if (is_string())
    return std::get<std::string>(data_);
  throw std::runtime_error("Value is not a string");
}

const Value::Array &Value::as_array() const {
  if (is_array())
    return std::get<Array>(data_);
  throw std::runtime_error("Value is not an array");
}

const Value::Object &Value::as_object() const {
  if (is_object())
    return std::get<Object>(data_);
  throw std::runtime_error("Value is not an object");
}

bool Value::truthy() const {
  if (is_none())
    return false;
  if (is_bool())
    return std::get<bool>(data_);
  if (is_int())
    return std::get<int64_t>(data_) != 0;
  if (is_double())
    return std::get<double>(data_) != 0.0;
  if (is_string())
    return !std::get<std::string>(data_).empty();
  if (is_array())
    return !std::get<Array>(data_).empty();
  if (is_object())
    return !std::get<Object>(data_).empty();
  return false;
}

std::string Value::to_string() const {
  if (is_none())
    return "";
  if (is_bool())
    return std::get<bool>(data_) ? "True" : "False";
  if (is_int())
    return std::to_string(std::get<int64_t>(data_));
  if (is_double()) {
    std::string s = std::to_string(std::get<double>(data_));
    // Remove trailing zeros
    size_t dot = s.find('.');
    if (dot != std::string::npos) {
      size_t last = s.find_last_not_of('0');
      if (last == dot)
        last++;
      s = s.substr(0, last + 1);
    }
    return s;
  }
  if (is_string())
    return std::get<std::string>(data_);
  if (is_array()) {
    std::string result = "[";
    const auto &arr = std::get<Array>(data_);
    for (size_t i = 0; i < arr.size(); i++) {
      if (i > 0)
        result += ", ";
      if (arr[i].is_string())
        result += "'" + arr[i].as_string() + "'";
      else
        result += arr[i].to_string();
    }
    return result + "]";
  }
  if (is_object()) {
    std::string result = "{";
    bool first = true;
    for (const auto &[k, v] : std::get<Object>(data_)) {
      if (!first)
        result += ", ";
      result += "'" + k + "': ";
      if (v.is_string())
        result += "'" + v.as_string() + "'";
      else
        result += v.to_string();
      first = false;
    }
    return result + "}";
  }
  return "";
}

Value Value::operator+(const Value &rhs) const {
  if (is_string() || rhs.is_string())
    return Value(to_string() + rhs.to_string());
  if (is_int() && rhs.is_int())
    return Value(as_int() + rhs.as_int());
  if (is_number() && rhs.is_number())
    return Value(as_double() + rhs.as_double());
  if (is_array() && rhs.is_array()) {
    Array result = as_array();
    const auto &r = rhs.as_array();
    result.insert(result.end(), r.begin(), r.end());
    return Value(result);
  }
  throw std::runtime_error("Cannot add " + to_string() + " and " +
                           rhs.to_string());
}

Value Value::operator-(const Value &rhs) const {
  if (is_int() && rhs.is_int())
    return Value(as_int() - rhs.as_int());
  if (is_number() && rhs.is_number())
    return Value(as_double() - rhs.as_double());
  throw std::runtime_error("Cannot subtract these types");
}

Value Value::operator*(const Value &rhs) const {
  if (is_int() && rhs.is_int())
    return Value(as_int() * rhs.as_int());
  if (is_number() && rhs.is_number())
    return Value(as_double() * rhs.as_double());
  // string * int (Python-like repeat)
  if (is_string() && rhs.is_int()) {
    std::string result;
    int64_t count = rhs.as_int();
    for (int64_t i = 0; i < count; i++)
      result += as_string();
    return Value(result);
  }
  throw std::runtime_error("Cannot multiply these types");
}

Value Value::operator/(const Value &rhs) const {
  if (is_number() && rhs.is_number())
    return Value(as_double() / rhs.as_double());
  throw std::runtime_error("Cannot divide these types");
}

Value Value::operator%(const Value &rhs) const {
  if (is_int() && rhs.is_int())
    return Value(as_int() % rhs.as_int());
  throw std::runtime_error("Cannot modulo these types");
}

bool Value::operator==(const Value &rhs) const {
  if (is_none() && rhs.is_none())
    return true;
  if (is_none() || rhs.is_none())
    return false;
  if (is_bool() && rhs.is_bool())
    return as_bool() == rhs.as_bool();
  if (is_number() && rhs.is_number()) {
    if (is_int() && rhs.is_int())
      return as_int() == rhs.as_int();
    return as_double() == rhs.as_double();
  }
  if (is_string() && rhs.is_string())
    return as_string() == rhs.as_string();
  return false;
}

bool Value::operator!=(const Value &rhs) const { return !(*this == rhs); }

bool Value::operator<(const Value &rhs) const {
  if (is_number() && rhs.is_number()) {
    if (is_int() && rhs.is_int())
      return as_int() < rhs.as_int();
    return as_double() < rhs.as_double();
  }
  if (is_string() && rhs.is_string())
    return as_string() < rhs.as_string();
  throw std::runtime_error("Cannot compare these types");
}

bool Value::operator<=(const Value &rhs) const {
  return *this < rhs || *this == rhs;
}

bool Value::operator>(const Value &rhs) const { return rhs < *this; }

bool Value::operator>=(const Value &rhs) const { return rhs <= *this; }

Value Value::get(const std::string &key) const {
  if (is_object()) {
    const auto &obj = std::get<Object>(data_);
    auto it = obj.find(key);
    if (it != obj.end())
      return it->second;
    return Value(); // None
  }
  throw std::runtime_error("Cannot get key '" + key + "' from non-object");
}

Value Value::get(size_t index) const {
  if (is_array()) {
    const auto &arr = std::get<Array>(data_);
    if (index < arr.size())
      return arr[index];
    throw std::runtime_error("Array index out of range");
  }
  if (is_string()) {
    const auto &s = std::get<std::string>(data_);
    if (index < s.size())
      return Value(std::string(1, s[index]));
    throw std::runtime_error("String index out of range");
  }
  throw std::runtime_error("Cannot index into this type");
}

Value Value::get(const Value &key) const {
  if (key.is_int())
    return get(static_cast<size_t>(key.as_int()));
  if (key.is_string())
    return get(key.as_string());
  throw std::runtime_error("Invalid key type for indexing");
}

size_t Value::size() const {
  if (is_array())
    return std::get<Array>(data_).size();
  if (is_string())
    return std::get<std::string>(data_).size();
  if (is_object())
    return std::get<Object>(data_).size();
  throw std::runtime_error("Cannot get size of this type");
}

bool Value::contains(const std::string &key) const {
  if (is_object())
    return std::get<Object>(data_).count(key) > 0;
  return false;
}

} // namespace jinja
} // namespace causallm

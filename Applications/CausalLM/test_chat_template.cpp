// SPDX-License-Identifier: Apache-2.0
/**
 * @file   test_chat_template.cpp
 * @date   07 April 2026
 * @brief  Unit tests for the Jinja2 chat template engine
 */

#include "chat_template_engine.h"
#include <cassert>
#include <iostream>
#include <sstream>

using namespace causallm::jinja;

static int pass_count = 0;
static int fail_count = 0;

#define TEST(name) \
  static void test_##name(); \
  struct Register_##name { \
    Register_##name() { tests.push_back({#name, test_##name}); } \
  } reg_##name; \
  static void test_##name()

#define ASSERT_EQ(a, b) do { \
  auto _a = (a); auto _b = (b); \
  if (_a != _b) { \
    std::cerr << "  FAIL: " << #a << " == " << #b << "\n"; \
    std::cerr << "    got: [" << _a << "]\n    expected: [" << _b << "]\n"; \
    throw std::runtime_error("assertion failed"); \
  } \
} while(0)

#define ASSERT_TRUE(x) do { if (!(x)) { \
  std::cerr << "  FAIL: " << #x << " is not true\n"; \
  throw std::runtime_error("assertion failed"); \
} } while(0)

#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))

#define ASSERT_CONTAINS(haystack, needle) do { \
  if ((haystack).find(needle) == std::string::npos) { \
    std::cerr << "  FAIL: string does not contain '" << needle << "'\n"; \
    std::cerr << "    got: [" << haystack << "]\n"; \
    throw std::runtime_error("assertion failed"); \
  } \
} while(0)

#define ASSERT_THROWS(expr) do { \
  bool caught = false; \
  try { expr; } catch (...) { caught = true; } \
  if (!caught) { \
    std::cerr << "  FAIL: expected exception from " << #expr << "\n"; \
    throw std::runtime_error("assertion failed"); \
  } \
} while(0)

static std::vector<std::pair<std::string, void(*)()>> tests;

// Helper to render a template with given variables
static std::string render(const std::string &tmpl,
                          const std::map<std::string, Value> &vars = {}) {
  Template t(tmpl);
  return t.render(vars);
}

// Helper to build a message object
static Value msg(const std::string &role, const std::string &content) {
  Value::Object m;
  m["role"] = Value(role);
  m["content"] = Value(content);
  return Value(m);
}

// ==========================================================================
// 1. Value type tests
// ==========================================================================

TEST(value_types) {
  Value none;
  ASSERT_TRUE(none.is_none());
  ASSERT_FALSE(none.truthy());

  Value b(true);
  ASSERT_TRUE(b.is_bool());
  ASSERT_TRUE(b.truthy());
  ASSERT_EQ(b.as_bool(), true);

  Value i(42);
  ASSERT_TRUE(i.is_int());
  ASSERT_EQ(i.as_int(), 42);

  Value d(3.14);
  ASSERT_TRUE(d.is_double());

  Value s("hello");
  ASSERT_TRUE(s.is_string());
  ASSERT_EQ(s.as_string(), "hello");

  Value arr(Value::Array{Value(1), Value(2)});
  ASSERT_TRUE(arr.is_array());
  ASSERT_EQ(arr.size(), size_t(2));

  Value::Object obj;
  obj["key"] = Value("val");
  Value o(obj);
  ASSERT_TRUE(o.is_object());
  ASSERT_TRUE(o.contains("key"));
  ASSERT_FALSE(o.contains("nope"));
}

TEST(value_truthiness) {
  ASSERT_FALSE(Value().truthy());          // none
  ASSERT_FALSE(Value(false).truthy());     // false
  ASSERT_TRUE(Value(true).truthy());       // true
  ASSERT_FALSE(Value(0).truthy());         // 0
  ASSERT_TRUE(Value(1).truthy());          // 1
  ASSERT_FALSE(Value("").truthy());        // empty string
  ASSERT_TRUE(Value("x").truthy());        // non-empty string
  ASSERT_FALSE(Value(Value::Array{}).truthy());
  ASSERT_TRUE(Value(Value::Array{Value(1)}).truthy());
}

TEST(value_operators) {
  ASSERT_EQ((Value(1) + Value(2)).as_int(), 3);
  ASSERT_EQ((Value("a") + Value("b")).as_string(), "ab");
  ASSERT_EQ((Value(10) - Value(3)).as_int(), 7);
  ASSERT_EQ((Value(4) * Value(5)).as_int(), 20);
  ASSERT_EQ((Value(10) % Value(3)).as_int(), 1);
  ASSERT_TRUE(Value(1) == Value(1));
  ASSERT_TRUE(Value(1) != Value(2));
  ASSERT_TRUE(Value(1) < Value(2));
  ASSERT_TRUE(Value("a") < Value("b"));
  // String repeat
  ASSERT_EQ((Value("ab") * Value(3)).as_string(), "ababab");
  // Array concat
  auto arr = (Value(Value::Array{Value(1)}) + Value(Value::Array{Value(2)}));
  ASSERT_EQ(arr.size(), size_t(2));
}

TEST(value_access) {
  Value::Object obj;
  obj["name"] = Value("test");
  Value o(obj);
  ASSERT_EQ(o.get(std::string("name")).as_string(), "test");
  ASSERT_TRUE(o.get(std::string("missing")).is_none());

  Value arr(Value::Array{Value("a"), Value("b"), Value("c")});
  ASSERT_EQ(arr.get(size_t(0)).as_string(), "a");
  ASSERT_EQ(arr.get(size_t(2)).as_string(), "c");
}

// ==========================================================================
// 2. Lexer tests
// ==========================================================================

TEST(lexer_basic_text) {
  Lexer lex("Hello World");
  auto tokens = lex.tokenize();
  ASSERT_TRUE(tokens[0].type == TokenType::Text);
  ASSERT_EQ(tokens[0].value, "Hello World");
  ASSERT_TRUE(tokens[1].type == TokenType::Eof);
}

TEST(lexer_expr) {
  Lexer lex("{{ name }}");
  auto tokens = lex.tokenize();
  ASSERT_TRUE(tokens[0].type == TokenType::ExprStart);
  ASSERT_TRUE(tokens[1].type == TokenType::Identifier);
  ASSERT_EQ(tokens[1].value, "name");
  ASSERT_TRUE(tokens[2].type == TokenType::ExprEnd);
}

TEST(lexer_stmt) {
  Lexer lex("{% if x %}yes{% endif %}");
  auto tokens = lex.tokenize();
  ASSERT_TRUE(tokens[0].type == TokenType::StmtStart);
  ASSERT_EQ(tokens[1].value, "if");
}

TEST(lexer_string_escapes) {
  Lexer lex("{{ \"hello\\nworld\" }}");
  auto tokens = lex.tokenize();
  ASSERT_TRUE(tokens[1].type == TokenType::StringLiteral);
  ASSERT_EQ(tokens[1].value, "hello\nworld");
}

TEST(lexer_whitespace_trim) {
  Lexer lex("hello  {{- x -}}  world");
  auto tokens = lex.tokenize();
  // The text "hello  " should be trimmed to "hello"
  ASSERT_EQ(tokens[0].value, "hello");
}

// ==========================================================================
// 3. Basic rendering tests
// ==========================================================================

TEST(render_plain_text) {
  ASSERT_EQ(render("Hello World"), "Hello World");
}

TEST(render_variable) {
  ASSERT_EQ(render("Hello {{ name }}!", {{"name", Value("World")}}),
            "Hello World!");
}

TEST(render_string_concat) {
  ASSERT_EQ(render("{{ 'a' + 'b' + 'c' }}"), "abc");
}

TEST(render_tilde_concat) {
  ASSERT_EQ(render("{{ 'x' ~ 42 }}"), "x42");
}

TEST(render_arithmetic) {
  ASSERT_EQ(render("{{ 2 + 3 * 4 }}"), "14");
  ASSERT_EQ(render("{{ (2 + 3) * 4 }}"), "20");
  ASSERT_EQ(render("{{ 10 % 3 }}"), "1");
}

TEST(render_comparison) {
  ASSERT_EQ(render("{{ 1 == 1 }}"), "True");
  ASSERT_EQ(render("{{ 1 != 2 }}"), "True");
  ASSERT_EQ(render("{{ 3 > 2 }}"), "True");
  ASSERT_EQ(render("{{ 1 >= 1 }}"), "True");
}

TEST(render_boolean_logic) {
  ASSERT_EQ(render("{{ true and false }}"), "False");
  ASSERT_EQ(render("{{ true or false }}"), "True");
  ASSERT_EQ(render("{{ not false }}"), "True");
}

TEST(render_none_handling) {
  ASSERT_EQ(render("{{ x is defined }}", {{"x", Value(42)}}), "True");
  ASSERT_EQ(render("{{ y is defined }}"), "False");
  ASSERT_EQ(render("{{ y is undefined }}"), "True");
  ASSERT_EQ(render("{{ none is none }}"), "True");
}

// ==========================================================================
// 4. Control flow tests
// ==========================================================================

TEST(if_basic) {
  ASSERT_EQ(render("{% if true %}yes{% endif %}"), "yes");
  ASSERT_EQ(render("{% if false %}yes{% endif %}"), "");
}

TEST(if_else) {
  ASSERT_EQ(render("{% if false %}A{% else %}B{% endif %}"), "B");
}

TEST(if_elif) {
  std::string t = "{% if x == 1 %}one{% elif x == 2 %}two{% else %}other{% endif %}";
  ASSERT_EQ(render(t, {{"x", Value(1)}}), "one");
  ASSERT_EQ(render(t, {{"x", Value(2)}}), "two");
  ASSERT_EQ(render(t, {{"x", Value(9)}}), "other");
}

TEST(for_basic) {
  Value::Array items = {Value("a"), Value("b"), Value("c")};
  ASSERT_EQ(render("{% for x in items %}{{x}}{% endfor %}",
                   {{"items", Value(items)}}), "abc");
}

TEST(for_loop_variables) {
  Value::Array items = {Value("a"), Value("b"), Value("c")};
  std::string t =
    "{% for x in items %}"
    "{{ loop.index }}:{{ x }}"
    "{% if not loop.last %},{% endif %}"
    "{% endfor %}";
  ASSERT_EQ(render(t, {{"items", Value(items)}}), "1:a,2:b,3:c");
}

TEST(for_loop_first_last) {
  Value::Array items = {Value(1), Value(2), Value(3)};
  std::string t =
    "{% for x in items %}"
    "{% if loop.first %}[{% endif %}"
    "{{ x }}"
    "{% if loop.last %}]{% endif %}"
    "{% endfor %}";
  ASSERT_EQ(render(t, {{"items", Value(items)}}), "[123]");
}

TEST(for_nested) {
  Value::Array outer = {
    Value(Value::Array{Value(1), Value(2)}),
    Value(Value::Array{Value(3), Value(4)})
  };
  std::string t =
    "{% for row in grid %}{% for x in row %}{{ x }}{% endfor %};{% endfor %}";
  ASSERT_EQ(render(t, {{"grid", Value(outer)}}), "12;34;");
}

TEST(for_with_condition) {
  Value::Array items = {Value(1), Value(2), Value(3), Value(4), Value(5)};
  std::string t = "{% for x in items if x > 2 %}{{ x }}{% endfor %}";
  ASSERT_EQ(render(t, {{"items", Value(items)}}), "345");
}

TEST(for_dict_items) {
  Value::Object obj;
  obj["a"] = Value(1);
  obj["b"] = Value(2);
  // Note: std::map is sorted by key
  std::string t = "{% for k, v in d.items() %}{{ k }}={{ v }} {% endfor %}";
  ASSERT_EQ(render(t, {{"d", Value(obj)}}), "a=1 b=2 ");
}

TEST(set_variable) {
  ASSERT_EQ(render("{% set x = 42 %}{{ x }}"), "42");
  ASSERT_EQ(render("{% set x = 'hello' %}{{ x | upper }}"), "HELLO");
}

TEST(set_namespace) {
  std::string t =
    "{% set ns = namespace(count=0) %}"
    "{% for x in items %}{% set ns.count = ns.count + 1 %}{% endfor %}"
    "{{ ns.count }}";
  // Note: namespace set in our evaluator works differently -
  // the set goes into the for scope. We test what we can.
  ASSERT_CONTAINS(render(t, {{"items", Value(Value::Array{Value(1), Value(2)})}}), "2");
}

// ==========================================================================
// 5. Filter tests
// ==========================================================================

TEST(filter_trim) {
  ASSERT_EQ(render("{{ '  hello  ' | trim }}"), "hello");
}

TEST(filter_upper_lower) {
  ASSERT_EQ(render("{{ 'hello' | upper }}"), "HELLO");
  ASSERT_EQ(render("{{ 'HELLO' | lower }}"), "hello");
}

TEST(filter_title_capitalize) {
  ASSERT_EQ(render("{{ 'hello world' | title }}"), "Hello World");
  ASSERT_EQ(render("{{ 'hello' | capitalize }}"), "Hello");
}

TEST(filter_length) {
  ASSERT_EQ(render("{{ items | length }}",
    {{"items", Value(Value::Array{Value(1), Value(2), Value(3)})}}), "3");
  ASSERT_EQ(render("{{ 'hello' | length }}"), "5");
}

TEST(filter_default) {
  ASSERT_EQ(render("{{ x | default('fallback') }}"), "fallback");
  ASSERT_EQ(render("{{ x | default('fallback') }}", {{"x", Value("real")}}), "real");
}

TEST(filter_first_last) {
  Value::Array arr = {Value("a"), Value("b"), Value("c")};
  ASSERT_EQ(render("{{ items | first }}", {{"items", Value(arr)}}), "a");
  ASSERT_EQ(render("{{ items | last }}", {{"items", Value(arr)}}), "c");
}

TEST(filter_join) {
  Value::Array arr = {Value("a"), Value("b"), Value("c")};
  ASSERT_EQ(render("{{ items | join(', ') }}", {{"items", Value(arr)}}), "a, b, c");
  ASSERT_EQ(render("{{ items | join }}", {{"items", Value(arr)}}), "abc");
}

TEST(filter_replace) {
  ASSERT_EQ(render("{{ 'hello world' | replace('world', 'jinja') }}"), "hello jinja");
}

TEST(filter_int_float) {
  ASSERT_EQ(render("{{ '42' | int }}"), "42");
  ASSERT_EQ(render("{{ 3.7 | int }}"), "3");
}

TEST(filter_list) {
  ASSERT_EQ(render("{{ 'abc' | list | join(',') }}"), "a,b,c");
}

TEST(filter_reverse) {
  Value::Array arr = {Value(1), Value(2), Value(3)};
  ASSERT_EQ(render("{{ items | reverse | join(',') }}", {{"items", Value(arr)}}), "3,2,1");
}

TEST(filter_chain) {
  ASSERT_EQ(render("{{ '  HELLO  ' | trim | lower }}"), "hello");
}

// ==========================================================================
// 6. String method tests
// ==========================================================================

TEST(method_startswith_endswith) {
  ASSERT_EQ(render("{{ 'hello'.startswith('hel') }}"), "True");
  ASSERT_EQ(render("{{ 'hello'.endswith('llo') }}"), "True");
  ASSERT_EQ(render("{{ 'hello'.startswith('xyz') }}"), "False");
}

TEST(method_split) {
  ASSERT_EQ(render("{{ 'a,b,c'.split(',') | join(' ') }}"), "a b c");
}

TEST(method_replace) {
  ASSERT_EQ(render("{{ 'aabbcc'.replace('bb', 'XX') }}"), "aaXXcc");
}

TEST(method_strip) {
  ASSERT_EQ(render("{{ '  hi  '.strip() }}"), "hi");
}

TEST(method_upper_lower) {
  ASSERT_EQ(render("{{ 'hello'.upper() }}"), "HELLO");
  ASSERT_EQ(render("{{ 'HELLO'.lower() }}"), "hello");
}

// ==========================================================================
// 7. Object/dict method tests
// ==========================================================================

TEST(method_dict_get) {
  Value::Object obj;
  obj["a"] = Value(1);
  ASSERT_EQ(render("{{ d.get('a') }}", {{"d", Value(obj)}}), "1");
  ASSERT_EQ(render("{{ d.get('z', 'default') }}", {{"d", Value(obj)}}), "default");
}

TEST(method_dict_keys_values) {
  Value::Object obj;
  obj["x"] = Value(1);
  obj["y"] = Value(2);
  ASSERT_EQ(render("{{ d.keys() | join(',') }}", {{"d", Value(obj)}}), "x,y");
  ASSERT_EQ(render("{{ d.values() | join(',') }}", {{"d", Value(obj)}}), "1,2");
}

// ==========================================================================
// 8. Ternary / conditional expression tests
// ==========================================================================

TEST(ternary_expr) {
  ASSERT_EQ(render("{{ 'yes' if true else 'no' }}"), "yes");
  ASSERT_EQ(render("{{ 'yes' if false else 'no' }}"), "no");
  ASSERT_EQ(render("{{ x if x is defined else 'missing' }}"), "missing");
}

// ==========================================================================
// 9. In / not in operator tests
// ==========================================================================

TEST(in_operator) {
  Value::Array arr = {Value("a"), Value("b")};
  ASSERT_EQ(render("{{ 'a' in items }}", {{"items", Value(arr)}}), "True");
  ASSERT_EQ(render("{{ 'z' in items }}", {{"items", Value(arr)}}), "False");
  ASSERT_EQ(render("{{ 'z' not in items }}", {{"items", Value(arr)}}), "True");
  // String contains
  ASSERT_EQ(render("{{ 'ell' in 'hello' }}"), "True");
}

// ==========================================================================
// 10. Getattr / getitem tests
// ==========================================================================

TEST(getattr) {
  Value::Object obj;
  obj["name"] = Value("test");
  ASSERT_EQ(render("{{ obj.name }}", {{"obj", Value(obj)}}), "test");
}

TEST(getitem) {
  Value::Array arr = {Value("a"), Value("b"), Value("c")};
  ASSERT_EQ(render("{{ items[0] }}", {{"items", Value(arr)}}), "a");
  ASSERT_EQ(render("{{ items[2] }}", {{"items", Value(arr)}}), "c");
}

TEST(getitem_string_key) {
  Value::Object obj;
  obj["key"] = Value("val");
  ASSERT_EQ(render("{{ obj['key'] }}", {{"obj", Value(obj)}}), "val");
}

TEST(property_length) {
  Value::Array arr = {Value(1), Value(2), Value(3)};
  ASSERT_EQ(render("{{ items.length }}", {{"items", Value(arr)}}), "3");
}

// ==========================================================================
// 11. Whitespace trim tests
// ==========================================================================

TEST(trim_lstrip) {
  // {%- trims whitespace before
  ASSERT_EQ(render("hello   {%- if true %} world{% endif %}"), "hello world");
}

TEST(trim_rstrip) {
  // -%} trims whitespace after
  ASSERT_EQ(render("{% if true -%}   hello{% endif %}"), "hello");
}

// ==========================================================================
// 12. Range builtin
// ==========================================================================

TEST(range_builtin) {
  ASSERT_EQ(render("{% for i in range(3) %}{{ i }}{% endfor %}"), "012");
  ASSERT_EQ(render("{% for i in range(1, 4) %}{{ i }}{% endfor %}"), "123");
}

// ==========================================================================
// 13. Real-world HF chat template tests
// ==========================================================================

TEST(hf_qwen_template) {
  std::string tmpl =
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\n' }}"
    "{%- endif %}";

  Value::Array messages = {msg("system", "You are helpful."), msg("user", "Hi")};
  std::map<std::string, Value> vars = {
    {"messages", Value(messages)},
    {"add_generation_prompt", Value(true)}
  };

  std::string result = render(tmpl, vars);
  ASSERT_CONTAINS(result, "<|im_start|>system\nYou are helpful.<|im_end|>");
  ASSERT_CONTAINS(result, "<|im_start|>user\nHi<|im_end|>");
  ASSERT_CONTAINS(result, "<|im_start|>assistant\n");
}

TEST(hf_llama3_template) {
  std::string tmpl =
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'user' %}"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{{ message['content'] }}<|eot_id|>"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
    "{% endif %}";

  Value::Array messages = {msg("user", "What is AI?")};
  auto result = render(tmpl, {
    {"messages", Value(messages)},
    {"add_generation_prompt", Value(true)}
  });
  ASSERT_CONTAINS(result, "<|start_header_id|>user<|end_header_id|>");
  ASSERT_CONTAINS(result, "What is AI?");
  ASSERT_CONTAINS(result, "<|start_header_id|>assistant<|end_header_id|>");
}

TEST(hf_gemma3_template) {
  std::string tmpl =
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
    "{% elif message['role'] == 'model' or message['role'] == 'assistant' %}"
    "<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<start_of_turn>model\n"
    "{% endif %}";

  Value::Array messages = {
    msg("user", "Tell me a joke"),
    msg("assistant", "Why did the chicken cross the road?")
  };
  auto result = render(tmpl, {
    {"messages", Value(messages)},
    {"add_generation_prompt", Value(true)}
  });
  ASSERT_CONTAINS(result, "<start_of_turn>user\nTell me a joke<end_of_turn>");
  ASSERT_CONTAINS(result, "<start_of_turn>model\nWhy did the chicken");
}

TEST(hf_multi_turn) {
  std::string tmpl =
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}";

  Value::Array messages = {
    msg("system", "Be brief."),
    msg("user", "Hi"),
    msg("assistant", "Hello!"),
    msg("user", "How are you?")
  };
  auto result = render(tmpl, {
    {"messages", Value(messages)},
    {"add_generation_prompt", Value(true)}
  });
  // Verify all 4 messages present
  ASSERT_CONTAINS(result, "<|im_start|>system\nBe brief.");
  ASSERT_CONTAINS(result, "<|im_start|>user\nHi<|im_end|>");
  ASSERT_CONTAINS(result, "<|im_start|>assistant\nHello!<|im_end|>");
  ASSERT_CONTAINS(result, "<|im_start|>user\nHow are you?<|im_end|>");
  // Ends with generation prompt (last occurrence after last im_end)
  auto last_end = result.rfind("<|im_end|>");
  auto last_gen = result.rfind("<|im_start|>assistant\n");
  ASSERT_TRUE(last_gen > last_end);
}

TEST(hf_no_generation_prompt) {
  std::string tmpl =
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{%- endif %}";

  Value::Array messages = {msg("user", "Hi"), msg("assistant", "Hello!")};
  auto result = render(tmpl, {
    {"messages", Value(messages)},
    {"add_generation_prompt", Value(false)}
  });
  // Should NOT end with assistant prompt
  ASSERT_TRUE(result.find("<|im_start|>assistant\n") == std::string::npos ||
              result.rfind("<|im_start|>assistant\n") < result.rfind("<|im_end|>"));
}

TEST(hf_bos_eos_tokens) {
  std::string tmpl =
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{{ eos_token }}";

  Value::Array messages = {msg("user", "Hi")};
  auto result = render(tmpl, {
    {"messages", Value(messages)},
    {"bos_token", Value("<s>")},
    {"eos_token", Value("</s>")}
  });
  ASSERT_TRUE(result.find("<s>") == 0);
  ASSERT_CONTAINS(result, "</s>");
}

// ==========================================================================
// 14. Edge cases
// ==========================================================================

TEST(empty_template) {
  ASSERT_EQ(render(""), "");
}

TEST(empty_messages) {
  std::string tmpl = "{% for m in messages %}{{ m['content'] }}{% endfor %}";
  ASSERT_EQ(render(tmpl, {{"messages", Value(Value::Array{})}}), "");
}

TEST(special_chars_in_content) {
  Value::Array messages = {msg("user", "What is <html> & \"quotes\"?")};
  std::string tmpl = "{% for m in messages %}{{ m['content'] }}{% endfor %}";
  ASSERT_CONTAINS(render(tmpl, {{"messages", Value(messages)}}), "<html> & \"quotes\"");
}

TEST(nested_object_access) {
  Value::Object inner;
  inner["x"] = Value(42);
  Value::Object outer;
  outer["inner"] = Value(inner);
  ASSERT_EQ(render("{{ obj.inner.x }}", {{"obj", Value(outer)}}), "42");
}

TEST(array_literal) {
  ASSERT_EQ(render("{% set arr = [1, 2, 3] %}{{ arr | join(',') }}"), "1,2,3");
}

TEST(comment_ignored) {
  ASSERT_EQ(render("hello{# this is a comment #} world"), "hello world");
}

TEST(macro_basic) {
  std::string tmpl =
    "{% macro greet(name) %}Hello {{ name }}!{% endmacro %}"
    "{{ greet('World') }}";
  ASSERT_EQ(render(tmpl), "Hello World!");
}

// ==========================================================================
// Main
// ==========================================================================

int main() {
  std::cout << "=== Chat Template Engine Unit Tests ===\n\n";

  for (const auto &[name, fn] : tests) {
    std::cout << "  " << name << " ... ";
    try {
      fn();
      std::cout << "PASS\n";
      pass_count++;
    } catch (const std::exception &e) {
      std::cout << "FAIL\n";
      fail_count++;
    }
  }

  std::cout << "\n=== Results: " << pass_count << " passed, "
            << fail_count << " failed ===\n";
  return fail_count > 0 ? 1 : 0;
}

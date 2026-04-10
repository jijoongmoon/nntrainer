# Tasks: HuggingFace-compatible Chat Template Engine

## Goal
Add support for HuggingFace-style chat templates to nntrainer's CausalLM
application so that any model providing a `chat_template` in its
`tokenizer_config.json` can be used for chat without hard-coding per-model
formatting logic.

## Approach
llama.cpp's approach is the reference: ship a C++ Jinja2-compatible engine
(minja/jinja) that can render the Jinja2 template strings shipped inside
HuggingFace `tokenizer_config.json` / GGUF metadata. We implement a
lightweight subset sufficient for the HF chat templates in the wild.

## Plan

### Phase 1 — Engine
- [x] **1.1** Split engine into `jinja/` subdir (value, lexer, parser, evaluator)
- [x] **1.2** `jinja/value.{h,cpp}` — dynamic value type (none, bool, int,
      double, string, array, object) with operators, truthiness, indexing
- [x] **1.3** `jinja/lexer.{h,cpp}` — tokenize `{{ }}`, `{% %}`, `{# #}`,
      identifiers, literals, operators, whitespace-trim markers (`{{-`, `-}}`)
- [x] **1.4** `jinja/parser.{h,cpp}` — AST for output, if/elif/else, for,
      set (incl. dotted names for namespaces), macro, expressions with
      precedence climbing, filters, method calls, ternary
- [x] **1.5** `jinja/evaluator.{h,cpp}` — scoped Context, render for/if/set,
      loop variables (`loop.index`, `loop.first`, `loop.last`, ...),
      tuple unpacking in for, built-in filters and string/dict/array methods
- [x] **1.6** `chat_template_engine.{h,cpp}` — top-level `Template` facade

### Phase 2 — Integration wrapper
- [x] **2.1** `chat_template.{h,cpp}` — load `tokenizer_config.json`, extract
      `chat_template`, `bos_token`, `eos_token`; convert `nlohmann::json`
      messages to `jinja::Value` and render

### Phase 3 — Integration into CausalLM app
- [x] **3.1** `main.cpp` — try dynamic template first, fall back to built-ins
- [x] **3.2** `api/causal_lm_api.cpp` — load template at model init,
      use in `apply_chat_template()` with built-in fallback
- [x] **3.3** `meson.build` — register new sources + include path

### Phase 4 — Testing
- [x] **4.1** Write 69 unit tests covering value types, lexer, rendering,
      control flow, filters, methods, edge cases, and real HF templates
      (Qwen3, Llama3, Gemma3)
- [x] **4.2** Fix bugs surfaced by tests:
      - for-loop `if` filter consumed by ternary parser
      - tuple unpacking for arrays (dict.items())
      - dotted names in `{% set %}`
      - namespace set now propagates across scopes
- [x] **4.3** Remove unused `ltrim` local in lexer (flagged by user)

### Phase 5 — Housekeeping
- [x] **5.1** Merge main (pick up CLAUDE.md)
- [x] **5.2** Create `tasks/todo.md` (this file)
- [x] **5.3** Create `tasks/lessons.md` from this session's corrections

## Review

### What landed
- `Applications/CausalLM/jinja/` — 8 files (~1.8k LoC)
- `Applications/CausalLM/chat_template_engine.{h,cpp}` — Template facade
- `Applications/CausalLM/chat_template.{h,cpp}` — HF config loader
- `Applications/CausalLM/test_chat_template.cpp` — 69 unit tests, all passing
- Integration patches to `main.cpp`, `api/causal_lm_api.cpp`, `meson.build`

### What is intentionally out of scope
- Tool-call auto-parser (llama.cpp's diff-based PEG grammar generation) —
  we support rendering tool-aware templates but don't auto-discover tool
  call formats
- Input marking (prompt injection mitigation)
- GGUF `tokenizer.chat_template` key extraction — we only read
  `tokenizer_config.json` for now; GGUF path can be added later when nntrainer
  has GGUF loader support for this metadata

### What to verify next
- Build on real nntrainer meson tree (we only `-fsyntax-only`-checked)
- Run against a live model checkout to confirm the rendered prompt matches
  what HuggingFace transformers would produce

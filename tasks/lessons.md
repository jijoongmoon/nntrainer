# Lessons

Patterns and corrections captured from sessions. Review at session start.

## Process

### L-001: Check for CLAUDE.md on every branch, not just the current one
**Trigger:** User asked "내가 main branch에 CLAUDE.md를 적어 두었어. 이거 적용하고
있는거지?" after I worked on a feature branch that had not yet been merged
with main.
**Rule:** At session start, if the current branch was created off an older
commit of main, always `git fetch origin main && git show origin/main:CLAUDE.md`
(and `git show origin/main -- tasks/`) to pick up project-level guidance
that may not be on the working branch yet.
**Why:** Branch-local views of the repo can miss instructions the user has
added to main. Acting "autonomously" without those instructions means
violating rules the user has already written down.

### L-002: Use plan mode for non-trivial multi-step features
**Trigger:** Implemented an entire Jinja2 engine (lexer/parser/evaluator +
integration) without entering plan mode or writing `tasks/todo.md` first.
The user had CLAUDE.md requiring plan-first for 3+ step work.
**Rule:** Before starting any task that touches more than ~2 files or
requires architectural decisions, write a plan to `tasks/todo.md` with
checkable items and get confirmation before implementing.
**Why:** Plans expose ambiguity early and give the user a chance to redirect
before significant work is wasted.

### L-003: Split large files proactively; don't wait to be told
**Trigger:** Started writing a single ~600-line `chat_template_engine.cpp`.
User said "크면 나눠서 작성해. 하나의 파일이 너무 크면 말야."
**Rule:** For engines/modules with multiple distinct responsibilities
(types, lexer, parser, evaluator), split into separate files from the start.
Target ~300-500 lines per file. One clear responsibility per file.
**Why:** Easier to review, test, and navigate. Smaller diffs. User shouldn't
have to ask.

### L-004: Remove dead code when you see it, don't just leave it
**Trigger:** User pointed out `bool ltrim = false; ... ltrim = true;` in
`lexer.cpp` where the variable was set but never read — the trim was done
inline.
**Rule:** Before committing, re-scan new code for variables that are
assigned but never read, imports that are unused, parameters that are
ignored. Delete them. Don't write scaffolding "just in case."
**Why:** Dead code confuses future readers and hides real bugs. If the
variable was supposed to gate behavior and that behavior is missing, the
dead assignment makes the bug invisible.

### L-005: Keep subagents in the loop for independent research tasks
**Trigger:** Only used a subagent once (for llama.cpp research). Implemented
the entire Jinja2 engine and wrote all 69 tests myself in the main context.
CLAUDE.md asks for liberal subagent use to keep main context clean.
**Rule:** When a task has independent subparts (write tests, research a
library, explore an unfamiliar directory), delegate to subagents in
parallel. Reserve the main context for orchestration, review, and decisions
that need full history.
**Why:** Main context window fills up fast and subagents can run in
parallel, reducing total latency.

## Engineering

### L-101: When parsing `for x in iterable if cond`, don't call a parser that
consumes `if` as a ternary
**Trigger:** `parse_for()` called `parse_expr()` → `parse_conditional()`
which gobbled the trailing `if` as a ternary, breaking
`{% for x in items if x > 2 %}`.
**Rule:** For contexts where `if` is a statement-level keyword (for-filter,
statement-level conditional), parse the iterable with `parse_or()` (the
level just below conditional) so the ternary path is not taken.
**Why:** Same token means different things in different grammatical
contexts. Precedence climbing parsers need per-context entry points.

### L-102: Tuple unpacking in for-loops must handle array element form too
**Trigger:** `{% for k, v in d.items() %}` failed because `.items()` returns
an array of `[k, v]` pairs, and the evaluator was only unpacking for
`iterable.is_object()`, not for pair-arrays inside an array.
**Rule:** When a for loop declares `loop_var, loop_var2`, always also
handle the case where the iterable is an array of 2-element arrays, not
just a dict.
**Why:** `dict.items()` is the canonical form in Jinja2 and always yields
an array, not an object.

### L-103: Namespace attribute writes must update the defining scope
**Trigger:** `{% set ns.count = ns.count + 1 %}` inside a for-loop wrote to
the inner scope, so the outer read got the original value.
**Rule:** Namespace-style mutable containers should update the scope where
the variable was originally defined, not the current scope. Implement a
`set_in_defining_scope()` on Context that walks the scope stack.
**Why:** Jinja's `namespace()` is explicitly designed to be mutable across
loop iterations; shadowing it in an inner scope defeats the purpose.

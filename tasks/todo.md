# Multi-Model nntr_config Support

## Goal
Let a single top-level `nntr_config.json` describe multiple sub-models, so
that already-registered architectures (e.g. `Qwen3ForCausalLM`,
`LlamaForCausalLM`, `Gemma3ForCausalLM`, ...) can be composed just via
configuration — no new model classes needed.

Move `architecture` selection out of HF `config.json` and into
`nntr_config.json`. Each sub-model lives in its own sub-directory with its
own `config.json` / `generation_config.json` / weight file.

Example demo: **two `Qwen3-0.6B` instances stacked** described by one
top-level `nntr_config.json`.

## Design

### Top-level `nntr_config.json` (multi-model) schema

```json
{
    "model_tensor_type": "FP32-FP32",
    "tokenizer_file": "/abs/path/to/tokenizer.json",
    "sample_input": "...",
    "num_to_generate": 128,
    "init_seq_len": 1024,
    "max_seq_len": 2048,
    "batch_size": 1,
    "bad_word_ids": [],
    "fsu": false,
    "fsu_lookahead": 2,

    "models": [
        {
            "name":  "qwen3_a",
            "architecture": "Qwen3ForCausalLM",
            "dir":   "qwen3-0.6b",
            "model_file_name": "nntr_qwen3_0.6b_fp32.bin",
            "model_type": "CausalLM",
            "fc_layer_dtype": "FP32",
            "embedding_dtype": "FP32"
        },
        {
            "name":  "qwen3_b",
            "architecture": "Qwen3ForCausalLM",
            "dir":   "qwen3-0.6b",
            "model_file_name": "nntr_qwen3_0.6b_fp32.bin",
            "model_type": "CausalLM",
            "fc_layer_dtype": "FP32",
            "embedding_dtype": "FP32"
        }
    ]
}
```

Notes:
- `architecture` replaces `config.json["architectures"][0]` — it becomes
  the Factory key.
- `dir` is relative to the top-level model_path. Two entries can point to
  the same dir (as above) when you want the same weights instantiated twice.
- Per-entry `model_type`, `fc_layer_dtype`, `embedding_dtype`,
  `model_file_name` are merged into a synthesized `nntr_sub_cfg` that is
  handed to `Transformer::setupParameters(...)` unchanged.

### Directory layout

```
res/multi_example/
├── nntr_config.json            # top-level (above)
└── qwen3-0.6b/
    ├── config.json             # HF-style, "architectures" field optional
    └── generation_config.json
```

### Config-loading rules
- `architecture` is taken from the per-entry value. HF `config.json`'s
  `architectures` field is ignored when the new multi-model flow is used.
- Top-level nntr keys (tokenizer_file, batch_size, max_seq_len, fsu, ...)
  are shared; per-entry keys override them.
- Each sub-model gets a `nntr_sub_cfg = deep_copy(top_level) U entry`, so
  existing `Transformer::setupParameters`, `CausalLM::setupParameters`, etc.
  keep working without modification.

### Orchestration (this PR)
Scope = *loading* multiple models from one config. For the demo the binary:
1. Reads top-level nntr_config.json.
2. For each entry in `models`: loads that sub-dir's config.json &
   generation_config.json, creates the model via Factory, initializes, and
   loads weights.
3. Runs each model once with the shared `sample_input` (sequential runs, no
   tensor plumbing between them yet).

Actual stacking (feeding one model's hidden state / tokens into the next)
is follow-up work and out of scope — the point of this PR is the config
scaffolding.

### Backward compatibility
Detect multi-model mode by presence of `"models"` array in
`nntr_config.json`. Otherwise fall through to the current single-model
code path unchanged.

## Tasks

- [ ] Refactor `Applications/CausalLM/main.cpp`:
  - After loading `nntr_cfg`, check for `"models"`.
  - If absent → current legacy path, unchanged.
  - If present → for each entry build `nntr_sub_cfg`, load
    `<model_path>/<dir>/config.json` and
    `<model_path>/<dir>/generation_config.json`, resolve `architecture`
    from the entry, `Factory::create(...)`, `initialize()`, `load_weight()`.
    Store in `std::vector<std::unique_ptr<Transformer>>`.
  - After all loaded, run each model once with `sample_input`.
- [ ] Create example dir `Applications/CausalLM/res/multi_example/`:
  - top-level `nntr_config.json` with two Qwen3-0.6B entries.
  - `qwen3-0.6b/config.json` with real HF Qwen3-0.6B dims (hidden_size=1024,
    num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
    head_dim=128, intermediate_size=3072, vocab_size=151936,
    max_position_embeddings=40960, rope_theta=1000000,
    tie_word_embeddings=true, rms_norm_eps=1e-6).
  - `qwen3-0.6b/generation_config.json` (eos/bos/temperature/top_k/top_p).
- [ ] `meson compile -C build` iterate until clean.
- [ ] Commit to `claude/multi-model-config-vrXd1` and push.

## Non-goals
- Actual tensor pipelining between sub-models.
- Creating new model classes (LinearCausalLM etc.) — user already has the
  architectures they want.
- Touching `quantize.cpp` / `api/causal_lm_api.cpp` — their single-model
  flow is unchanged.
- Providing weight `.bin` files for Qwen3-0.6B; user converts on their own.

## Review
(To be filled in after implementation.)

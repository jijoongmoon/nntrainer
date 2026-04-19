# Multi-Model nntr_config Support

## Goal
Extend Applications/CausalLM so that a single top-level `nntr_config.json`
can describe multiple sub-models (e.g. vision-encoder + LLM, or Linear +
Qwen3-0.6B). Move `architectures` out of HF `config.json` into the per-model
entry of `nntr_config.json`. Each sub-model lives in its own directory with
its own `config.json` / `generation_config.json` / weight file.

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
            "name":  "prelinear",
            "architecture": "LinearCausalLM",
            "dir":   "linear_model",
            "model_file_name": "nntr_linear.bin",
            "model_type": "CausalLM",
            "fc_layer_dtype": "FP32",
            "embedding_dtype": "FP32"
        },
        {
            "name":  "qwen3",
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

### Per-sub-model directory layout

```
res/multi_example/
├── nntr_config.json            # top-level (above)
├── linear_model/
│   ├── config.json             # HF-ish, no "architectures" field
│   └── generation_config.json
└── qwen3-0.6b/
    ├── config.json             # standard HF Qwen3 config w/o architectures
    └── generation_config.json
```

Weight file locations follow `model_file_name` relative to the sub-model dir.

### Config-loading rules
- `architecture` comes from the sub-model entry in `nntr_config.json`.
- Per-sub-model `config.json` provides HF dims; `architectures` field (if
  still present) is ignored.
- Top-level nntr keys (tokenizer_file, batch_size, max_seq_len, fsu, ...) are
  shared; per-sub-model keys override them.
- Each sub-model sees a *synthesized* `nntr_cfg` = top-level keys merged
  with its own entry fields, so existing `setupParameters(...)` code works
  unchanged.

### Orchestration
For this PR we only wire up *independent load/initialize* for N models.
Actual pipelining (vision→LLM, logit fusion, etc.) is future work. The
binary runs the **last** model in `models` against `sample_input` so the
existing single-model run path is preserved; earlier models are just loaded
and initialized (demonstrates that multiple models co-exist).

### Backward compatibility
`main.cpp` detects multi-model mode by presence of `"models"` array in
`nntr_config.json`. Otherwise it falls through to the current single-model
path unchanged.

## Tasks

- [ ] Implement `LinearCausalLM` at `models/linear/linear_causallm.{h,cpp}`
      — a `CausalLM` subclass whose `constructModel()` is just
      `input → embedding → fully_connected → lm_head` (no transformer
      blocks, no RoPE/attention). Override `setupParameters` to tolerate
      missing transformer-specific fields.
- [ ] Add `subdir('linear')` to `Applications/CausalLM/models/meson.build`
      and a `linear/meson.build` mirroring `qwen3/meson.build`.
- [ ] Register `"LinearCausalLM"` factory entry in `main.cpp`.
- [ ] Refactor `main.cpp`:
  - Detect `nntr_cfg["models"]`.
  - For each entry: build per-model `nntr_sub_cfg` by deep-copying top-level
    and overlaying entry fields; load `<model_path>/<dir>/config.json` and
    `<model_path>/<dir>/generation_config.json`; call
    `Factory::create(architecture, cfg, gen, nntr_sub_cfg)`; `initialize()`;
    `load_weight(<model_path>/<dir>/<model_file_name>)`.
  - Keep the current legacy path when `"models"` is absent.
  - Change architecture resolution: prefer per-entry `architecture`; if not
    set, fall back to legacy `cfg["architectures"][0]`.
- [ ] Create example dir `res/multi_example/` with:
  - top-level `nntr_config.json`
  - `linear_model/{config.json, generation_config.json}`
  - `qwen3-0.6b/{config.json, generation_config.json}` (mirroring existing
    qwen3-4b configs scaled to 0.6B: hidden_size=1024, num_hidden_layers=28,
    num_attention_heads=16, num_key_value_heads=8, intermediate_size=3072,
    vocab_size=151936, head_dim=128, etc. — matching HF
    `Qwen/Qwen3-0.6B`).
- [ ] `meson compile -C build` (or existing build dir) and iterate until
      clean.
- [ ] Commit to `claude/multi-model-config-vrXd1` and push.

## Non-goals (out of scope for this PR)
- Actual multi-model tensor plumbing (e.g., feeding one model's output
  embeddings into another). Will be a follow-up once the config scaffolding
  lands.
- Vision encoder implementation.
- Re-running existing single-model configs to add `architectures` to
  nntr_config.json — legacy path stays, we just don't *require* config.json
  to have `architectures` anymore when using the new multi-model flow.

## Review
(To be filled in after implementation.)

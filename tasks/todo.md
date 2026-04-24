# PR 분리 계획 — `claude/add-claude-documentation-P6NvS` 브랜치 기능별 분할

## 워크플로우 (항목당 반복)

1. `main`에서 `feature/<이름>` 브랜치 생성
2. 해당 기능 커밋을 cherry-pick (또는 재구성)
3. 커밋 메시지/내용 정리
4. 빌드 + 테스트 수행 → 통과 확인
5. `origin`에 푸시 → PR 생성
6. 리뷰 반영 → 머지

각 PR이 "main 기준으로 단독 동작"하는지가 핵심 검증 포인트.

---

## 진행 상태 범례

- `[ ]` not-started
- `[WIP]` 브랜치 생성됨, 편집/테스트 중
- `[TEST]` 빌드/테스트 수행 중
- `[PR]` 원격 푸시 + PR open
- `[MERGED]` 머지 완료

---

## Group A — 독립적, 작은 수정 (병렬 가능)

- `[ ]` **A1. Build/lint housekeeping (잡동사니 수정 묶음)**
  - 커밋: `d056cff`, `49a6de9`, `17e2350`, `61ad559`, `a3db461`, `e4ca6bd`, `1c4b029`, `06e9d2d`
  - 규모: ~10 files / ~90 lines
  - 의존: 없음
  - 브랜치: `feature/build-lint-fixes`

- `[ ]` **A2. CLAUDE.md 추가/확장**
  - 커밋: `64be271`, `185cfb4`
  - 규모: 1 file / 282 lines
  - 의존: 없음
  - 브랜치: `feature/claude-md`

- `[ ]` **A3. Depthwise Conv1D 레이어 (im2col 최적화)**
  - 커밋: `6493bab`
  - 규모: 2 files / 554 lines
  - 의존: 없음
  - 브랜치: `feature/depthwise-conv1d`

---

## Group B — Tensor / Public API 재설계

- `[WIP]` **B1. Tensor API 재설계 (Pimpl, symbolic/eager, graph-based compile)**
  - 원본 커밋: `466abfc` ("Add safetensors format support and weight loading API" — 제목은 misleading, 실제 내용은 Tensor API 재설계 + safetensors enum 1줄)
  - 브랜치: `feature/tensor-api` (생성 완료, cherry-pick 성공 @ `3011fb3`)
  - **TODO**:
    - [ ] 커밋 메시지 정정 (예: `"Redesign ml::train::Tensor API with Pimpl pattern + symbolic graph compile"`)
    - [ ] `model.h`의 `MODEL_FORMAT_SAFETENSORS` enum 1줄 제거 (차후 B2 PR로 이동)
    - [ ] 빌드 테스트
    - [ ] `ccapi` 테스트 실행
    - [ ] 푸시 + PR
  - 규모: 5 files / ~2188 lines
  - 의존: 없음

- `[ ]` **B2. Safetensors 포맷 지원 + weight loading 구현**
  - 포함: `MODEL_FORMAT_SAFETENSORS` enum (B1에서 제외한 1줄) + 실제 로딩 구현
  - 실제 로딩 구현 위치: **조사 필요** — 가능성: `1b67c7d` (layers/model/graph/API 업데이트) 또는 `0f22588` (CausalLM) 내부에 분산
  - 의존: B1 머지 후

- `[ ]` **B3. LazyTensor infrastructure**
  - 커밋: `bc4b2cb`
  - 규모: 21 files / 924 lines
  - 의존: 없음 (단 B1과 동일 영역 일부 건드릴 수 있음 — 선행 병합 순서 확인 필요)
  - 브랜치: `feature/lazy-tensor`

---

## Group C — 양자화 + 변환 도구

- `[ ]` **C1. Q1_0 1-bit 양자화 + AVX2/NEON 커널 + 단위 테스트**
  - 커밋: `674a950`, `887b8de`
  - 규모: 3 files / ~1050 lines
  - 의존: D1 (CPU backend 업데이트) 선행 권장
  - 브랜치: `feature/q1-0-quantization`

- `[ ]` **C2. GGUF → NNTrainer converter (Bonsai Q1_0)**
  - 커밋: `07a5866`
  - 규모: 2 files / 593 lines
  - 의존: C1
  - 브랜치: `feature/gguf-converter`

---

## Group D — 기반 인프라 (base, 단일 커밋이라 재구성 고려)

- `[ ]` **D1. CPU backend: NEON/GGML/AVX2/KleidiAI 업데이트**
  - 커밋: `283706e`
  - 규모: 24 files / 1688 lines
  - 의존: 없음
  - 브랜치: `feature/cpu-backend-update`

- `[ ]` **D2. ThreadManager refactor**
  - 커밋: `066c6c3`
  - 규모: 19 files / 6830 lines
  - 의존: 없음
  - 브랜치: `feature/thread-manager`
  - **주의**: 단일 커밋이라 크기 큼. 필요 시 sub-PR로 분해

---

## Group E — ComputeOps dispatch 시리즈 (순차 의존, 순서대로 머지)

- `[ ]` **E1. ComputeOps 테이블 도입 (core)**
  - 커밋: `344d614`, `7872ef7`, `0e6820a`, `367b872`, `2677168`, `9890f13`, `a3fb620` (docs)
  - 규모: ~20 files / ~1600 lines
  - 의존: D1
  - 브랜치: `feature/compute-ops-core`

- `[ ]` **E2. ComputeOps: tensor ops (B-2 패턴)**
  - 커밋: `2131b40`, `c7a7cfc`, `548bd6a`, `74afaec`, `6ac322b`, `7d9919a`, `91f03c7`, `2f9796a`
  - 규모: ~35 files / ~1200 lines
  - 의존: E1
  - 브랜치: `feature/compute-ops-b2`

- `[ ]` **E3. ComputeOps: tensor types & quantize 확장**
  - 커밋: `0455197`, `6f05f44`, `c45a96a`, `8faadf4`, `e314f00`
  - 규모: ~20 files / ~500 lines
  - 의존: E2
  - 브랜치: `feature/compute-ops-quantize`

- `[ ]` **E4. ComputeOps: per-arch 분리 + vendor 통합**
  - 커밋: `cd9bdbf`, `2f0e40e`, `26d454d`, `6a541bb`, `537b888`, `fb49758`, `29e9ea3`, `6021fee`
  - (drop: `793701e` + `86ae489` — revert 쌍)
  - 규모: ~20 files / ~2000 lines
  - 의존: E3
  - 브랜치: `feature/compute-ops-per-arch`

- `[ ]` **E5. ComputeOps: OpenCL ops + ContextData 확장**
  - 커밋: `2d3895f`, `0d8967e`, `87ee8ec`
  - 규모: ~10 files / ~250 lines
  - 의존: E4
  - 브랜치: `feature/compute-ops-opencl`

---

## Group F — QNN (Qualcomm NPU) 플러그인

- `[ ]` **F1. QNN context plugin (.so)**
  - 커밋: `5f87af0`, `19b67ae` (cherry-pick from pr/3826), `782971f`, `6c15105`, `5a0ea1a`, `47cfef8` (docs)
  - 규모: ~10 files / ~200 lines
  - 의존: E5
  - 브랜치: `feature/qnn-plugin`

---

## Group G — 독립 기능 (큰 것들)

- `[ ]` **G1. Chat template engine (Jinja2) + 69 테스트**
  - 커밋: `bf0349f`, `c751297`
  - 규모: ~19 files / ~3900 lines
  - 의존: 없음
  - 브랜치: `feature/chat-template`

---

## Group H — KV Cache externalization 시리즈 (가장 복잡, 순차)

- `[ ]` **H1. Typed input layers + non-FP32 inference API**
  - 커밋: `e180a14`
  - 규모: 5 files / 69 lines
  - 의존: 없음
  - 브랜치: `feature/typed-input-layer`

- `[ ]` **H2. External tensor injection API (`setLayerExternalTensor`)**
  - 커밋: `fc54c7e`, `0be4748` (+ WIP `ac76b1b`, `2b6b180`, `ea5187a`, `37ee507` squash)
  - 규모: ~23 files / ~510 lines
  - 의존: H1
  - 브랜치: `feature/external-tensor-api`

- `[ ]` **H3. KVCacheManager + `forwarding()` 구현 (Phase 1-3)**
  - 커밋: `f27620a`, `ef5a3fb`, `8a937e7`, `92e2048`
  - 규모: ~17 files / ~2000 lines
  - 의존: H2, I1 (CausalLM 모델 업데이트 선행)
  - 브랜치: `feature/kv-cache-manager`

- `[ ]` **H4. KV cache 통합 + multi-batch/multi-turn 테스트 (Phase 4-7)**
  - 커밋: `bf1f43b`, `b36b641`, `684bfee`, `9aa816d`, `e6efb97`, `4808954`, `3e7ead5`, `72da005`
  - 규모: ~32 files / ~1600 lines
  - 의존: H3
  - 브랜치: `feature/kv-cache-integration`

- `[ ]` **H5. `incremental_forwarding` 완전 제거 (Phase 9)**
  - 커밋: `ac3d795`, `12fb772`, `ed8a4f2`
  - 규모: ~78 files / ~3350 lines (기계적 제거라 리뷰는 수월)
  - 의존: H4
  - 브랜치: `feature/remove-incremental-forwarding`

---

## Group I — 거대 커밋 (재구성 필요, 여러 PR로 쪼개야 함)

### I1. CausalLM 확장 (원 커밋 `0f22588`, 33 files / 2734 lines)
"Qwen3, Gemma3, MoE models, quantize utility" — 여러 모델이 섞여있음. **3개로 분해 권장**:
- `[ ]` I1a. Qwen3 모델
- `[ ]` I1b. Gemma3 모델
- `[ ]` I1c. MoE 모델 + quantize utility

### I2. TorchFXConverter (원 커밋 `37c5504`, 71 files / **22,990 lines**) — 매우 큼
`tools/TorchFXConverter/` 전체. **4-5개로 분해 권장**:
- `[ ]` I2a. Core tracer + node_mapper + op_registry
- `[ ]` I2b. emitter_cpp/ (C++ source generation)
- `[ ]` I2c. emitter_ini/ (INI config generation)
- `[ ]` I2d. patterns/ (attention, ffn, block, ssm 패턴 감지)
- `[ ]` I2e. tests/ + plugin_system + weight_converter

### I3. Build + docs + sample apps + tests 업데이트 (원 커밋 `6d2c6af`, 27 files / 7516 lines)
"Update build system, documentation, sample apps, and tests" — **4개로 분해 권장**:
- `[ ]` I3a. Build system 변경
- `[ ]` I3b. Sample apps 업데이트
- `[ ]` I3c. Documentation
- `[ ]` I3d. Tests

### I4. Layers/model props/Graph/API 업데이트 (원 커밋 `1b67c7d`, 25 files / 1905 lines)
"Update layers, model properties, graph, and API" — 테마 불명확. 내용 파악 후 분해 필요:
- `[ ]` I4. (내용 확인 후 1~3개 PR로 결정)

### I5. GraphVisualizer VS Code 확장 (원 커밋 `33e962d`, 17 files / 6524 lines)
- `[ ]` I5. 단일 PR 가능 (언어·타겟이 분리돼 있어 리뷰 부담 상대적으로 낮음). TypeScript 툴.
  - 브랜치: `feature/graph-visualizer-vscode`

### I6. KV cache TorchFXConverter 바인딩 (커밋 `57f5414`, `df53ada`)
- `[ ]` I6. TorchFXConverter PR 그룹(I2) 중 마지막 PR에 흡수

---

## 드롭할 것

- `9dcd056`, `c0f997d`, `f99d658`, `2cc557b` — merge commits (rebase 시 자동 제거, cherry-pick 불필요)
- `793701e` + `86ae489` — 중복 제거 + revert 쌍 (E4에서 둘 다 제외)

---

## 현재 진행 상황

- **완료**:
  - `main` 브랜치를 `github.com/nntrainer/nntrainer` upstream과 동기화 (force-push)
  - 전체 분리 항목 목록 작성

- **진행 중**:
  - **B1 (Tensor API 재설계)**: `feature/tensor-api` 브랜치에 `466abfc` cherry-pick 완료 @ `3011fb3`
    - 남은 작업: 커밋 메시지 정정 → `MODEL_FORMAT_SAFETENSORS` enum 제거 → 빌드/테스트 → PR

- **다음**: B1 완료 후 사용자 지시에 따라 다음 항목 선택 (Group A 쉬운 것들 먼저 or 의존성 없는 G1/D1 중 선택)

---

## 규모 요약

| Group | PR 수 | 총 규모 |
|-------|------|---------|
| A (housekeeping) | 3 | ~950 lines |
| B (Tensor/API) | 3 | ~5,100 lines |
| C (양자화) | 2 | ~1,650 lines |
| D (기반) | 2 | ~8,500 lines |
| E (ComputeOps) | 5 | ~5,550 lines |
| F (QNN) | 1 | ~200 lines |
| G (독립 대형) | 1 | ~3,900 lines |
| H (KV cache) | 5 | ~7,500 lines |
| I (재구성 필요) | ~13 | ~33,000 lines |
| **합계** | **~35 PR** | **~66,000 lines** |

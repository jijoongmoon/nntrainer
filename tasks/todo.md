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

- `[PUSHED]` **B1. Tensor API 재설계 (Pimpl, symbolic/eager, graph-based compile)**
  - 브랜치: `feature/tensor-api` (pushed)
  - 커밋:
    - `0b51dc1` Redesign ml::train::Tensor API with Pimpl + symbolic graph compile (5 files / +3750 / −64)
    - `d6ed9df` Migrate sample apps and ccapi tests to symbolic tensor graph API (15 files / +2442 / −1546)
  - author/committer: `Jijoong Moon <jijoong.moon@samsung.com>` ✓
  - 검증:
    - 본체 + 13 migrated Apps 빌드 **561/561 타겟 ✓**
    - `unittest_ccapi` **117/117 ✓** (tensor API 새 그룹 86개 + lazy chain 테스트 포함)
    - `unittest_models` 156/156, `unittest_nntrainer_modelfile` 267/267, `unittest_nntrainer_graph` 12/12 모두 회귀 없음
  - **B3 (LazyTensor)와의 관계**: 독립. chain/lazy API는 `std::function` 람다 큐로 자체 구현.
  - 제외하여 타 PR로 이동:
    - `MODEL_FORMAT_SAFETENSORS` enum + `docs/weight-format-specification.md` → B2
    - `external_cache_mha_compile_p` 테스트 (5-input MHA 사용) → H2
  - **남은 작업**: GitHub에서 PR 생성 (사용자 판단)

- `[PR]` **B2. Safetensors 포맷 지원 + weight loading 구현**
  - 브랜치: `feature/safetensors` tip `bb4b778` (main 기반, 단독 PR 가능)
  - 포함 파일: `api/ccapi/include/model.h` (`MODEL_FORMAT_SAFETENSORS`), `api/nntrainer-api-common.h` (`ML_TRAIN_MODEL_FORMAT_SAFETENSORS = 6`), `nntrainer/models/neuralnet.cpp` (save/load with JSON header + `std::thread` parallel mmap in INFERENCE), `nntrainer/models/neuralnet.h` (`convertBinToSafetensors` 선언), `nntrainer/utils/safetensors_util.{h,cpp}` (dtype mapping + header build/parse), `Applications/CausalLM/models/transformer.cpp` (load_weight/save_weight 확장자 자동 감지), `test/unittest/unittest_nntrainer_safetensors.cpp` (4개 라운드트립 테스트), `docs/weight-format-specification.md`
  - **후속(별도 PR)**:
    - D2 머지 후 `std::thread` → `ThreadManager` 전환
    - B2-f1. `bin_to_safetensors` CLI (name-aware 변환 유틸) — 후속 I2(TorchFXConverter)보다 선행 가능
    - B2-f2. `Applications/CausalLM/res/*/weight_converter.py` (6개 — gemma3, qwen2, qwen3(4b·30b-a3b), gpt-oss-20b, kalm-embedding) 에 safetensors 출력 모드 추가. 각 모델별 `hf_key → nntrainer weight name (<layer>:<role>)` 매핑 테이블 필요
    - B2-f3. `Applications/CausalLM/quantize.cpp` 가 `.safetensors` 출력 지원 (양자화된 config 로 재빌드 → load tmp BIN → save safetensors). dtype 변환 + safetensors 동시 지원 전제
  - **TorchFXConverter (I2) 연계 필수 노트**: TorchFXConverter 가 safetensors 를 생성할 때 nntrainer canonical weight name 규칙 `<layer_name>:<param_role>` 을 따라야 함 (예: `fc1:weight`, `embedding0:weight`). `layer_context.h::requestWeight` 에서 `prefix + ":" + name` 로 조립되며, `prefix` 는 layer property `name=...`, `name` 은 각 layer 구현의 고정 역할 문자열. I2 PR 에서 `WeightMap` 을 이 규칙에 맞춰 확장해야 B2-f1/f2 와 호환 가능.
  - PR URL: https://github.com/jijoongmoon/nntrainer/pull/new/feature/safetensors

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
  - **현재 상태 (main)**: `thread_manager.{h,cpp}` + `completion_token.h` + `barrier.h` 가 이미 main 에 들어있고 `parallel_for` 경로가 좋은 성능을 내고 있음 (유지해야 함). 그러나 **IO thread 가 없어** FSU swap in/out 이 동기 경로로 실행됨.

- `[ ]` **D3. ThreadManager IO thread + FSU swap / look-ahead 통합 (큰 작업, 주의 필요)**
  - 목표: `CachePool` / `TensorPool` / `MemoryPool` / `CacheLoader` 의 swap-in/out 경로를 ThreadManager 기반 비동기 IO 로 옮기고, `fsu_lookahead` 지정 시 다음 실행 순서의 weight 를 선행 로드하도록 연결.
  - **제약**:
    - main 의 `ThreadManager::parallel_for` 경로는 **건드리지 않는다** (성능 검증됨).
    - IO thread 는 **별도 thread (single-producer / few-consumer 큐)** 로 모델 실행 thread 와 격리.
    - FSU-on 상태에서 기존 동작(정확성)이 회귀 없어야 함 → `unittest_cache_pool_fsu`, `integration_test_fsu` 로 방어.
  - **조사(PR 작성 전 필수)**:
    - main `CacheLoader` 의 현재 swap-in/out 경로 — blocking read, task queue, promise/future 사용 여부
    - main `Manager::LoadTensors` / `UnloadTensors` / `checkLoadComplete` / `checkUnloadComplete` 의 완료 추적
    - main `CachePool::loadCacheExecAsync` / `flushCacheExecAsync` 시그니처 — `TaskExecutor::CompleteCallback` 의존 vs `std::function<void(int)>` 직접
    - `CacheElem::CompletionToken`(혹은 등가물) 존재 여부 / main 에서 사용 중인지
    - source branch (`claude/add-claude-documentation-P6NvS`) 의 B3 커밋(`bc4b2cb`)이 이 영역에서 **CacheLoader 제거 + ThreadManager 기반 CompletionToken** 으로 리팩터한 부분을 참고 — 단 그 커밋은 다른 infra 변경과 섞여있어 관련 부분만 추출 필요.
    - `fsu_lookahead` 가 현재 어떻게 주입되는지 (`Fsu` / `FsuLookahead` model property → `Manager` → `TensorPool`).
  - **구현 축 (예상)**:
    1. `ThreadManager::submit_io(std::function<void()>)` — IO 전용 thread(pool? single? few?) 추가. `parallel_for` 와 동일한 ThreadManager 인스턴스에 IO 슬롯만 추가하여 단일 지점 관리.
    2. `CompletionToken` + `Barrier` 를 CacheElem 당 하나씩 두고, Manager 의 promise/future maps 를 제거.
    3. `CacheLoader` 의 blocking 경로를 ThreadManager IO submit 로 교체. 기존 `CacheLoader` 클래스를 얇게 유지하거나, 역할을 `CachePool::loadCacheExecAsync` 로 흡수.
    4. `Manager::LoadTensors(order, remainder_lookahead)` 의 look-ahead 계산은 그대로 유지, 실제 submit 만 ThreadManager 로 위임.
    5. 각 layer_node 실행 직전에 `checkLoadComplete(order)` 로 기다림 — `CompletionToken::wait()` 한 줄로 교체.
  - **테스트**:
    - 기존: `unittest_cache_pool_fsu` (정확성), `integration_test_fsu` (end-to-end)
    - 추가: `unittest_thread_manager_io` (IO submit + completion 단위), FSU on/off 비교 벤치 (lookahead=1,2,3 에서 throughput 측정)
  - **롤아웃**:
    - Phase 1: IO thread 추가 + CacheElem CompletionToken 도입 — 기존 CacheLoader 경로와 병존 (feature flag)
    - Phase 2: CacheLoader 를 ThreadManager 기반으로 교체, FSU 경로 전환
    - Phase 3: FSU off 경로 회귀 검증 + 벤치 공개 + feature flag 제거
  - **의존**: D2 선행 (ThreadManager refactor 자체가 먼저 main-ized) 불필요 — 이미 main 에 ThreadManager 존재. 단 `CacheElem::CompletionToken` / `Barrier` 는 main 에 있는지 재확인 후 결정.
  - **주의**: 이 작업은 **cache pool + tensor pool + memory pool + cache loader** 네 컴포넌트를 가로지르므로 잘못 건드리면 FSU 기반 LLM 추론 전체가 깨질 수 있음. 설계 리뷰(사용자 / myungjoo/lhs8928) 후 착수.
  - 브랜치 (예정): `feature/thread-manager-io-fsu`

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
- **safetensors 연계 (B2 cross-ref)**: `tools/TorchFXConverter/weight_converter.py::WeightMap` 가 safetensors 출력 모드를 지원할 때 nntrainer canonical weight name (`<layer_name>:<param_role>`, layer_context.h::requestWeight 참조) 규칙을 따라야 함. B2 PR 의 `docs/weight-format-specification.md` 에 해당 규칙이 문서화됨 (B2 관련 후속 작업: B2-f1 `bin_to_safetensors` CLI / B2-f2 `res/*/weight_converter.py` 확장).

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

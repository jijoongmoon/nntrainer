# 새 머신 이사 — nntrainer GPU CausalLM (2026-06-19 갱신)

이사 방식 = **repo는 git clone, x86 모델은 단말(S26U) adb-pull, 코드/세션메모리만 패키징.**
새 머신에서 **스크립트 하나** 실행하면 clone→메모리복원→모델pull→Intel빌드→Android빌드→단말배포→smoke test 까지 끝.

## 🚀 한 줄 (새 머신)
```bash
# 1) 이 머신에서 패키지 생성
.claude/scripts/migrate_pack.sh ~/migrate          # → ~/migrate/ (수십 MB)
# 2) ~/migrate 디렉토리를 새 머신으로 복사 (scp/rsync/USB 등)
# 3) 새 머신에서
bash ~/migrate/migrate_setup.sh [NEW_BASE] [NDK경로] [단말serial]
#   NEW_BASE 기본=$HOME → repo=$HOME/nntrainer, 모델=$HOME/qwen3_e2e
```
`migrate_setup.sh`는 **경로 무관**(NEW_BASE에서 전부 파생) — dir path 바뀌어도 OK.
단계별로 SKIP_CLONE / SKIP_MEMORY / SKIP_MODELS / SKIP_INTEL / SKIP_ANDROID / SKIP_DEPLOY / SKIP_TEST=1 로 부분 실행 가능.

## 무엇이 어디서 오나
| 자산 | 출처 | 비고 |
|---|---|---|
| repo base (코드) | **git** `origin/main` (github, public) | clone — 베이스만 |
| **unpushed 커밋** (이 세션 작업: CUDA 등 +99) | **`session.bundle`** (단말 경유) | github에 없음 → bundle delta(≈0.8MB)로 운반, setup이 clone 위에 적용. **jijoongmoon push 불필요** |
| `.claude/` 툴링(scripts/reports/profiles/baselines) | `claude_tooling.tar.gz` | presentation(16MB)/로그 제외 |
| **세션 메모리** `memory/` | `claude_memory.tar.gz` | 매 세션 자동 로드분(MEMORY.md) |
| x86 모델 3종 | **단말 adb-pull** | gemma4 3.2G/gemma2 1.5G/qwen3 351M, tokenizer 경로 자동 보정 |
| 단말 배포물 | 새 머신에서 **재빌드 후 push** | 6 artifact (아래) |

> ⚠️ **운반체 단말 = S26U(`R3CY70LV96T`)** — x86 모델 3종이 거기 있어야 setup이 adb-pull 가능.
> `migrate_pack.sh`는 연결된 단말 중 R3CY70LV96T 우선 선택(없으면 첫 device). **S26U를 연결**하고 패킹할 것
> (모델 없는 단말에 적재하면 setup의 모델 pull이 실패). 특정 단말 강제: `DEVICE=R3CY70LV96T migrate_pack.sh`.

세션 transcript(417MB)는 기본 제외 — `INCLUDE_TRANSCRIPT=<uuid> migrate_pack.sh` 로 특정 세션만 동봉 가능(literal `claude --resume`용). 실질적 "세션 이어서"는 `memory/`(MEMORY.md 인덱스 자동 로드)로 충분.

## 단말(S26U) — 이미 적재됨, 새 머신선 재빌드/재배포만
- serial **`R3CY70LV96T`** (Adreno 840), root `/data/local/tmp/nntrainer/causallm`.
- 모델·프롬프트(prompt_1p2k.txt)는 단말에 보존 → 새 머신은 코드만 빌드해 **6 artifact push**:
  `nntrainer_causallm`·`libcausallm_core.so`(jni/obj/local/arm64-v8a),
  **`libccapi-nntrainer.so`**·`libnntrainer.so`(builddir/android_build_result/lib/arm64-v8a),
  `libOpenCL.so`(builddir/opencl/lib/arm64-v8a), `libc++_shared.so`(NDK).
  ⚠ **libccapi-nntrainer.so 가 #1 자주 누락** — Tensor-API 그래프 컴파일이 여기 있어서 빠지면 무음 dtype/그래프 버그. (KV placeholder dtype abort 사례: [[project_adreno_fp16kv_build_regression]])

## 빌드 함정 (스크립트가 처리하지만 알아둘 것)
1. **Intel**: meson에 **`-Denable-transformer=true`** 필수 (없으면 `nntr_causallm` 타깃 자체가 없음) + `-Denable-opencl -Denable-fp16`.
2. **Android lib**: `package_android.sh` 기본이 **opencl off** → `-Denable-opencl=true` 전달(스크립트가 forward). 안 하면 libnntrainer에 GPU 심볼(clSVMAlloc 등) 없음.
3. **Android app**: `cd Applications/CausalLM/jni && ndk-build ... causallm_core nntrainer_causallm` → 산물은 `obj/local/arm64-v8a/`(libs/는 stale 가능).
4. NDK 27.2.12479018 (`~/Android/Sdk/ndk/`), Intel은 host Intel NEO ICD(`libOpenCL.so.1`) 필요.

## 검증 기준 (이사 직전, from-scratch 양 플랫폼, 회귀 0)
- Adreno 1K: gemma4 **2473**/15.6 "Seoul", gemma2 **819**/14.0, qwen3 **2151**/22.0
- Intel  1K: gemma4 **~1600**/5.17, gemma2 **690**/7.79, qwen3 **1936**/9.35
- 실행/env/troubleshooting 상세 = `Applications/CausalLM/README_GPU.md`

## 새 머신 사전조건 (스크립트가 preflight로 점검)
git, meson, ninja, adb, python3, NDK 27.2.12479018, Intel NEO OpenCL ICD, adb 단말 인증(R3CY70LV96T).
git push 필요 시 jijoongmoon 인증(커밋 author `Jijoong Moon <jijoong.moon@samsung.com>`).

## 참조 무결성
- git: `jijoongmoon/gpu/v8c-on-main` = `origin/gpu/v8c-on-main` = `f5a5d2d6` (origin/main `82ac8d32` 위). 복구 태그 `backup/pre-rebase-origin-main-2026-06-19`=`1233b21d`(rebase 전).
- 메모리 인덱스: `~/.claude/projects/<repo경로키>/memory/MEMORY.md`.

---
이전(2026-06-12) repo-tar 방식은 절대경로 baked-in 문제로 폐기. 현재는 git-clone + path-agnostic setup.

# run_llm.sh — HW × 모델 실행 매트릭스 (2026-07-08 검증)

단일 러너: `./.claude/scripts/run_llm.sh <backend> <model> [perf|"질문"]`
(3번째 인자 없음 = coherence 테스트: 표준 질문을 모델별 chat template로 자동 래핑)

## 백엔드

| backend | HW / 빌드 | 핵심 env (스크립트가 자동 설정) |
|---|---|---|
| `xmx` | Intel Xe3 GPU, XMX/DPAS (`build_cl`) | CL 공통 + `NNTR_FC_XMX=1` |
| `intel` | Intel Xe3 GPU, dp4a (`build_cl`) | CL 공통 + `NNTR_FC_XMX=0` |
| `wrap` | Intel XMX, uint16-Half wrapper (`build_wrap`) | xmx와 동일 |
| `cuda` | NVIDIA RTX, SAFE 지원구성 (`build_cuda`) | `NNTR_ENGINE=cuda` + ROPE/ATTN/KV_UVM/GEGLU/ELTWISE/QKNORM/FLASH_DECODE=64/BLOCKQ/CUBLAS/PREWARM |
| `cuda-fast` | RTX discrete 가속 | SAFE + `DEV_ACT/VCOPY_PREFILL/RMSNORM_OFF=all/M2B/ASYNC` |
| `adreno` | S26U(R3CY70LV96T) Adreno 840 | `FC_INT8_GPU/MHA_GPU/SVM_POOL/KV_IMG_ATTN/CLMEM_POOL` + threads 4 |
| `adreno-cpu` | S26U ARM CPU (KAI, fp32-act) | `NNTR_ENGINE=cpu` — **quant CPU 골든은 여기서만** (x86 int4 CPU = NYI) |

CL 공통: `NNTR_GPU_SVM_POOL=1 NNTR_V8C_BUF=1 NNTR_MHA_GPU=1 NNTR_FC_GPU=1 NNTR_FC_INT8_GPU=1 NNTR_GPU_CLMEM_POOL=1`

## 모델

| model | 로컬 dir | 단말 dir | 템플릿 |
|---|---|---|---|
| `gauss4` | GAUSS4/nntr_model_untie2_qs4cx | models/gauss4_packed | gemma4 턴마커 + no-think 고스트 채널(`<|channel>thought\n<channel|>`), eos=106 |
| `gemma4` | qwen3_e2e/gemma4_qs4cx_fp16 | models/gemma4_qs4cx_fp16 | `<bos><|turn>user\n…<turn|>\n<|turn>model\n` |
| `gemma4-lmint4` | qwen3_e2e/gemma4_lmint4 | models/gemma4_lmint4 | 〃 (README known-good/CUDA 기준 모델) |
| `gemma2` | qwen3_e2e/gemma2_lg_q6k_qs4cx | models/gemma2_qs4cx_regen (md5 동일) | 없음(base) — raw 이어쓰기 |
| `qwen3` | qwen3_e2e/qwen3_lg_q6k_qs4cx | models/qwen3_qs4cx_regen (md5 동일) | ChatML + 빈 `<think>` 필수 |

## 기대 결과 (2026-07-08 실측)

### coherence (기본 모드)
| model | 기대 답 | 백엔드별 편차(기지 사항) |
|---|---|---|
| gauss4 | " Seoul" (+"28.5 million people"류) | 전 백엔드 정답 |
| gemma4(-lmint4) | "**Seoul**" | 전 백엔드 정답 |
| gemma2 | 프랑스 관련 이어쓰기 (base라 "Paris" 미보장 — ARM-CPU만 Paris 언급) | fp16-act GPU들은 마진 이탈 정상 |
| qwen3 | CUDA/adreno-cpu: "**Seoul**" | Intel "Da Yon"·"Hae Soo"류/Adreno "trick question" = OpenCL fp16 마진(0.6B 한계, 버그 아님) |

### perf @~1K prefill (`perf` 모드, prompt_1p2k, tok/s prefill/decode)
| model | xmx | cuda(-fast) | adreno | adreno-cpu |
|---|---|---|---|---|
| gauss4 | ~2100/16.4 | 5134/62.5 | **1249/19.2** | (느림, 골든용) |
| gemma4 qs4cx | ~2500/16+ | — | **2373/21.9** | 170/2.0 |
| gemma4-lmint4 | — | **5808/64** (cuda-fast, 목표 5550/36 초과) | **2461/21.8** (README 2454/18.2 일치) | — |
| gemma2 | ~1839/15 | 3246/52 | 835/16.0 | 75/8.4 |
| qwen3 | ~2551/38 | 4861/145 | 2198/33 | 315/34.7 |

## 함정 모음 (이 매트릭스와 얽힌 것만)
- **842tok 완결 지문(prompt_1k.txt)에 chat 모델이 1토큰 EOS를 내는 건 정상** (버그 아님). perf는 중간 절단되는 prompt_1p2k 기준.
- `NNTR_KV_IMG_ATTN`은 `ca7e36a9` 이후 **값-체크** — `=0`이 실제로 끔 (그 전엔 presence-check라 못 껐음). image attn은 프로세스 단위 all-or-nothing — 레이어/콜 단위로 섞으면 안 됨.
- sliding window는 OHWI 커널이 직접 마스킹(`local_window` 인자) — window<max_seq 모델(gauss4 W=1024, gemma4 W=512)도 image 경로 안전.
- x86에서 no-env 실행은 CPU가 아님(XMX/lmhead-q6k auto-on). 진짜 CPU 레버는 `NNTR_ENGINE=cpu`지만 **x86 int4 CPU GEMM은 NYI** — quant CPU 골든은 단말(`adreno-cpu`)에서.
- `build_cuda` 바이너리를 `NNTR_ENGINE=cuda` 없이 실행 금지(깨진 하이브리드).
- CUDA 실행마다 뜨는 `failed to register factory on cuda ctx ... already taken key` 는 **benign** (known-good 정답/제성능 실행에서도 항상 출력됨) — 쫓지 말 것.
- 진단: `NNTR_V8C_FP16_TRACE=1`(FC별 RELERR, **buffer 경로 전용** — image 경로는 -1), `NNTR_V8C_BUF=1`(Adreno에서 buffer 경로 강제 → 프로브 활성), `[IMG-ATTN] engaged ... hQ/hKV/d` 라인으로 engage 인스턴스 판별.
- 단말은 폰 2대 연결 가능 — 항상 `adb -s R3CY70LV96T`.

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
| `gauss4-ple` | GAUSS4/nntr_model_untie2_qs4cx_ple | models/gauss4_ple | 〃 — PLE sidecar(mmap) 분리, 출력은 gauss4와 **bit-exact** |
| `gemma4-ple` | qwen3_e2e/gemma4_qs4cx_fp16_ple | models/gemma4_ple | gemma4와 동일 — PLE sidecar(mmap) 분리, 출력 bit-exact |
| `gauss4-side` | GAUSS4/nntr_model_untie2_qs4cx_side | models/gauss4_side | 〃 — **PLE+embedding0 둘 다** sidecar(mmap), 출력 bit-exact, 상주 최소 |
| `gemma4-side` | qwen3_e2e/gemma4_qs4cx_fp16_side | models/gemma4_side | 〃 — PLE+embedding0 sidecar, 출력 bit-exact |
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
| gauss4-ple | 2140/16.7 (**peak 3616→2662MB**) | 5193/60 (peak 6071→4879MB) | 1252/18.5 | — |
| gemma4-ple | 2595/16.6 (**peak 3784→2152MB**) | (coh 검증) | 2437/21.7 (peak 5497→4274MB) | — |
| gauss4-side | 2163/18.5 (**peak 2285MB**) | 5064/61 (peak 4501MB) | 1258/19.1 (peak 4908MB) | — |
| gemma4-side | 2760/17.2 (**peak 1842MB**) | (coh 검증, peak 3488MB) | 2390/22.1 (peak 3913MB) | — |
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
- **PLE sidecar**: `nntr_quantize <이미-양자화된 모델dir> --fc_dtype <동일> --embd_dtype <동일> --lmhead_dtype <동일> --ple_sidecar -o <새dir>` = 무재양자화 repack (dtype 일치 → passthrough, old bin = new bin + ple.bin 바이트 일치). 런타임 키는 nntr_config.json `ple_file_name`(= 매니페스트 .json). mmap(MADV_RANDOM)+prefill시 배치 row MADV_WILLNEED 선인출 — 선인출 없으면 cold prefill이 ~25% 느려짐(랜덤 major fault 직렬화). 출력은 in-bin과 bit-exact (같은 GGML row를 같은 dequant로 읽음).
- **embedding0 sidecar** (`--embd_sidecar`, 런타임 키 `embedding_file_name`): **untied lm_head(lmhead_untie=true) 전용** — tied lm_head는 decode마다 테이블 전 row를 스캔하므로 sidecar 이득 0. untie 시 embedding0가 tie_word_embeddings → embedding_layer로 전환됨(lookup 코드는 거울상이라 bit-exact; 전환은 gemma4/gauss4 한정). `--ple_sidecar`와 한 번에 조합 가능. **함정(해결됨)**: CUDA dev-act staging 버퍼가 함수-static이어서 embedding0·PLE가 같은 클래스가 되자 공유→레이스(가비지) — 인스턴스 멤버로 분리 완료. UVM 풀도 device-only 판정이라 minimal env에서도 staging 경로가 쓰임.
- 단말 peak RSS는 SVM 풀 성장 순서에 따라 수백 MB 편차 — gauss4-ple가 base보다 높게 찍히기도 함. **PLE 상주 아님이 실측으로 확정**: 1K prefill 중 /proc/smaps에서 PLE 매핑 Rss = 945MB 중 **3.5MB**(터치된 ~1023 rows뿐). 메모리 비교는 x86 수치가 깨끗함.

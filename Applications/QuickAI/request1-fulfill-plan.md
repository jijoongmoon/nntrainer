# Quick.AI Structured Chat/Session API 이행 계획

> 출처: `request-mail1.md` + `request-mail2.md`
> 작성일: 2026-04-13
> 상태: Draft

---

## 0. 요구사항 요약

| # | 요구사항 | 출처 |
|---|---------|------|
| R1 | Structured Chat Session API (`openChatSession`, run/stream/cancel/close) | mail1 §1 |
| R2 | `LoadModelRequest.maxNumTokens` (load-time 설정) | mail1 §2 |
| R3 | Structured Chat Sampling Config (`temperature`, `topK`, `topP`, `minP`, `maxTokens`, `seed`) | mail1 §3 |
| R4 | Structured Multimodal Support (turn 단위 text+image, part 순서 보존) | mail1 §4 |
| R5 | Structured Cancellation API (`cancel()`) | mail1 §5 |
| R6 | `enable_thinking` 지원 (`chatTemplateKwargs`) | mail2 |

### 확정된 설계 결정사항

| 항목 | 결정 |
|------|------|
| 세션 다중성 | 엔진당 단일 chat session (LiteRT-LM Conversation 제약). 기존 세션을 닫아야 새 세션 생성 가능 |
| 히스토리 관리 | 백엔드가 히스토리를 누적 관리 |
| Conversation Rebuild | 별도 `rebuild()` API 제공 |
| 이미지 안정성 | SHA-256 해시 기반 컨텐츠 비교, unload/close 시 저장 이미지 삭제 |
| NativeQuickDotAI | 모든 chat API에 대해 dummy 구현 (UNSUPPORTED 반환) |
| REST 엔드포인트 | `/v1/models/{id}/chat/*` 구조 |
| enable_thinking | 양 백엔드 적용, Native는 dummy |

---

## 1. 아키텍처 설계

### 1.1 새로운 타입 계층

```
QuickAiChatRole (enum)
  ├─ SYSTEM
  ├─ USER
  └─ ASSISTANT

QuickAiChatContentPart (sealed class)
  ├─ Text(text: String)
  ├─ ImageFile(absolutePath: String)
  └─ ImageBytes(bytes: ByteArray)

QuickAiChatMessage
  ├─ role: QuickAiChatRole
  └─ parts: List<QuickAiChatContentPart>

QuickAiChatSamplingConfig
  ├─ temperature: Double?
  ├─ topK: Int?
  ├─ topP: Double?
  ├─ minP: Double?
  ├─ maxTokens: Int?
  └─ seed: Int?

QuickAiChatTemplateKwargs
  └─ enableThinking: Boolean?

QuickAiChatSessionConfig
  ├─ sampling: QuickAiChatSamplingConfig?
  └─ chatTemplateKwargs: QuickAiChatTemplateKwargs?

QuickAiChatResult
  ├─ content: String            // assistant 응답 텍스트
  └─ metrics: PerformanceMetrics?
```

### 1.2 Chat Session 인터페이스

```kotlin
interface QuickAiChatSession : AutoCloseable {
    val sessionId: String

    /** 새 메시지를 보내고 assistant 응답을 받음 (히스토리 자동 누적) */
    fun run(messages: List<QuickAiChatMessage>): BackendResult<QuickAiChatResult>

    /** 스트리밍 방식으로 메시지를 보냄 (히스토리 자동 누적) */
    fun runStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<QuickAiChatResult>

    /** 진행 중인 생성을 취소 */
    fun cancel()

    /** 히스토리를 교체하고 세션 상태를 재구성 */
    fun rebuild(messages: List<QuickAiChatMessage>): BackendResult<Unit>

    /** 세션을 닫고 리소스(캐시 이미지 포함) 해제 */
    override fun close()
}
```

### 1.3 QuickDotAI 인터페이스 확장

```kotlin
interface QuickDotAI {
    // ... 기존 API 유지 ...

    /** 새 chat session을 연다 (엔진당 단일 세션, 기존 세션 있으면 BAD_REQUEST) */
    fun openChatSession(
        config: QuickAiChatSessionConfig? = null
    ): BackendResult<QuickAiChatSession>
}
```

### 1.4 이미지 저장소 (ImageStore)

```
ImageStore
  ├─ store(bytes: ByteArray): String       // SHA-256 해시 반환, 내부 캐시 저장
  ├─ store(absolutePath: String): String   // 파일 읽어서 해시, 캐시 저장
  ├─ get(hash: String): ByteArray?         // 해시로 이미지 조회
  ├─ contains(hash: String): Boolean
  └─ clear()                               // 모든 캐시 이미지 삭제
```

- 세션별로 `ImageStore` 인스턴스를 보유
- `QuickAiChatSession.close()` 또는 `QuickDotAI.unload()` 시 `clear()` 호출
- 히스토리 내 이미지는 해시로 참조하여, 동일 이미지가 다른 경로로 들어와도 동일하게 취급

### 1.5 REST API 엔드포인트

| Endpoint | Method | Request Body | Response | 설명 |
|----------|--------|-------------|----------|------|
| `/v1/models/{id}/chat/open` | POST | `ChatOpenRequest` (config?) | `ChatOpenResponse` (sessionId) | 세션 열기 |
| `/v1/models/{id}/chat/run` | POST | `ChatRunRequest` (sessionId, messages) | `ChatRunResponse` (content, metrics?) | 동기 실행 |
| `/v1/models/{id}/chat/run_stream` | POST | `ChatRunRequest` (sessionId, messages) | NDJSON stream | 스트리밍 실행 |
| `/v1/models/{id}/chat/cancel` | POST | `ChatCancelRequest` (sessionId) | `ChatCancelResponse` | 취소 |
| `/v1/models/{id}/chat/rebuild` | POST | `ChatRebuildRequest` (sessionId, messages) | `ChatRebuildResponse` | 히스토리 재구성 |
| `/v1/models/{id}/chat/close` | POST | `ChatCloseRequest` (sessionId) | `ChatCloseResponse` | 세션 닫기 |

### 1.6 NDJSON 스트리밍 포맷 (chat 전용)

```json
{"type":"delta","text":"Hello"}
{"type":"delta","text":" world"}
{"type":"done","duration_ms":1234}
```
> 기존 run_stream과 동일한 포맷 유지. `enable_thinking` 활성화 시에도 응답은 assistant content text만 반환.

---

## 2. 구현 단계

### Phase 1: 데이터 타입 & 인터페이스 정의

**목표**: 새로운 타입과 인터페이스를 정의하여 컴파일 가능한 상태 만들기

**대상 파일**:
- `QuickDotAI/src/main/java/com/example/quickdotai/Types.kt`
- `QuickDotAI/src/main/java/com/example/quickdotai/QuickDotAI.kt`

**작업 항목**:
- [ ] `QuickAiChatRole` enum 추가
- [ ] `QuickAiChatContentPart` sealed class 추가
- [ ] `QuickAiChatMessage` data class 추가
- [ ] `QuickAiChatSamplingConfig` data class 추가
- [ ] `QuickAiChatTemplateKwargs` data class 추가
- [ ] `QuickAiChatSessionConfig` data class 추가
- [ ] `QuickAiChatResult` data class 추가
- [ ] `LoadModelRequest`에 `maxNumTokens: Int? = null` 필드 추가
- [ ] `QuickAiChatSession` interface 정의 (run, runStreaming, cancel, rebuild, close)
- [ ] `QuickDotAI` interface에 `openChatSession()` 메서드 추가 (default = UNSUPPORTED)

**의존성**: 없음 (첫 번째 단계)

---

### Phase 2: ImageStore 구현

**목표**: 해시 기반 이미지 캐시/비교 시스템

**대상 파일**:
- `QuickDotAI/src/main/java/com/example/quickdotai/ImageStore.kt` (신규)

**작업 항목**:
- [ ] `ImageStore` 클래스 구현
  - SHA-256 해시 계산
  - `ConcurrentHashMap<String, ByteArray>` 기반 인메모리 캐시
  - `store(ByteArray)`, `store(absolutePath)` → 해시 반환
  - `get(hash)`, `contains(hash)`
  - `clear()` — 모든 캐시 삭제
- [ ] 스레드 안전성 보장 (세션별 인스턴스이므로 단일 워커에서 접근, 방어적 동기화)

**의존성**: Phase 1 (타입 정의)

---

### Phase 3: LiteRTLm Chat Session 구현

**목표**: LiteRT-LM 백엔드에 완전한 structured chat session 구현

**대상 파일**:
- `QuickDotAI/src/main/java/com/example/quickdotai/LiteRTLm.kt`
- `QuickDotAI/src/main/java/com/example/quickdotai/LiteRTLmChatSession.kt` (신규)

**작업 항목**:

#### 3a. 세션 관리
- [ ] `LiteRTLmChatSession` 클래스 구현 (`QuickAiChatSession` 구현체)
- [ ] 세션별 고유 `sessionId` 생성 (UUID)
- [ ] 세션별 `Conversation` 객체 보유
- [ ] 세션별 `ImageStore` 인스턴스 보유
- [ ] 세션별 히스토리 `MutableList<QuickAiChatMessage>` 관리

#### 3b. run / runStreaming
- [ ] `run(messages)` 구현
  - 새 messages를 히스토리에 추가
  - `QuickAiChatMessage` → LiteRT-LM `Contents` 변환
  - 이미지는 `ImageStore`에 저장하고 해시로 히스토리 참조
  - `conversation.sendMessage()` 호출
  - assistant 응답을 히스토리에 추가
  - `QuickAiChatResult` 반환
- [ ] `runStreaming(messages, sink)` 구현
  - 위와 동일하되 `conversation.sendMessageAsync()` + `MessageCallback` 사용
  - 기존 LiteRTLm의 delta 추출 로직 재활용

#### 3c. cancel
- [ ] `cancel()` 구현
  - 진행 중인 `sendMessageAsync`를 취소할 수 있는 메커니즘
  - LiteRT-LM `Conversation`의 cancel 지원 여부에 따라 구현 (지원하지 않으면 latch interrupt 방식)

#### 3d. rebuild
- [ ] `rebuild(messages)` 구현
  - 기존 `Conversation` 객체를 close
  - 새 `Conversation` 객체 생성
  - 전달받은 messages를 새 히스토리로 설정
  - 이전 히스토리에만 존재하던 이미지 해시 참조는 `ImageStore`에서 제거
  - 새 messages의 이미지를 `ImageStore`에 재등록

#### 3e. close
- [ ] `close()` 구현
  - `Conversation` 객체 close
  - `ImageStore.clear()` 호출
  - 히스토리 초기화

#### 3f. Sampling Config 적용
- [ ] `QuickAiChatSamplingConfig` → LiteRT-LM engine/conversation 설정 매핑
  - LiteRT-LM이 내부적으로 honor하지 못하는 필드도 wrapper에서는 수용

#### 3g. enable_thinking 적용
- [ ] `QuickAiChatTemplateKwargs.enableThinking` → LiteRT-LM template/prompt 렌더링에 반영

#### 3h. maxNumTokens 적용
- [ ] `LoadModelRequest.maxNumTokens` → `Engine` 생성 시 전달

#### 3i. LiteRTLm에 openChatSession 구현
- [ ] `LiteRTLm.openChatSession(config)` 구현
  - 새 `LiteRTLmChatSession` 생성
  - 단일 `activeSession` 변수로 관리 (기존 세션 있으면 BAD_REQUEST)
  - unload/close 시 활성 세션 close

**의존성**: Phase 1, Phase 2

---

### Phase 4: NativeQuickDotAI Dummy 구현

**목표**: NativeQuickDotAI에 chat session API의 dummy 스텁 제공

**대상 파일**:
- `QuickDotAI/src/main/java/com/example/quickdotai/NativeQuickDotAI.kt`
- `QuickDotAI/src/main/java/com/example/quickdotai/NativeChatSession.kt` (신규)

**작업 항목**:
- [ ] `NativeChatSession` 클래스 구현
  - `run()` → `BackendResult.Err(UNSUPPORTED, "Chat session not supported for native backend")`
  - `runStreaming()` → `sink.onError(UNSUPPORTED, ...)` + `BackendResult.Err`
  - `cancel()` → no-op
  - `rebuild()` → `BackendResult.Err(UNSUPPORTED, ...)`
  - `close()` → no-op
- [ ] `NativeQuickDotAI.openChatSession(config)` 구현
  - `NativeChatSession` 인스턴스 생성 반환
  - 또는: dummy이므로 `BackendResult.Err(UNSUPPORTED)` 반환도 고려
  - **결정**: 세션 객체는 생성하되 run/rebuild에서 UNSUPPORTED 반환 (향후 확장 용이)

**의존성**: Phase 1

---

### Phase 5: Service Layer (REST API)

**목표**: REST 엔드포인트를 통해 chat session API 노출

**대상 파일**:
- `LauncherApp/.../service/Protocol.kt`
- `LauncherApp/.../service/RequestDispatcher.kt`
- `LauncherApp/.../service/HttpServer.kt`
- `LauncherApp/.../service/ModelWorker.kt`
- `LauncherApp/.../service/ModelRegistry.kt`

**작업 항목**:

#### 5a. Protocol 확장
- [ ] Chat 관련 Request/Response DTO 추가
  - `ChatOpenRequest`, `ChatOpenResponse`
  - `ChatRunRequest`, `ChatRunResponse` (messages는 Base64 이미지 인코딩)
  - `ChatCancelRequest`, `ChatCancelResponse`
  - `ChatRebuildRequest`, `ChatRebuildResponse`
  - `ChatCloseRequest`, `ChatCloseResponse`
- [ ] `Request` sealed class에 chat 관련 variant 추가
  - `ChatOpen`, `ChatRun`, `ChatRunStream`, `ChatCancel`, `ChatRebuild`, `ChatClose`
- [ ] `StreamFrame` 은 기존 포맷 그대로 사용

#### 5b. HttpServer 라우팅
- [ ] `/v1/models/{id}/chat/open` → POST → `Request.ChatOpen`
- [ ] `/v1/models/{id}/chat/run` → POST → `Request.ChatRun`
- [ ] `/v1/models/{id}/chat/run_stream` → POST → `Request.ChatRunStream`
- [ ] `/v1/models/{id}/chat/cancel` → POST → `Request.ChatCancel`
- [ ] `/v1/models/{id}/chat/rebuild` → POST → `Request.ChatRebuild`
- [ ] `/v1/models/{id}/chat/close` → POST → `Request.ChatClose`

#### 5c. ModelWorker 확장
- [ ] 새 Job 타입 추가
  - `Job.ChatOpen(config, onResult)`
  - `Job.ChatRun(sessionId, messages, onResult)`
  - `Job.ChatRunStream(sessionId, messages, sink)`
  - `Job.ChatCancel(sessionId, onResult)`
  - `Job.ChatRebuild(sessionId, messages, onResult)`
  - `Job.ChatClose(sessionId, onResult)`
- [ ] `processJob()` 에서 각 Job 처리 로직 구현
- [ ] 세션 ID → `QuickAiChatSession` 매핑 관리

#### 5d. RequestDispatcher 핸들러
- [ ] `handleChatOpen()`, `handleChatRun()`, `handleChatRunStream()`, `handleChatCancel()`, `handleChatRebuild()`, `handleChatClose()` 구현
- [ ] 세션 ID 유효성 검증
- [ ] 적절한 HTTP 상태 코드 반환 (404 = unknown session 등)

#### 5e. ModelRegistry 변경
- [ ] unload 시 해당 모델의 모든 chat session close 처리 보장

**의존성**: Phase 1, Phase 3, Phase 4

---

### Phase 6: Client 업데이트

**목표**: clientapp의 QuickAiClient에 chat session 엔드포인트 호출 메서드 추가

**대상 파일**:
- `clientapp/.../api/Models.kt`
- `clientapp/.../api/QuickAiClient.kt`

**작업 항목**:
- [ ] Client-side DTO 추가 (독립 복사본 — 기존 패턴 유지)
  - `ChatOpenRequest`, `ChatOpenResponse`
  - `ChatRunRequest`, `ChatRunResponse`
  - `ChatCancelRequest`, `ChatCancelResponse`
  - `ChatRebuildRequest`, `ChatRebuildResponse`
  - `ChatCloseRequest`, `ChatCloseResponse`
- [ ] `QuickAiClient`에 메서드 추가
  - `chatOpen(modelKey, config?)`: `ApiResult<ChatOpenResponse>`
  - `chatRun(modelKey, sessionId, messages)`: `ApiResult<ChatRunResponse>`
  - `chatRunStream(modelKey, sessionId, messages)`: `Flow<StreamChunk>`
  - `chatCancel(modelKey, sessionId)`: `ApiResult<ChatCancelResponse>`
  - `chatRebuild(modelKey, sessionId, messages)`: `ApiResult<ChatRebuildResponse>`
  - `chatClose(modelKey, sessionId)`: `ApiResult<ChatCloseResponse>`

**의존성**: Phase 5 (엔드포인트 확정 후)

---

### Phase 7: 문서 업데이트

**목표**: Architecture.md, AsyncAndStreaming.md 등 기존 문서에 chat session 관련 내용 반영

**작업 항목**:
- [ ] `Architecture.md`에 chat session 아키텍처 섹션 추가
- [ ] `AsyncAndStreaming.md`에 chat streaming 동작 설명 추가
- [ ] 새 REST 엔드포인트 사양 문서화

**의존성**: Phase 5, Phase 6

---

## 3. 파일별 변경 요약

| 파일 | 변경 유형 | Phase |
|------|----------|-------|
| `QuickDotAI/.../Types.kt` | 수정 — 새 데이터 클래스 추가 + LoadModelRequest 필드 추가 | 1 |
| `QuickDotAI/.../QuickDotAI.kt` | 수정 — ChatSession 인터페이스 + openChatSession 추가 | 1 |
| `QuickDotAI/.../ImageStore.kt` | **신규** — 해시 기반 이미지 캐시 | 2 |
| `QuickDotAI/.../LiteRTLm.kt` | 수정 — openChatSession 구현, maxNumTokens 적용 | 3 |
| `QuickDotAI/.../LiteRTLmChatSession.kt` | **신규** — LiteRT-LM chat session 구현체 | 3 |
| `QuickDotAI/.../NativeQuickDotAI.kt` | 수정 — openChatSession dummy 구현 | 4 |
| `QuickDotAI/.../NativeChatSession.kt` | **신규** — Native dummy chat session | 4 |
| `LauncherApp/.../Protocol.kt` | 수정 — chat DTO 추가 | 5 |
| `LauncherApp/.../HttpServer.kt` | 수정 — chat 라우팅 추가 | 5 |
| `LauncherApp/.../RequestDispatcher.kt` | 수정 — chat 핸들러 추가 | 5 |
| `LauncherApp/.../ModelWorker.kt` | 수정 — chat Job 타입 + 처리 로직 | 5 |
| `LauncherApp/.../ModelRegistry.kt` | 수정 — unload 시 session cleanup | 5 |
| `clientapp/.../Models.kt` | 수정 — chat DTO 추가 | 6 |
| `clientapp/.../QuickAiClient.kt` | 수정 — chat 메서드 추가 | 6 |
| `Architecture.md` | 수정 — 문서 업데이트 | 7 |
| `AsyncAndStreaming.md` | 수정 — 문서 업데이트 | 7 |

---

## 4. 리스크 & 고려사항

### 4.1 LiteRT-LM Conversation 단일 인스턴스 제약
- LiteRT-LM `Engine` 하나에서 `Conversation`은 하나만 허용됨 (확인 완료)
- 해결: 엔진당 단일 세션만 허용, `openChatSession()` 시 기존 세션이 있으면 `BAD_REQUEST` 반환

### 4.2 Sampling Config 반영 범위
- LiteRT-LM이 `temperature`, `topK` 등을 실제로 honor하는지는 런타임 확인 필요
- Wrapper에서는 무조건 수용하되, 미지원 필드는 silent ignore (메일에서 이 방향 명시)

### 4.3 enable_thinking 동작
- LiteRT-LM의 chat template이 `enable_thinking` kwarg를 어떻게 처리하는지 확인 필요
- 모델(Gemma4)이 thinking을 지원하지 않으면 무시될 수 있음

### 4.4 이미지 메모리 사용
- `ImageStore`가 인메모리이므로 대량 이미지 히스토리 시 메모리 압박 가능
- 세션 close/rebuild 시 적극적으로 정리하여 완화

### 4.5 스레드 안전성
- `ModelWorker`가 단일 스레드로 Job을 처리하므로, 같은 모델의 세션들은 자연스럽게 직렬화됨
- `cancel()`만 외부 스레드에서 호출될 수 있으므로 해당 경로만 동기화 필요

---

## 5. 구현 순서 의존성 그래프

```
Phase 1 (타입/인터페이스)
    ├── Phase 2 (ImageStore)
    │       └── Phase 3 (LiteRTLm 구현)
    │                └── Phase 5 (Service Layer) ── Phase 6 (Client)
    └── Phase 4 (Native Dummy)                           │
                    └─────────────────────────────────────┘
                                                          └── Phase 7 (문서)
```

Phase 2와 Phase 4는 독립적이므로 병렬 진행 가능.

---

## 6. 완료 기준

- [ ] `openChatSession()` → 세션 열기 → `run(messages)` → assistant 응답 수신 (LiteRTLm)
- [ ] 엔진당 단일 세션 제약 확인 (두 번째 open 시 BAD_REQUEST 반환)
- [ ] 스트리밍 chat 동작 확인 (NDJSON delta 수신)
- [ ] `cancel()` 호출 시 진행 중 생성 중단 확인
- [ ] `rebuild(messages)` 호출 시 히스토리 교체 후 정상 동작 확인
- [ ] 동일 이미지를 다른 경로로 전송 시 ImageStore 해시 일치 확인
- [ ] 세션 close / 모델 unload 시 이미지 캐시 삭제 확인
- [ ] `maxNumTokens` 전달 시 모델 로드 정상 동작 확인
- [ ] sampling config 전달 시 정상 수용 확인
- [ ] `enable_thinking` 전달 시 정상 수용 확인
- [ ] NativeQuickDotAI에서 chat API 호출 시 UNSUPPORTED 에러 정상 반환 확인
- [ ] 모든 REST 엔드포인트 접근 및 응답 확인
- [ ] 기존 `run(prompt)`, `runStreaming()`, `runMultimodal()` API 하위 호환성 유지 확인

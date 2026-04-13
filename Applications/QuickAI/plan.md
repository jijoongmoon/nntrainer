# QuickAI Structured Chat Session API - 구현 계획

## 1. 개요

본 문서는 aistudio-mobile 데모앱 개발팀의 요청에 따라 Quick.AI API 확장을 위한 구현 계획을 정의합니다.

### 요청 배경
현재 aistudio-mobile에서 Quick.AI 기반 AI Chat 서비스를 연동 중이나, Quick.AI wrapper만으로 서비스 계약을 충족하기 어려워 일부 경로에서 LiteRT-LM API를 직접 사용하고 있음. 이를 Quick.AI만으로 통합하기 위한 API 확장 필요.

---

## 2. 요구사항 요약

| # | 요구사항 | 설명 |
|---|----------|------|
| 1 | Structured Chat Session API | system/user/assistant role을 유지하는 structured messages[] 기반 chat |
| 2 | maxNumTokens Load-time 옵션 | LiteRT 경로에서 load-sensitive 설정으로 동작하는 max_tokens |
| 3 | Sampling Config | temperature, top_k, top_p 등 request/session sampling 설정 |
| 4 | Structured Multimodal Support | chat turn 단위로 multimodal input 처리 |
| 5 | Cancellation API | active streaming generation 취소 |
| 6 | enable_thinking | chat_template_kwargs를 통한 thinking 모드 제어 |

---

## 3. API 설계

### 3.1 새로운 데이터 타입 (Types.kt)

```kotlin
// ==================== Chat Session Types ====================

/**
 * @brief Chat message role - mirrors conversation participant types
 */
@Serializable
enum class QuickAiChatRole {
    SYSTEM,
    USER,
    ASSISTANT
}

/**
 * @brief Content part within a chat message - supports text and multimodal
 */
sealed class QuickAiChatContentPart {
    @Serializable
    data class Text(val text: String) : QuickAiChatContentPart()

    data class ImageFile(val absolutePath: String) : QuickAiChatContentPart()

    data class ImageBytes(val bytes: ByteArray) : QuickAiChatContentPart() {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is ImageBytes) return false
            return bytes.contentEquals(other.bytes)
        }
        override fun hashCode(): Int = bytes.contentHashCode()
    }
}

/**
 * @brief Single message in a structured chat conversation
 */
data class QuickAiChatMessage(
    val role: QuickAiChatRole,
    val parts: List<QuickAiChatContentPart>
)

/**
 * @brief Sampling configuration for chat session/run
 */
@Serializable
data class QuickAiChatSamplingConfig(
    val temperature: Double? = null,
    @SerialName("top_k")
    val topK: Int? = null,
    @SerialName("top_p")
    val topP: Double? = null,
    @SerialName("min_p")
    val minP: Double? = null,
    @SerialName("max_tokens")
    val maxTokens: Int? = null,
    val seed: Int? = null
)

/**
 * @brief Chat template keyword arguments for prompt control
 *
 * enableThinking: Controls thinking/reasoning mode in supported models
 *                 (e.g., Qwen3, DeepSeek-R1 style reasoning)
 */
@Serializable
data class QuickAiChatTemplateKwargs(
    @SerialName("enable_thinking")
    val enableThinking: Boolean? = null
)

/**
 * @brief Configuration for chat session creation
 */
@Serializable
data class QuickAiChatSessionConfig(
    val sampling: QuickAiChatSamplingConfig? = null,
    @SerialName("chat_template_kwargs")
    val chatTemplateKwargs: QuickAiChatTemplateKwargs? = null
)

/**
 * @brief Result from chat session run
 */
data class QuickAiChatResult(
    val content: String,
    val metrics: PerformanceMetrics? = null
)
```

### 3.2 LoadModelRequest 확장

```kotlin
@Serializable
data class LoadModelRequest(
    val backend: BackendType = BackendType.GPU,
    val model: ModelId,
    val quantization: QuantizationType = QuantizationType.W4A32,
    @SerialName("model_path") val modelPath: String? = null,
    @SerialName("vision_backend") val visionBackend: BackendType? = null,
    @SerialName("cache_dir") val cacheDir: String? = null,

    /**
     * Maximum number of tokens for the loaded model context.
     * LiteRT-LM treats this as a load-time setting rather than request-time.
     */
    @SerialName("max_num_tokens")
    val maxNumTokens: Int? = null
)
```

### 3.3 QuickDotAI 인터페이스 확장

```kotlin
/**
 * @brief Chat session interface for structured multi-turn conversations
 */
interface QuickAiChatSession : AutoCloseable {
    /**
     * @brief Blocking structured chat inference
     */
    fun run(messages: List<QuickAiChatMessage>): BackendResult<QuickAiChatResult>

    /**
     * @brief Streaming structured chat inference
     */
    fun runStreaming(
        messages: List<QuickAiChatMessage>,
        sink: StreamSink
    ): BackendResult<Unit>

    /**
     * @brief Cancel active streaming generation
     */
    fun cancel()

    /**
     * @brief Close session and release resources
     */
    override fun close()
}

/**
 * @brief Common interface for QuickDotAI engines
 */
interface QuickDotAI {
    // ... 기존 메서드들 ...

    /**
     * @brief Open a new chat session for structured conversations
     *
     * The session maintains conversation history and allows:
     * - Structured messages with system/user/assistant roles
     * - Per-session sampling configuration
     * - Chat template kwargs (e.g., enable_thinking)
     * - Streaming with cancellation support
     */
    fun openChatSession(
        config: QuickAiChatSessionConfig? = null
    ): BackendResult<QuickAiChatSession>
}
```

---

## 4. 구현 단계

### Phase 1: 타입 정의 (Types.kt)
- [ ] QuickAiChatRole enum 추가
- [ ] QuickAiChatContentPart sealed class 추가
- [ ] QuickAiChatMessage data class 추가
- [ ] QuickAiChatSamplingConfig data class 추가
- [ ] QuickAiChatTemplateKwargs data class 추가
- [ ] QuickAiChatSessionConfig data class 추가
- [ ] QuickAiChatResult data class 추가
- [ ] LoadModelRequest에 maxNumTokens 필드 추가

### Phase 2: 인터페이스 정의 (QuickDotAI.kt)
- [ ] QuickAiChatSession interface 추가
- [ ] QuickDotAI.openChatSession() 메서드 추가
- [ ] 기본 구현 (UNSUPPORTED 반환) 추가

### Phase 3: LiteRTLm 구현
- [ ] LiteRTLmChatSession 내부 클래스 구현
- [ ] openChatSession() 구현
- [ ] messages → LiteRT-LM Contents 변환 로직
- [ ] sampling config 적용 로직
- [ ] enable_thinking 적용 로직
- [ ] cancel() 구현
- [ ] runStreaming() 구현

### Phase 4: NativeQuickDotAI 구현
- [ ] QuickAiChatSession 기본 구현 (UNSUPPORTED 또는 간단한 래퍼)
- [ ] 향후 C API 확장을 위한 플레이스홀더

### Phase 5: C API 확장 (선택사항, 향후 진행)
- [ ] quick_dot_ai_api.h 확장
- [ ] QuickAiChatSessionHandle 및 관련 함수 추가
- [ ] JNI 바인딩 (NativeCausalLm.kt)
- [ ] NativeQuickDotAIChatSession 구현

---

## 5. 파일 수정 목록

| 파일 | 수정 내용 |
|------|----------|
| `QuickDotAI/src/main/java/com/example/quickdotai/Types.kt` | 새로운 타입 정의, LoadModelRequest 확장 |
| `QuickDotAI/src/main/java/com/example/quickdotai/QuickDotAI.kt` | QuickAiChatSession interface, openChatSession() 추가 |
| `QuickDotAI/src/main/java/com/example/quickdotai/LiteRTLm.kt` | Chat session 구현 |
| `QuickDotAI/src/main/java/com/example/quickdotai/NativeQuickDotAI.kt` | 기본 구현 또는 플레이스홀더 |

---

## 6. LiteRT-LM API 확인 필요사항

구현 전 확인이 필요한 LiteRT-LM API:

1. **Role 지원**: `Message.Role` 또는 유사 enum 존재 여부
2. **SamplingConfig**: temperature, top_k, top_p 등 설정 API
3. **Cancel**: streaming 취소 API 지원 여부
4. **Chat Template Kwargs**: enable_thinking 등 template 변수 전달 방식

```kotlin
// LiteRT-LM API 예시 (확인 필요)
interface Message {
    enum class Role { SYSTEM, USER, ASSISTANT }
    val role: Role
    val content: List<Content>
}

interface Conversation {
    fun sendMessage(message: Message): Message
    fun sendMessageAsync(message: Message, callback: MessageCallback)
    fun cancel()  // 존재 여부 확인 필요
}
```

---

## 7. 하위 호환성

- 기존 `run()`, `runStreaming()`, `runMultimodal()` API는 변경 없이 유지
- 새로운 `openChatSession()` API는 추가만 수행
- `LoadModelRequest.maxNumTokens`은 optional 필드로 추가 (기본값 null)

---

## 8. 테스트 계획

### Unit Tests
- [ ] QuickAiChatMessage → LiteRT Contents 변환 테스트
- [ ] Sampling config 적용 테스트
- [ ] enable_thinking 전달 테스트

### Integration Tests
- [ ] LiteRTLmChatSession 기본 동작 테스트
- [ ] Multi-turn conversation 테스트
- [ ] Streaming + cancel 테스트
- [ ] Multimodal chat 테스트

---

## 9. 구현 우선순위

| 순위 | 항목 | 난이도 | 의존성 |
|------|------|--------|--------|
| 1 | Types.kt 타입 정의 | 낮음 | 없음 |
| 2 | LoadModelRequest.maxNumTokens | 낮음 | 없음 |
| 3 | QuickDotAI 인터페이스 확장 | 낮음 | #1 |
| 4 | LiteRTLmChatSession (기본) | 중간 | #1, #2, #3 |
| 5 | Sampling config 적용 | 중간 | #4 |
| 6 | enable_thinking 적용 | 중간 | #4 |
| 7 | Cancel 구현 | 중간 | #4 |
| 8 | NativeQuickDotAI 구현 | 높음 | C API 확장 |

---

## 10. 참고 문서

- [Architecture.md](./Architecture.md) - QuickAI 전체 아키텍처
- [AsyncAndStreaming.md](./AsyncAndStreaming.md) - 스트리밍 설계
- [LiteRT-LM Guide](../../how-to-use-litert-lm-guide.md) - LiteRT-LM API 가이드

---

## 11. 변경 이력

| 날짜 | 버전 | 내용 |
|------|------|------|
| 2026-04-13 | 1.0 | 초기 계획 수립 - Structured Chat Session API, maxNumTokens, Sampling Config, enable_thinking |
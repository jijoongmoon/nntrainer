안녕하세요 aistudio-mobile 데모앱 개발 중인 안용주입니다.



현재 `aistudio-mobile`에서 Quick.AI 기반 AI Chat 서비스를 연동하면서, Quick.AI wrapper만으로는 현재 서비스 계약을 충족하기 어려워 일부 경로에서 LiteRT-LM API를 직접 사용하고 있습니다.



저희가 원하는 방향은 Quick.AI LiteRT 경로도 Quick.AI만을 통해 사용하도록 정리하는 것입니다. 다만 현재 공개된 Quick.AI API는 single prompt 실행 중심이라, 실제 AI Chat 서비스에서 필요한 structured chat/session 동작을 모두 표현하기 어렵습니다.



현재 서비스에서 특히 필요한 동작은 아래와 같습니다.



- `system` / `user` / `assistant` role을 유지하는 structured `messages[]` 기반 chat

- `session_config.backend` 및 load-sensitive `max_tokens` 같은 load-time 설정

- `temperature`, `top_k`, `top_p` 등 request/session sampling 설정

- append-only multi-turn 대화의 conversation reuse

- history 변경, sampling 변경, 실패/취소 이후 conversation rebuild

- multimodal image+text turn 처리

- active streaming generation cancel

- 동일 이미지가 다른 temporary file 경로로 들어와도 같은 history로 판단할 수 있는 stable image handling



이 요구사항이 Quick.AI에서 직접 지원되지 않다 보니, 현재는 host 쪽에서 별도의 structured chat layer와 direct LiteRT-LM integration을 유지하고 있습니다.



저희가 요청드리고 싶은 Quick.AI 추가 API는 아래와 같습니다.



### 1. Structured chat session API

Prompt 기반 `run(prompt)` 외에 chat session 기반 API가 필요합니다.



필요 기능:

- model load 이후 chat session open

- `system` / `user` / `assistant` role을 갖는 structured message 전송

- structured chat streaming

- active chat run cancel

- engine lifecycle과 별도의 chat session close



예시 shape:



```kotlin

data class QuickAiChatMessage(

    val role: QuickAiChatRole,

    val parts: List<QuickAiChatContentPart>

)



enum class QuickAiChatRole {

    SYSTEM,

    USER,

    ASSISTANT

}



sealed class QuickAiChatContentPart {

    data class Text(val text: String) : QuickAiChatContentPart()

    data class ImageFile(val absolutePath: String) : QuickAiChatContentPart()

    data class ImageBytes(val bytes: ByteArray) : QuickAiChatContentPart()

}



interface QuickAiChatSession : AutoCloseable {

    fun run(messages: List<QuickAiChatMessage>): BackendResult<QuickAiChatResult>

    fun runStreaming(

        messages: List<QuickAiChatMessage>,

        sink: StreamSink

    ): BackendResult<QuickAiChatResult>

    fun cancel()

    override fun close()

}



interface Quick.AI {

    fun openChatSession(

        config: QuickAiChatSessionConfig? = null

    ): BackendResult<QuickAiChatSession>

}

```



### 2. Load-time option에 `maxNumTokens` 추가

현재 Quick.AI LiteRT 경로에서는 `max_tokens`가 request-time hint가 아니라 load-sensitive 설정으로 동작합니다.



그래서 `LoadModelRequest` 수준에서 아래 field가 필요합니다.



```kotlin

data class LoadModelRequest(

    val backend: BackendType = BackendType.GPU,

    val model: ModelId,

    val quantization: QuantizationType = QuantizationType.W4A32,

    val modelPath: String? = null,

    val visionBackend: BackendType? = null,

    val cacheDir: String? = null,

    val maxNumTokens: Int? = null

)

```



### 3. Structured chat용 sampling config

Structured chat/session API에서 sampling 설정을 직접 받을 수 있어야 합니다.



```kotlin

data class QuickAiChatSessionConfig(

    val sampling: QuickAiChatSamplingConfig? = null

)



data class QuickAiChatSamplingConfig(

    val temperature: Double? = null,

    val topK: Int? = null,

    val topP: Double? = null,

    val minP: Double? = null,

    val maxTokens: Int? = null,

    val seed: Int? = null

)

```



LiteRT가 일부 field를 내부적으로 아직 fully honor하지 못하더라도, wrapper 수준에서는 이 설정을 명시적으로 받아줄 수 있으면 좋겠습니다.



### 4. Structured multimodal support

Single prompt part가 아니라 chat turn 단위로 multimodal input을 다룰 수 있어야 합니다.



필요 기능:

- text-only turn

- text+image turn

- turn 내부 part 순서 보존

- image file path와 image bytes 모두 지원



### 5. Structured cancellation API

현재는 active streaming generation 취소를 위해 host가 LiteRT conversation API를 직접 호출하고 있습니다.



이 부분을 Quick.AI chat session API에서 직접 제공해주시면 host 쪽 direct dependency를 제거할 수 있습니다.



=============================================



이러한 API들이 추가되면 host 쪽의 direct LiteRT-LM integration을 제거하고 Quick.AI만으로 통합 경로를 단순화할 수 있습니다.

LiteRT-LM 모델 뿐만 아니라 quick.ai native model (causal_lm) 의 경우도 위의 API 들이 지원돼야 LiteRT-LM 모델과 같은 수준으로 서빙할 수 있을 것 같습니다.



검토 부탁드립니다.

감사합니다.
안녕하세요.



앞서 전달드린 Quick.AI structured chat/session API 요청에 추가로, `enable_thinking` 지원도 함께 요청드리고자 합니다.



현재 저희 서비스 계약에서는 `enable_thinking`을 prompt control 성격의 옵션으로 다루고 있습니다. 즉, 클라이언트는 `chat_template_kwargs.enable_thinking` 값을 전달할 수 있어야 하고, runtime은 이를 prompt/template rendering 단계에 반영하되, 기존 응답 schema는 그대로 유지하는 것이 목표입니다.



저희가 필요한 동작은 아래와 같습니다:

- client가 `chat_template_kwargs.enable_thinking`를 request에 포함할 수 있어야 함

- Quick.AI가 이 값을 template/prompt construction 단계에 반영할 수 있어야 함

- JSON 응답과 SSE 응답 모두 현재와 동일하게 assistant `content` text만 반환하면 됨

- 별도의 reasoning field나 새로운 response channel은 현재 범위에서 필요하지 않음



아래와 같은 방식이 예상됩니다:

```kotlin

data class QuickAiChatTemplateKwargs(

    val enableThinking: Boolean? = null

)



data class QuickAiChatSessionConfig(

    val sampling: QuickAiChatSamplingConfig? = null,

    val chatTemplateKwargs: QuickAiChatTemplateKwargs? = null

)

```

또는 최소한 structured chat 실행 API에서 `chatTemplateKwargs["enable_thinking"]`를 전달할 수 있는 형태여도 괜찮습니다.



검토 부탁드립니다.

감사합니다.
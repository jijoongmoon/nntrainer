 Applications/QuickAI 는 안드로이드 앱을 빌드하는 root 디렉토리야 여기에 안드로이드 앱과 서비스 코드를 작성해면 돼

 QuickAIService는 안드로이드 서비스야 Applications/QuickAI/LauncherApp/src/main/java/com/example/QuickAI/QuickAIService.kt 여기에 구현되어 있어
 Launcher App 는 QuickAIService를 실행하는 앱이야 안드로이드에서는 백그라운드 서비스가 단독으로 실행될 수 없어 반드시 앱이 실행을 시켜야 해
 android:process=":remote" , android:exported="true"> 옵션을 통해서  QuickAIService를 실행시켜서 Launcher 앱 뿐만 아니라 다른 ClientApp들도 QuickAIService에 작업을 요청할 수 있도록 할꺼야
 Launcher App은 Applications/QuickAI/LauncherApp/src/main/java/com/example/QuickAI/LauncherApp.kt 여기에 구현되어 있어
 ClientApp은 Applications/QuickAI/clientapp 여기에 있어 QuickAIService와 통신하는 샘플용 클라이언트 앱이야 rest API를 통해서 QuickAIService와 요청과 결과를 주고받아

 일단 필요한 구현 내용은
 QuickAIService라는 서비스가 실행되면 ClientApp이 QuickAIService와 3453 포트를 통해서 REST API로 통신할 수 있어야 해(로컬로 디바이스 안에서 통신)
    만약 지정 포트를 사용하는 방법이 안드로이드에서 금지하거나 이런 방식으로 작동시킬수 없다면 다른 좋은방법을 찾아줘

    그리고 ClientApp이 호출하는 REST API는 libcausallm_api.so 에 있는 api 여야 해
    libcausallm_api.so 에서 제공하는 api는 Applications/CausalLM/api/causal_lm_api.h 여기에 선언되어 있어

    causal_lm_api는 nntrainer 엔진을 통해서 AI 작업을 수행해

    REST API 호출은 libcausallm_api.so의 API와 1:1 로 맵핑이 되어야 하고 libcausallm_api.so 에서 실행한 결과를 ClientApp 이 kotlin레벨에서 받아볼 수 있어야 해

    추가로 특정모델(gemma4)에 대해서만 litert-lm(https://github.com/google-ai-edge/LiteRT-LM) 을 통해서 처리하도록 할꺼야 Litert-lm은 코틀린 레벨의 API를 제공하고 있으니 kotlin레벨에서 요청을 routing 해서 litert-lm에서 처리하도록 하면 될것 같아

    위와 같은 요구사항에 따라 구현을 하려고해 좋은 구조를 설계하는데 필요한 것들을 같이 계획하고 코드를 작성해 보자

    먼저 계획을 세우고 Architecture.md 등의 파일을 작성한 후에 코드 작성을 시작해줘

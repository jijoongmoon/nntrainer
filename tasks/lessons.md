# Lessons (자기 교정 기록)

## 파일 작성 규칙

- **파일은 항상 작은 chunk로 나눠서 작성한다.**
  - 처음 생성 시: 최소한의 헤더/틀만 `Write` 로 생성
  - 나머지 섹션은 `Edit` 로 한 섹션씩 추가
  - 한 번의 큰 `Write` 로 전체 파일을 작성하지 않는다.
  - 이유: 사용자가 중간 단계에서 확인/중단/수정할 수 있도록.
  - 기록일: 2026-04-24 (CLAUDE.md 및 tasks/todo.md 작성 시 지적받음)

## 커밋 규칙

- **모든 커밋의 author / committer 는 `Jijoong Moon <jijoong.moon@samsung.com>`** 으로 한다.
  - 이 저장소 규칙상 `git config` 를 수정할 수 없으므로, 각 `git commit` 호출 시 env var로 지정:
    ```bash
    GIT_AUTHOR_NAME="Jijoong Moon" GIT_AUTHOR_EMAIL="jijoong.moon@samsung.com" \
    GIT_COMMITTER_NAME="Jijoong Moon" GIT_COMMITTER_EMAIL="jijoong.moon@samsung.com" \
    git commit -m "..."
    ```
  - 또는 `--author='Jijoong Moon <jijoong.moon@samsung.com>'`. env var 방식이 committer까지 한 번에 맞춰서 더 안전.
  - 이미 만들어진 커밋의 author 가 틀린 경우: `git reset --soft` 로 풀어서 env var 지정 후 재커밋 → `push --force-with-lease`.
  - 기록일: 2026-04-24 (B1 PR 커밋 시 지적받음)

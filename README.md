# LLM Bot
## 설치방법
### .env 파일 생성
프로젝트 root 디렉토리에 아래 파일을 생성한다.
```
vi .env
```
### .env 파일 수정
```bash
# DEBUG
DEBUG='true'

# 테스트 서버 정보
SERVER_HOST='localhost'
SERVER_PORT=5000

# LLM Model
GPT_MODEL="gpt-4o-mini"

# OPENAI API KEY
OPENAI_API_KEY=YOUR_OPENAI_API_KEY

# LangSmith
LANGCHAIN_PROJECT=YOUR_LANGCHAIN_PROJECT
LANGSMITH_API_KEY=YOUR_LANGSMITH_API_KEY
```
### 패키지 설치
```
pip install -r requirements.txt
```

## 실행방법

### 직접 실행

```
python web_server.py
```

### Docker에서 실행

#### .env 파일 생성

```bash
cd docker
vi .env
```

#### .env 파일 수정

```bash
# 외부에서 접근할 포트번호
WEB_PORT=5000

# 기타 환경변수
ENVIRONMENT=prod
COMPOSE_PROJECT_NAME=prod-llm-bot
```

#### 이미지 빌드

```bash
bash docker-build.sh
```

#### 컨테이너 실행

```bash
bash docker-run.sh
```

#### 컨테이너 종료

```bash
bash docker-stop.sh
```

## API 정의

### 요청 URL

아래와 같은 형태로 요청

```
http://localhost:5000/api/chat
```

### 채팅 응답 요청

- **URL**: `/api/chat`

- **Method**: `POST`

- **Path Parameters**:  

  - - `session_id`: 채팅에 대한 고유 값
      - `session_id` 값을 기반으로 내부적으로 대화 이력을 저장함
    - `text`: 사용자의 입력 텍스트

- **Response**: 

  - **Type**: `Content-Type: application/json`

  - **data**:

    - `answer`: AI Bot의 응답 텍스트

    - `category`: 사용자의 대화 의도에 대한 카테고리 값
      - **카테고리 종류:** 진로탐색, 자아탐색, 학업, 입시, 고교학점제

      - `is_finished`가 true로 반환될 때 `category`값을 함께 반환함

    - `history`: `session_id`에 의해 저장되고 있는 대화의 이력

    - `is_finished`: 대화가 종료되었는지 여부

  - **응답 데이터 예시**

    ```json
    {
        "answer": "안녕! 잘 지냈어? 무슨 고민이 있어서 찾아왔니?",
        "category": null,
        "history": [
            {
                "content": "안녕하세요",
                "role": "User"
            },
            {
                "content": "안녕! 잘 지냈어? 무슨 고민이 있어서 찾아왔니?",
                "role": "AI"
            }
        ],
        "is_finished": false
    }
    ```

    

## 참고자료

- OpenAI - https://platform.openai.com/

- LangSmith - https://www.langchain.com/langsmith

- [<랭체인LangChain 노트> - LangChain 한국어 튜토리얼](https://wikidocs.net/book/14314)

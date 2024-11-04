# LLM Bot
## 설치방법
### .env 파일 생성
프로젝트 root 디렉토리에 아래 파일을 생성한다.
```
vi .env
```
### .env 파일 수정
```
# DEBUG
DEBUG='true'

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
```
python main.py
```


## OpenAI
https://platform.openai.com/

## LangSmith
https://www.langchain.com/langsmith


import os
from dotenv import load_dotenv

load_dotenv()


def str_to_bool(value):
    return value.lower() in ('true', 't', 'yes', 'y', '1')


# DEBUG
DEBUG = str_to_bool(os.getenv('DEBUG', 'false'))

PROTOCOL = 'http'

# 테스트 서버 정보
SERVER_HOST = os.getenv('SERVER_HOST')
SERVER_PORT = os.getenv('SERVER_PORT')
SERVER_URL = f'{PROTOCOL}://{SERVER_HOST}:{SERVER_PORT}/'

# LLM Model
GPT_MODEL = os.getenv('GPT_MODEL')

# OPENAI API KEY
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# LangSmith
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')
LANGSMITH_API_KEY = os.getenv('LANGSMITH_API_KEY')

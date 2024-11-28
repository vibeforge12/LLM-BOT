import os

from config import *

# 애플리케이션 루트 디렉토리
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'uploads'

# Set OpenAI environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")

if not openai_api_key:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Set Langsmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY  # Update to your API key

# Hugging Face Transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
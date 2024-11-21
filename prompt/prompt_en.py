# 사용자가 입력한 질문에 대해 새로운 질문으로 재작성하는 프롬프트
CONTEXT_Q_SYSTEM_PROMPT = """
Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""

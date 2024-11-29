import random
import string
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage


def generate_session_id():
    # 세션 아이디 생성
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
    session_id = f"{date_str}_{rand_str}"

    return session_id


def reverse_history_role(chat_history):
    reversed_history = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            reversed_history.append(AIMessage(message.content))
        elif isinstance(message, AIMessage):
            reversed_history.append(HumanMessage(message.content))
    return reversed_history

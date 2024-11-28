from langchain_core.messages import HumanMessage, AIMessage


def reverse_history_role(chat_history):
    reversed_history = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            reversed_history.append(AIMessage(message.content))
        elif isinstance(message, AIMessage):
            reversed_history.append(HumanMessage(message.content))
    return reversed_history

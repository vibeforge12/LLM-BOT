import gradio as gr
import time

from chat_example import *

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    config = {"configurable": {"thread_id": "abc123"}}

    def respond(message, chat_history):
        input_messages = [HumanMessage(message)]
        output = app.invoke({"messages": input_messages}, config)
        bot_message = output["messages"][-1].content

        # bot_message = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()

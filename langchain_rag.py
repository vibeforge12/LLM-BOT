import string
import random
from datetime import datetime

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

import config
from common import *

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from prompt.prompt_ko import *


class Chat(BaseModel):
    response: str = Field(description="answer to the question ")


class DialogLLM:
    def __init__(self, model_name: str, retriever: Chroma, session_id: str):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
        )

        self.retriever = retriever
        self.session_id = session_id

    def generate_response(self, message: str):
        #
        # 대화이력을 통해 질문을 재작성 하는 단계
        #
        contextualize_q_system_prompt = CONTEXT_Q_SYSTEM_PROMPT

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        #
        # 대화이력을 통해 질문을 재작성한 결과를 통해 답변을 생성하는 단계
        #
        system_prompt = SYSTEM_PROMPT
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        chain_with_history = RunnableWithMessageHistory(rag_chain, lambda get_session_history: SQLChatMessageHistory(
            session_id=self.session_id, connection="sqlite:///sqlite.db"
        ), input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer")

        config = {"configurable": {"session_id": "test"}}  # session_id를 넘기긴 해야하나 사용되지 않음
        response = chain_with_history.invoke({"input": message}, config=config)

        return response


class DialogRetriever:
    def __init__(self, collection_name: str, chroma_db_path: str, data: list = None):
        self.collection_name = collection_name
        self.chroma_db_path = chroma_db_path

        self.embeddings_model = OpenAIEmbeddings()

        if not os.path.exists(chroma_db_path):
            self.create_vectorstore(data)
        else:
            self.load_vectorstore()

    def create_vectorstore(self, data):
        self.vectorstore = Chroma.from_documents(documents=data, embedding=self.embeddings_model,
                                                 collection_name=self.collection_name,
                                                 persist_directory=self.chroma_db_path)

    def load_vectorstore(self):
        self.vectorstore = Chroma(collection_name=self.collection_name, embedding_function=self.embeddings_model,
                                  persist_directory=self.chroma_db_path)

    def retrieve(self, input_text):
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        retrieved_docs = retriever.invoke(input_text)

        return retrieved_docs

    def get_retriever(self):
        return self.vectorstore.as_retriever()


if __name__ == "__main__":
    loader = CSVLoader(file_path='data/dialog_data.csv', metadata_columns=["id", "category"],
                       content_columns=["content"])
    data = loader.load()

    retriever = DialogRetriever(collection_name="dialog_data", chroma_db_path="vectordb", data=data)

    # retrieved_docs = retriever.retrieve(data)
    #
    # for doc in retrieved_docs:
    #     print(doc.page_content)
    #     print(doc.metadata)

    # 세션 아이디 생성
    date_str = datetime.now().strftime("%Y%m%d")
    rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
    session_id = f"{date_str}_{rand_str}"

    print(f'Current session_id: {session_id}')

    print("AI: 반가워! 나는 대화를 좋아하는 오렌지큐라고해. 함께 이야기하며 너의 고민을 들어주고 싶어. 어떤 고민이 있는지 말해줄래?")
    while True:
        input_text = input('사용자: ')
        llm = DialogLLM(model_name=config.GPT_MODEL, retriever=retriever.get_retriever(),
                        session_id=session_id)
        result = llm.generate_response(input_text.strip())

        print(f'AI: {result["answer"]}')

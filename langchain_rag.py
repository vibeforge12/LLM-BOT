from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

import config
from common import *

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Chat(BaseModel):
    response: str = Field(description="answer to the question ")


class DialogLLM:
    def __init__(self, model_name: str, retriever):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
        )

        self.retriever = retriever

    def generate_response(self, message: str):
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        chat_history = [
            HumanMessage("Hi! I'm Bob."),
            AIMessage("What's my name?"),
        ]

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        input_data = {
            "chat_history": chat_history,
            "input": message,
        }
        response = rag_chain.invoke(input_data)

        return response


class DialogRetriever:
    def __init__(self, collection_name: str, chroma_db_path: str, data: list = None):
        self.collection_name = collection_name
        self.chroma_db_path = chroma_db_path

        self.llm = ChatOpenAI(model="gpt-4o-mini")
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

    def retrieve(self, data):
        retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        # retrieved_docs = retriever.invoke("제가 어떤 것을 하고싶은지 잘 모르겠어요")
        retrieved_docs = retriever.invoke("공부를 어떻게 해야할지 쉽지가 않아요")

        return retrieved_docs

    def get_retriever(self):
        return self.vectorstore.as_retriever()


loader = CSVLoader(file_path='data/dialog_data.csv', metadata_columns=["id", "category"], content_columns=["content"])
data = loader.load()

retriever = DialogRetriever(collection_name="dialog_data", chroma_db_path="vectordb", data=data)

retrieved_docs = retriever.retrieve(data)

for doc in retrieved_docs:
    print(doc.page_content)
    print(doc.metadata)

llm = DialogLLM(model_name=config.GPT_MODEL, retriever=retriever.get_retriever())
result = llm.generate_response("공부를 어떻게 해야할지 쉽지가 않아요")

print(result['answer'])

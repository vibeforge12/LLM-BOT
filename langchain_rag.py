import logging
import os
import string
import random
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.agents import AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from logger import logger
from common import *
from prompt.prompt_ko import *


class CustomHandler(BaseCallbackHandler):
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        formatted_prompts = "\n".join(prompts)

        # save log
        self.logger.info(f"Prompt:\n{formatted_prompts}")

    def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        output_text = response.generations[0][0].text

        # save log
        self.logger.info(f"Output:\n{output_text}")


class Chat(BaseModel):
    response: str = Field(description="answer to the question ")


class DialogAgentLLM:
    def __init__(self, model_name: str, retriever: Chroma, session_id: str, log_dir: str = None):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
        )

        self.critic_llm = ChatOpenAI(
            model_name='gpt-4o',
            temperature=0,
        )

        self.retriever = retriever
        self.session_id = session_id

        #
        # 파일 로그 설정
        #
        if log_dir is None:
            log_dir = os.path.join(APP_ROOT, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logger = logging.Logger(session_id)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s (%(module)s:%(lineno)d) %(name)s - %(levelname)s: %(message)s')

        file_handler = logging.FileHandler(os.path.join(log_dir, f"{session_id}.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        self.logger = logger

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
            session_id=self.session_id, connection=f"sqlite:///{APP_ROOT}/sqlite.db"
        ), input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer")

        config = {"configurable": {"session_id": self.session_id},
                  "callbacks": [CustomHandler(self.logger)]}  # session_id를 넘기긴 해야하나 사용되지 않음
        response = chain_with_history.invoke({"input": message}, config=config)

        #
        # 대화 이력을 통해 클래스를 분류하는 단계
        #
        # chat_history = response["chat_history"]
        # logger.debug(f"chat_history: {chat_history}")
        #
        # # chat_history를 텍스트로 변환
        # chat_history_list = []
        # for message_obj in chat_history:
        #     if message_obj.type == "human":
        #         chat_history_list.append(f"학생 : {message_obj.content}")
        #     else:
        #         chat_history_list.append(f"선생님 : {message_obj.content}")
        #
        # classifier_prompt = PromptTemplate.from_template(CLASSIFIER_PROMPT)
        #
        # class Classification(BaseModel):
        #     probabilities: Dict[str, float] = Field(description="대화이력에 해당하는 각 분류의 확률 딕셔너리, 확률의 합은 1이 되어야 함(0~1사이의 값)")
        #
        # output_parser = JsonOutputParser(pydantic_object=Classification)
        # format_instructions = output_parser.get_format_instructions()
        #
        # classifier_chain = classifier_prompt | self.critic_llm | output_parser
        #
        # response_2 = classifier_chain.invoke(
        #     {"chat_history": chat_history_list, "format_instructions": format_instructions},
        #     config=config)
        #
        # print(response_2)

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


class UserSimulatorLLM:
    def __init__(self, model_name: str, retriever: DialogRetriever):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
        )

        self.retriever = retriever

    def generate_response(self, chat_history: list):
        chat_history_list = []
        for message_obj in chat_history:
            if message_obj.type == "human":
                chat_history_list.append(f"학생 : {message_obj.content}")
            else:
                chat_history_list.append(f"선생님 : {message_obj.content}")

        chat_history_text = "\n".join(chat_history_list)

        system_prompt = USER_PROMPT
        simulated_user_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, simulated_user_prompt)
        retriever_chain = create_retrieval_chain(self.retriever.get_retriever(), question_answer_chain)

        retriever_chain.config['run_name'] = "RetrievalChain(UserSimulator)"

        response = retriever_chain.invoke({"input": chat_history_text, "chat_history": chat_history})

        return response


class DialogAgent:
    chat_history = []

    def __init__(self, session_id: str):
        csv_file = os.path.join(APP_ROOT, 'data/train.csv')
        loader = CSVLoader(file_path=csv_file, metadata_columns=["id", "category"],
                           content_columns=["category", "content"])
        data = loader.load()

        vectordb_path = os.path.join(APP_ROOT, 'vectordb')
        retriever = DialogRetriever(collection_name="dialog_data", chroma_db_path=vectordb_path, data=data)

        self.llm = DialogAgentLLM(model_name=config.GPT_MODEL, retriever=retriever.get_retriever(),
                                  session_id=session_id)

        logger.info(f'DialogAgent session_id: {session_id}')

    def postprocess_response(self, response: str):
        answer = response["answer"]
        chat_history = response["chat_history"]

        history = []
        for message_obj in chat_history:
            history_dict = {}
            if message_obj.type == "human":
                history_dict["role"] = "User"
                history_dict["content"] = message_obj.content
            else:
                history_dict["role"] = "AI"
                history_dict["content"] = message_obj.content

            history.append(history_dict)

        return {"answer": answer, "history": history}

    def generate_response(self, message):
        response = self.llm.generate_response(message)
        self.update_chat_history(response)
        answer = response["answer"]

        return self.postprocess_response(response)

    def update_chat_history(self, response):
        chat_history = response["chat_history"]
        input_meg = HumanMessage(response["input"])
        chat_history.append(input_meg)
        answer_msg = AIMessage(response["answer"])
        chat_history.append(answer_msg)

        self.chat_history = chat_history

    def get_chat_history(self, return_type="text"):
        if return_type == "text":
            chat_history = []
            for message_obj in self.chat_history:
                if message_obj.type == "human":
                    chat_history.append(f"User : {message_obj.content}")
                else:
                    chat_history.append(f"AI : {message_obj.content}")

            return chat_history
        elif return_type == "object":
            return self.chat_history
        else:
            raise ValueError("Invalid return_type")


class UserSimulator:
    def __init__(self):
        csv_file = os.path.join(APP_ROOT, 'data/test.csv')
        loader = CSVLoader(file_path=csv_file, metadata_columns=["id", "category"],
                           content_columns=["category", "content"])
        data = loader.load()

        vectordb_path = os.path.join(APP_ROOT, 'vectordb')
        retriever = DialogRetriever(collection_name="dialog_data", chroma_db_path=vectordb_path, data=data)

        self.user_llm = UserSimulatorLLM(config.GPT_MODEL, retriever)

    def generate_response(self, history):
        result = self.user_llm.generate_response(history)
        ansewer = result["answer"]

        return ansewer


if __name__ == "__main__":

    # 세션 아이디 생성
    date_str = datetime.now().strftime("%Y%m%d")
    rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
    session_id = f"{date_str}_{rand_str}"

    logger.info(f'Current session_id: {session_id}')

    dialog_agent = DialogAgent(session_id)

    print("AI: 반가워! 나는 대화를 좋아하는 오렌지큐라고해. 함께 이야기하며 너의 고민을 들어주고 싶어. 어떤 고민이 있는지 말해줄래?")
    while True:
        input_text = input('사용자: ')

        result = dialog_agent.generate_response(input_text)

        print(f'AI: {result}')

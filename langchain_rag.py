from common import *

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI


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


llm = ChatOpenAI(model="gpt-4o-mini")

loader = CSVLoader(file_path='data/dialog_data.csv', metadata_columns=["id", "category"], content_columns=["content"])
data = loader.load()

retriever = DialogRetriever(collection_name="dialog_data", chroma_db_path="vectordb", data=data)

retrieved_docs = retriever.retrieve(data)

for doc in retrieved_docs:
    print(doc.page_content)
    print(doc.metadata)


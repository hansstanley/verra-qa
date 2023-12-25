from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from verra import schema


class VerraQA:
    def __init__(self, model_path: str) -> None:
        self.llm = GPT4All(model=model_path, max_tokens=2048)
        self.qa_path = None
        self.qa_chain = None

    def load_document(self, path: str):
        if path == self.qa_path:
            return
        # https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf#using-pymupdf
        loader = PyMuPDFLoader(path)
        # https://python.langchain.com/docs/modules/data_connection/vectorstores/
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
        )
        pages = loader.load_and_split(text_splitter)
        vectorstore = Chroma.from_documents(
            documents=pages,
            embedding=GPT4AllEmbeddings(),  # type: ignore
        )
        self.qa_path = path
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=vectorstore.as_retriever(),
        )

    def get_response(self, question: str):
        if self.qa_chain is None:
            raise Exception("document has not been loaded")
        return schema.QAResponse.model_validate(
            self.qa_chain({"query": question}),
        )

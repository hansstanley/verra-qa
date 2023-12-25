from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import GPT4All
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from verra import schema


class VerraQA:
    def __init__(self, pdf_path: str, model_path: str) -> None:
        # https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf#using-pymupdf
        loader = PyMuPDFLoader(pdf_path)
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

        llm = GPT4All(model=model_path, max_tokens=2048)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
        )

    def get_response(self, question: str):
        return schema.QAResponse.model_validate(
            self.qa_chain({"query": question}),
        )

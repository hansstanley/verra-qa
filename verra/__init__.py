from os import path

from litestar import Litestar, get

from verra import qa, schema

# models obtained from https://gpt4all.io/index.html
PATH_MODEL_CHAT = "models/mistral-7b-openorca.Q4_0.gguf"
PATH_MODEL_INST = "models/gpt4all-falcon-q4_0.gguf"
PATH_MODEL_RAG = "models/all-MiniLM-L6-v2-f16.gguf"
PATH_DOCS = "docs"

verra_qa = qa.VerraQA(model_path=PATH_MODEL_INST)


@get("/")
async def index(question: str, document: str) -> schema.QAResponse:
    verra_qa.load_document(path.join(PATH_DOCS, document + ".pdf"))
    return verra_qa.get_response(question)


app = Litestar([index])

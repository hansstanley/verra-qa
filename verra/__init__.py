from litestar import Litestar, get

from verra import qa, schema

# model obtained from https://gpt4all.io/index.html
PATH_MODEL_CHAT = "models/mistral-7b-openorca.Q4_0.gguf"
PATH_MODEL_INST = "models/gpt4all-falcon-q4_0.gguf"
PATH_MODEL_RAG = "models/all-MiniLM-L6-v2-f16.gguf"
PATH_DOC_0007 = "docs/VM0007.pdf"

verra_qa = qa.VerraQA(pdf_path=PATH_DOC_0007, model_path=PATH_MODEL_INST)


@get("/")
async def index(question: str) -> schema.QAResponse:
    return verra_qa.get_response(question)


app = Litestar([index])

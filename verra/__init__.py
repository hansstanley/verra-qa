from litestar import Litestar, get

from verra import qa

# model obtained from https://gpt4all.io/index.html
PATH_MODEL = "models/mistral-7b-openorca.Q4_0.gguf"
PATH_DOC_0007 = "docs/VM0007.pdf"

verra_qa = qa.VerraQA(pdf_path=PATH_DOC_0007, model_path=PATH_MODEL)


@get("/")
async def index(question: str) -> str:
    return verra_qa.get_response(question)


app = Litestar([index])

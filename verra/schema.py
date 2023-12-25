import pydantic


class QAResponse(pydantic.BaseModel):
    query: str
    result: str

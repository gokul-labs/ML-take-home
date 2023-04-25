from pydantic import BaseModel


class FiltererResponseSchema(BaseModel):
    label: str
    score: float

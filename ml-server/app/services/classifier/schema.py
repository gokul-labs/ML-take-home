from pydantic import BaseModel


class ClassifierResponseSchema(BaseModel):
    label: str
    score: float

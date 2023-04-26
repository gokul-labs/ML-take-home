from pydantic import BaseModel


class ClassifierResponseSchema(BaseModel):
    """
    Classifier Response Schema
    """
    label: str
    score: float

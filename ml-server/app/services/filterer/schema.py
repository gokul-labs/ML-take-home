from pydantic import BaseModel


class FiltererResponseSchema(BaseModel):
    """
    Filterer Response Schema
    """
    label: str
    score: float

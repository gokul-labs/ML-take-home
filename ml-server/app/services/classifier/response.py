from typing import List

from .schema import ClassifierResponseSchema


def ClassifyResponseModel(
        *, message: str, success=True, data: List[ClassifierResponseSchema] = None
) -> dict:
    return {"success": success, "message": message, "data": data}

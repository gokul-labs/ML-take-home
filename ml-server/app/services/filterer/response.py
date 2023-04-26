from typing import List

from .schema import FiltererResponseSchema


def FiltererResponseModel(
        *, message: str, success=True,
        data: List[FiltererResponseSchema] = None
) -> dict:
    return {"success": success, "message": message, "data": data}

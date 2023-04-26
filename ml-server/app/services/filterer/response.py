from typing import List

from .schema import FiltererResponseSchema


def FiltererResponseModel(
        *, message: str, success=True,
        data: List[FiltererResponseSchema] = None
) -> dict:
    """
    Model for Filter endpoint
    :param message: Message
    :param success: Success status
    :param data: Result
    :return:
    """
    return {"success": success, "message": message, "data": data}

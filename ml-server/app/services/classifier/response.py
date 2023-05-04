from typing import List, Dict

from .schema import ClassifierResponseSchema


def ClassifyResponseModel(
        *, message: str, success=True,
        data: List[List[ClassifierResponseSchema]] = None
) -> dict:
    """
    Model for Classify endpoint
    :param message: Message
    :param success: Success status
    :param data: Result
    :return:
    """
    return {"success": success, "message": message, "data": data}

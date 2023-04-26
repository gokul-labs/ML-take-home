from typing import List

from .schema import ClassifierResponseSchema


def ClassifyResponseModel(
        *, message: str, success=True,
        data: List[ClassifierResponseSchema] = None
) -> dict:
    """
    Model for Classify endpoint
    :param message: Message
    :param success: Success status
    :param data: Result
    :return:
    """
    return {"success": success, "message": message, "data": data}

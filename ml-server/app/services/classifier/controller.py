import io
from .client import ImageClassifier
from PIL import Image
from fastapi import UploadFile, APIRouter
from typing import List, Optional

from .response import ClassifyResponseModel

classify_router = APIRouter(prefix="/classify")

Classifier = ImageClassifier()


@classify_router.post("/", tags=["Image Classification"])
async def classify(files: list[UploadFile] = None):
    """
    Endpoint entry for classification
    :param files: Images
    :return: Classification results
    """
    if not files:
        return ClassifyResponseModel(message="No file sent", success=False)
    results = []
    for file in files:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        result = Classifier.classify(image)
        results.append(result)
    return ClassifyResponseModel(data=results,
                                 message="Successful classification")

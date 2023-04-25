import io
from .client import ImageClassifier
from PIL import Image
from fastapi import UploadFile, APIRouter, File

from .response import ClassifyResponseModel
from typing import Union

classify_router = APIRouter(prefix="/classify")

Classifier = ImageClassifier()


@classify_router.post("/", tags=["Image Classification"])
async def classify(files: list[UploadFile, None] = None):
    print("Files", files)
    if not files:
        return ClassifyResponseModel(message="No file sent", success=False)
    results = []
    for file in files:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        result = Classifier.classify(image)
        results.append(result)
    return ClassifyResponseModel(data=results, message="Successful classification")

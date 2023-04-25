import io
from .client import ImageFilterer
from PIL import Image
from fastapi import UploadFile, APIRouter

from .response import FiltererResponseModel
from typing import Union

filterer_router = APIRouter(prefix="/filter")

Filterer = ImageFilterer()


@filterer_router.post("/", tags=["Image Filterer"])
async def filter(file: Union[UploadFile, None] = None):
    if not file:
        return FiltererResponseModel(message="No file sent", success=False)
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    result = Filterer.filter(image)

    return FiltererResponseModel(data=result, message="Successfully verified")

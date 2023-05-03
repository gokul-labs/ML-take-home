import io
from typing import Union, Optional

from PIL import Image
from fastapi import UploadFile, APIRouter

from .client import ImageFilterer
from .response import FiltererResponseModel

filter_router = APIRouter(prefix="/filter")

Filterer = ImageFilterer()


@filter_router.post("/", tags=["Image Filtering"])
async def filter(file: Union[UploadFile, None] = None):
    """
    Endpoint entry for filtering
    :param file: Images
    :return: Filtering results
    """
    if not file:
        return FiltererResponseModel(message="No file sent", success=False)
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    result = Filterer.filter(image)

    return FiltererResponseModel(data=result, message="Successfully verified")

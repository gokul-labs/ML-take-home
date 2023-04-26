import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .services.classifier.controller import classify_router
from .services.filterer.controller import filter_router


def create_app():
    """
    Create the REST application from all routers
    :return: app object
    """
    app = FastAPI(title="Potato Leaf Health Status",
                  description="Uses ML algorithm to classify the health status"
                              " of a potato leaf image,"
                              " into one of 3 categories: early blight, "
                              "late blight and healthy.",
                  version="1.0",
                  docs_url='/docs',
                  openapi_url='/openapi.json',
                  redoc_url=None)

    origins = [
        "http://localhost:3000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    app.include_router(classify_router)
    app.include_router(filter_router)

    @app.get("/", tags=["Root"])
    async def read_root():
        """
        Root endpoint for ml application server
        :return:
        """
        return {
            "message": f"Welcome to the image classifier "
                       f"server!!, servertime {time.time()}"
        }

    return app


app = create_app()

from contextlib import asynccontextmanager

import gradio as gr
from api.config import settings
from api.routers import routers
from fastapi import FastAPI

from gradio_ui import interface_video
from logger_config import logger


@asynccontextmanager  # type:ignore
async def lifespan(app: FastAPI):
    yield


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    # Verificar si TensorFlow está usando GPU
    # gpus = tf.config.list_physical_devices("GPU")
    # if gpus:
    #     logger.info(f"TensorFlow usará la(s) GPU: {gpus}")
    # else:
    #     logger.info("TensorFlow usará la CPU")

    for router, prefix, tags in routers:

        app.include_router(router, prefix=prefix, tags=tags)  # type:ignore

    app = gr.mount_gradio_app(app, interface_video, path="/video")

    @app.get("/")
    async def root():
        return {"message": "Hello World"}

    # conn = get_db()

    return app

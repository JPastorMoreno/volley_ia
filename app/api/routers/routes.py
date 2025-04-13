import shutil
import uuid

from celery import Celery
from fastapi import APIRouter, File, UploadFile

router = APIRouter()
celery_app = Celery('worker', broker='redis://redis:6379/0')

@router.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    path = f"./videos/input_{video_id}.mp4"

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    celery_app.send_task("analysis_worker.worker.analyze_video", args=[path])

    return {"message": "Video recibido", "id": video_id}
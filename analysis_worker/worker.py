from celery import Celery
import os
from logger_config import logger
import torch
from services.detection import run_detection_and_tracking

app = Celery("worker", broker="redis://redis:6379/0")
logger.info(torch.cuda.get_device_name(torch.cuda.current_device()))


@app.task(name="analysis_worker.worker.analyze_video")
def analyze_video(path):
    output_path = path.replace("input", "output")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_detection_and_tracking(path, output_path)

import cv2
import logger_config
from logger_config import logger
import numpy as np
import shutil
import tempfile

import os


def read_video(video_path, chunk_size=200):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    temp_dir = tempfile.mkdtemp(prefix="video_processing_")
    current_chunk = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Liberar memoria después de cada chunk
        if len(current_chunk) >= chunk_size:
            yield current_chunk.copy(), fps, width, height
            current_chunk.clear()

        current_chunk.append(frame.copy())

    # Procesar chunk final
    if current_chunk:
        yield current_chunk.copy(), fps, width, height

    cap.release()
    shutil.rmtree(temp_dir)


def save_video(output_video_frames, output_video_path, fps, width, height):
    logger.info(f"[INFO] Guardando video en: {output_video_path}")

    # Configurar codec H264 para mejor compresión
    fourcc = cv2.VideoWriter_fourcc(*"H264")

    # Escribir frames en bloques
    chunk_size = 30
    temp_files = []

    for i in range(0, len(output_video_frames), chunk_size):
        chunk = output_video_frames[i : i + chunk_size]
        temp_path = f"{tempfile.mkdtemp()}/chunk_{i}.mp4"
        temp_files.append(temp_path)

        writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

        for frame in chunk:
            writer.write(frame)

        writer.release()

    # Combinar archivos temporales
    final_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for temp_path in temp_files:
        cap = cv2.VideoCapture(temp_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            final_writer.write(frame)
        cap.release()
        os.remove(temp_path)

    final_writer.release()

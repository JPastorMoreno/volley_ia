from utils import read_video, save_video
from trackers import Tracker
import cv2
from logger_config import logger


def run_detection_and_tracking(input_path: str, output_path: str):
    tracker: Tracker = Tracker(model_path="Yolo11_1080.pt")
    output_video_frames = []
    # Abrimos el writer al inicio
    writer = None

    for video_frames, fps, width, height in read_video(input_path, chunk_size=200):
        tracks = tracker.get_objet_tracks(
            video_frames, read_from_stub=False, stub_path="lechuga"
        )
        annotated_frames = tracker.draw_annotations(
            video_frames=video_frames, tracks=tracks
        )

        # Inicializamos writer si aún no está creado
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            logger.info(f"guardando video en {output_path}")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in annotated_frames:
            writer.write(frame)

    # Liberamos el writer al final
    if writer:
        writer.release()

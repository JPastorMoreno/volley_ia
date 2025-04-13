from ultralytics import YOLO
import supervision as sv
import pickle
import torch
import os
from logger_config import logger
from utils import (
    get_center_of_bbox,
    get_bbox_width,
)
import cv2

import torch.nn.functional as F


class Tracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size: int = 20
        detections = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tensor_batch = []

        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]

            tensor_batch = []

            for frame in batch_frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = (
                    torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
                )  # [3, H, W]

                # Padding para que sea múltiplo de 32 (sin deformar)
                h, w = tensor.shape[1], tensor.shape[2]
                pad_h = (32 - h % 32) % 32
                pad_w = (32 - w % 32) % 32
                tensor = F.pad(
                    tensor, (0, pad_w, 0, pad_h), value=0
                )  # [3, H+pad, W+pad]

                tensor_batch.append(tensor)

            input_tensor = torch.stack(tensor_batch).to(
                device
            )  # [B, 3, H_padded, W_padded]

            with torch.no_grad():
                detections_batch = self.model.predict(input_tensor, conf=0.4)

            detections += detections_batch

        return detections

    def get_objet_tracks(self, frames, read_from_stub: bool, stub_path: str):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks: dict = {"volleyball": [], "players": []}
        for frame_num, detection in enumerate(detections):

            # Getting the class names}")
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            # Converting to supervision format
            detection_supervision = sv.Detections.from_ultralytics(
                ultralytics_results=detection
            )
            # En caso de que tengamos 2 clases que se superponen
            # for object_ind, class_id in enumerate(detection_supervision.class_id):
            #     if cls_names[class_id] == "goalkeeper":
            #         detection_supervision.class_id[object_ind] = cls_names_inv["person"]

            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )
            tracks["volleyball"].append({})
            tracks["players"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                logger.info(cls_names)
                if cls_id == cls_names_inv.get("player"):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    # print(f"Frame: {frame_num}, Track ID: {track_id}, BBox: {bbox}")
                if cls_id == cls_names_inv.get("volleyball"):
                    tracks["volleyball"][frame_num][1] = {"bbox": bbox}
                    logger.info(
                        "*********************************************************"
                    )
                    logger.info(
                        f"Frame: {frame_num}, Track ID: {track_id}, BBox: {bbox}"
                    )

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)
        logger.info(len(tracks["volleyball"]))
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_annotations(self, video_frames, tracks):
        logger.info("Drawing annotations")
        logger.info(tracks["volleyball"])
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            volleyball_dict = tracks["volleyball"][frame_num]
            for track_id, volleyball in volleyball_dict.items():
                frame = self.draw_ellipse(
                    frame, volleyball["bbox"], (0, 255, 0), track_id
                )
            output_frames.append(frame)
        return output_frames

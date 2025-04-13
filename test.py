import cv2
import numpy as np
import supervision as sv
import torch
from roboflow import Roboflow

# Verificar disponibilidad de GPU
print(f"GPU disponible: {torch.cuda.is_available()}")

# Configuración
VIDEO_INPUT = "clip_10_40_10_45.mp4"
VIDEO_OUTPUT = "video_anotado_gpu.mp4"
CONFIDENCE_THRESHOLD = 40  # Roboflow usa 0-100
OVERLAP_THRESHOLD = 30

# Inicializar Roboflow (internamente usa GPU si está disponible)
rf = Roboflow(api_key="D2OAJtACl5jFOQJfPUCq")
project = rf.workspace().project("volleyball-eg0ze")
model = project.version(8).model

# Configurar anotadores
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

def process_frame(frame):
    # Convertir a RGB (Roboflow espera este formato)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Realizar predicción (Roboflow maneja GPU internamente)
    result = model.predict(frame_rgb, 
                         confidence=CONFIDENCE_THRESHOLD,
                         overlap=OVERLAP_THRESHOLD).json()
    
    # Procesar resultados
    boxes = []
    confidences = []
    class_ids = []
    labels = []
    
    for item in result["predictions"]:
        x_min = item["x"] - item["width"] / 2
        y_min = item["y"] - item["height"] / 2
        x_max = item["x"] + item["width"] / 2
        y_max = item["y"] + item["height"] / 2
        
        boxes.append([x_min, y_min, x_max, y_max])
        confidences.append(item["confidence"])
        class_ids.append(item["class_id"])
        labels.append(item["class"])
    
    return {
        "boxes": np.array(boxes),
        "confidences": np.array(confidences),
        "class_ids": np.array(class_ids),
        "labels": labels
    }

def process_video():
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print("Error al abrir el video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(VIDEO_OUTPUT,
                         cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            detections_data = process_frame(frame)

            detections = sv.Detections(
                xyxy=detections_data["boxes"],
                confidence=detections_data["confidences"],
                class_id=detections_data["class_ids"]
            )

            annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(annotated_frame, 
                                                     detections=detections, 
                                                     labels=detections_data["labels"])

            out.write(annotated_frame)
            frame_count += 1
            print(f"Procesado frame {frame_count}", end='\r')

        except Exception as e:
            print(f"\nError en frame {frame_count}: {str(e)}")
            continue

    cap.release()
    out.release()
    print(f"\n✅ Procesamiento completado. Total frames: {frame_count}")

if __name__ == "__main__":
    process_video()
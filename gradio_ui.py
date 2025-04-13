# app/gradio_ui.py
import os
import shutil
import uuid

import gradio as gr
from services.detection import run_detection_and_tracking

# Ensure the directories exist
os.makedirs("./videos/input", exist_ok=True)
os.makedirs("./videos/output", exist_ok=True)

def process_video(file_path):
    video_id = str(uuid.uuid4())
    input_path = f"./videos/input/{video_id}.mp4"
    output_path = f"./videos/output/{video_id}.mp4"

    # Copy the uploaded file to our input directory
    shutil.copy(file_path, input_path)

    run_detection_and_tracking(input_path, output_path)

    return output_path

interface_video = gr.Blocks()
with interface_video:
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Sube tu video de voleibol")
        with gr.Column():
            video_outputs = gr.Video(label="Video Analizado")
    title = "Analizador de Partidos de Voleibol"
    description = "Sube un video de un partido y observa la detección de jugadores y la pelota con YOLOv8."
    procesar_btn = gr.Button("Procesar", visible=True)

    procesar_btn.click(
        process_video,
        inputs=[video_input],
        outputs=[video_outputs],
    )
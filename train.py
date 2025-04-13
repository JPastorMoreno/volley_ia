from roboflow import Roboflow

rf = Roboflow(api_key="D2OAJtACl5jFOQJfPUCq")
project = rf.workspace("taner-g3yrh").project("volleyball-eg0ze")
version = project.version(9)
dataset = version.download("yolov8")
                
dataset.location
import shutil

shutil.move('football-players-detection-1/train',
            'football-players-detection-1/football-players-detection-1/train'
            )

shutil.move('football-players-detection-1/test',
            'football-players-detection-1/football-players-detection-1/test'
            )

shutil.move('football-players-detection-1/valid',
            'football-players-detection-1/football-players-detection-1/valid'
            )

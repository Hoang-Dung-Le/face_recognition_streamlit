from typing import List
from deepface import DeepFace
from fastapi import FastAPI, UploadFile, File
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import cv2
import base64
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_images")
async def predict_images(files: List[UploadFile]):
    predictions = []
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert image to numpy array
        image_np = np.array(image)
        # Convert RGB to BGR
        image_np = image_np[:, :, ::-1]

        try:
            result = DeepFace.find(img_path=image_np, db_path="../public/DiemDanh/DI20Z6A1/", model_name="Facenet512")          
            print(type(result[0]))

            directory_names = [p.iloc[0]["identity"].split('/')[4] for p in result]
            predictions.append(directory_names)

        except Exception as e:
            print(e)
            predictions.append("Người lạ")
        
    return JSONResponse(content=predictions)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
from deepface import DeepFace
from fastapi import FastAPI, UploadFile
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
import cv2
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict_image")
async def predict_image(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Convert image to numpy array
    image_np = np.array(image)
    # Convert RGB to BGR
    image_np = image_np[:, :, ::-1]

    # Perform face recognition
    result = DeepFace.find(img_path=image_np, db_path="./DI20Z6A1/", model_name="Facenet512")
    print(type(result[0]))
    r = []

    directory_names = [p.iloc[0]["identity"].split('/')[2] for p in result]
    return directory_names

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
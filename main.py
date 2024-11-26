from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import io
import cv2
from ultralytics import YOLO
import requests
import os

# URL to the model weights file in your GitHub repository
model_url = "https://github.com/yashseth391/wasify_api_final/blob/c3406a358ea4cdee3fd60da5cb99ec0c457e7f1c/last.pt"
model_path = "last.pt"

# Download the model weights if they do not exist
# Download the model weights if they do not exist
if not os.path.exists(model_path):
    print(f"Downloading model weights from {model_url}")
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Model weights downloaded successfully")
    else:
        print(f"Failed to download model weights: {response.status_code}")
# Load the model
try:
    model = YOLO(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

@app.get("/")
def read_root():
    return {"Model Run Successfully"}

@app.post("/predict/")
async def classify_image(file: UploadFile = File(...)):
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        image = image.convert("RGB")
        new_size = (image.width * 2, image.height * 2)
        image = image.resize(new_size)
        image.save("image.jpg")
        results = model(image)
        for result in results:
            result.show()
            result.save(filename="ans.jpg")
        img = cv2.imread("ans.jpg")
        return {"message": "Image saved successfully"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/ans/")
def ans():
    return FileResponse("ans.jpg")

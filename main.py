from typing import Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image
import io
import cv2
from ultralytics import YOLO

model_path = "https://github.com/yashseth391/testing_render/blob/main/last.pt"
model = YOLO('yolov8n.pt')
model = YOLO(model_path)

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
        return {"message": "Image saved successfully."}
    except Exception as e:
        return {"error": str(e)}

@app.get("/ans/")
def ans():
    return FileResponse("ans.jpg")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="debug")

import io
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import pathlib

# 🚑 Fix PosixPath issue on Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = r"D:\model_Api\model1_api\model\best.pt"

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    source="github",
    force_reload=True
)

@app.get("/")
def home():
    return {"message": "Billboard Detection API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes))

    results = model(img)
    predictions = results.pandas().xyxy[0].to_dict(orient="records")

    results.render()
    boxed_img = Image.fromarray(results.ims[0])

    buf = io.BytesIO()
    boxed_img.save(buf, format="JPEG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse(content={
        "detections": predictions,
        "boxed_image": img_base64
    })

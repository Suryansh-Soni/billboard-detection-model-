import io
import os
import base64
import pathlib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

# 🛠 Fix PosixPath bug on Windows
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

app = FastAPI(title="Billboard Detection API", version="1.0.0")

# Enable CORS (for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load YOLOv5 model (custom trained)
MODEL_PATH = "model/best.pt"  # Update path if needed

model = torch.hub.load(
    "ultralytics/yolov5",
    "custom",
    path=MODEL_PATH,
    source="github",
    force_reload=True
)

@app.get("/")
def home():
    return {"message": "✅ Billboard Detection API is running."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {e}"})

    results = model(image)
    predictions = results.pandas().xyxy[0].to_dict(orient="records")

    results.render()
    rendered_img = Image.fromarray(results.ims[0])

    # Encode image with boxes to base64
    buf = io.BytesIO()
    rendered_img.save(buf, format="JPEG")
    encoded_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse(content={
        "detections": predictions,
        "boxed_image": encoded_img
    })

# ✅ For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)

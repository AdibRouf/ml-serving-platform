from fastapi import FastAPI, UploadFile
from torchvision import models, transforms
from PIL import Image
import torch
import io
import asyncio
import uuid

app = FastAPI()

model = models.resnet18(weights="IMAGENET1K_V1")
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

job_queue = asyncio.Queue()
results = {}  # job_id -> result or "pending"

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    job_id = str(uuid.uuid4())
    results[job_id] = "pending"
    await job_queue.put((job_id, image_bytes))
    return {"job_id": job_id}

@app.get("/result/{job_id}")
async def get_result(job_id: str):
    return {"status": results.get(job_id, "not_found")}

async def worker():
    while True:
        job_id, image_bytes = await job_queue.get()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        predicted_class = output.argmax().item()
        results[job_id] = predicted_class

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())
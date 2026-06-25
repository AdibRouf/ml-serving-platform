from fastapi import FastAPI, UploadFile
from torchvision import models, transforms
from PIL import Image
import torch
import io

app = FastAPI()

# Load model once, when the server starts (not on every request — that would be slow)
model = models.resnet18(weights="IMAGENET1K_V1")
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    predicted_class = output.argmax().item()
    return {"predicted_class": predicted_class}
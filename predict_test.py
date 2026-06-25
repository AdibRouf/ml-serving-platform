import torch
from torchvision import models, transforms
from PIL import Image

# Load a pretrained model (trained already on 1000 object categories)
model = models.resnet18(weights="IMAGENET1K_V1")
model.eval()  # tells the model "we're using you to predict, not to train"

# Prepare an image the way the model expects
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("test.jpg")
input_tensor = transform(img).unsqueeze(0)  # model expects a "batch" dimension

with torch.no_grad():  # we're not training, so skip gradient tracking (saves memory/time)
    output = model(input_tensor)

predicted_class = output.argmax().item()
print(f"Predicted class index: {predicted_class}")


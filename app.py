from fastapi import FastAPI, UploadFile
from torchvision import models, transforms
from PIL import Image
import torch
import io
import asyncio
import uuid

app = FastAPI()

# ── Model setup (loaded once at startup, not per-request) ──────────────────
model = models.resnet18(weights="IMAGENET1K_V1")
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── In-memory queue + results store ─────────────────────────────────────────
# NOTE: this is in-memory only — restarting the server wipes all jobs/results.
# A production version would back this with Redis or a database.
job_queue = asyncio.Queue()
results = {}  # job_id -> "pending" | int (predicted class)

# ── Batching configuration ──────────────────────────────────────────────────
BATCH_SIZE = 8       # max number of jobs to process together
MAX_WAIT = 0.05      # max seconds to wait collecting a batch before processing


@app.post("/predict")
async def predict(file: UploadFile):
    """Accepts an image, queues it for inference, and returns a job ID immediately."""
    image_bytes = await file.read()
    job_id = str(uuid.uuid4())
    results[job_id] = "pending"
    await job_queue.put((job_id, image_bytes))
    return {"job_id": job_id}


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Check the status/result of a previously submitted job."""
    return {"status": results.get(job_id, "not_found")}


def process_batch(batch):
    """Run a batch of (job_id, image_bytes) pairs through the model in one call."""
    job_ids = [job_id for job_id, _ in batch]
    images = []

    for _, image_bytes in batch:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        images.append(transform(img))

    # Stack individual [3, 224, 224] tensors into one [batch_size, 3, 224, 224] tensor
    input_batch = torch.stack(images)

    with torch.no_grad():
        outputs = model(input_batch)  # single model call for the whole batch

    predicted_classes = outputs.argmax(dim=1).tolist()

    for job_id, predicted_class in zip(job_ids, predicted_classes):
        results[job_id] = predicted_class


async def worker():
    """Background loop: collects jobs into batches, then processes each batch."""
    while True:
        batch = []

        # Always wait for at least one job before doing anything
        job = await job_queue.get()
        batch.append(job)

        # Try to gather more jobs quickly, up to BATCH_SIZE or MAX_WAIT, whichever first
        start_time = asyncio.get_event_loop().time()
        while len(batch) < BATCH_SIZE:
            elapsed = asyncio.get_event_loop().time() - start_time
            remaining = MAX_WAIT - elapsed
            if remaining <= 0:
                break
            try:
                job = await asyncio.wait_for(job_queue.get(), timeout=remaining)
                batch.append(job)
            except asyncio.TimeoutError:
                break

        process_batch(batch)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())
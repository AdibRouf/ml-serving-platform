"""
Benchmark script for the ML serving platform.

Sends a burst of concurrent requests to /predict, then polls /result
until every job is done. Reports total wall-clock time and throughput
(requests per second).

Usage:
    python benchmark.py [num_requests]

To get a fair before/after comparison for your resume metric:
1. Run this with your current app.py (BATCH_SIZE = 8) and record the result.
2. Temporarily edit app.py, set BATCH_SIZE = 1 (effectively no batching),
   restart the server, and run this again.
3. Compare the two throughput numbers — that difference is your real,
   honest "improved throughput by X%" metric.
"""

import requests
import time
import sys
from concurrent.futures import ThreadPoolExecutor

SERVER_URL = "http://127.0.0.1:8000"
IMAGE_PATH = "test.jpg"


def submit_job():
    """Send one prediction request, return its job_id."""
    with open(IMAGE_PATH, "rb") as f:
        response = requests.post(f"{SERVER_URL}/predict", files={"file": f})
    response.raise_for_status()
    return response.json()["job_id"]


def poll_until_done(job_id, timeout=30, interval=0.01):
    """Poll /result until the job is no longer 'pending', or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(f"{SERVER_URL}/result/{job_id}")
        status = response.json()["status"]
        if status != "pending":
            return status
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


def run_benchmark(num_requests):
    print(f"Submitting {num_requests} requests concurrently...")

    start_time = time.time()

    # Submit all requests concurrently using a thread pool
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        job_ids = list(executor.map(lambda _: submit_job(), range(num_requests)))

    submit_done_time = time.time()
    submission_time = submit_done_time - start_time
    print(f"All {num_requests} requests submitted in {submission_time:.3f}s")

    # Poll for all results concurrently
    with ThreadPoolExecutor(max_workers=num_requests) as executor:
        results = list(executor.map(poll_until_done, job_ids))

    end_time = time.time()
    total_time = end_time - start_time
    processing_time = end_time - submit_done_time  # isolates inference from submission overhead

    print(f"\n--- Results ---")
    print(f"Total requests: {num_requests}")
    print(f"Submission phase: {submission_time:.3f}s  (client-side overhead, NOT inference)")
    print(f"Processing phase: {processing_time:.3f}s  (this is the number that reflects batching/inference speed)")
    print(f"Processing throughput: {num_requests / processing_time:.2f} requests/sec")
    print(f"Overall throughput (incl. submission): {num_requests / total_time:.2f} requests/sec")
    print(f"Sample result: {results[0]}")


if __name__ == "__main__":
    num_requests = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_benchmark(num_requests)
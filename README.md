# Khmer TTS Backend

This backend keeps RunPod credentials and VoxCPM2 inference off the Windows desktop.

## Deployment status

The API and worker source in this workspace are prepared for deployment, but the GitHub repository must contain these files before RunPod can build anything. The current `main` branch of `csan79446-oss/khmer-tts-backend` was checked on 2026-09-08 and has an empty file tree.

## API service

The FastAPI service exposes `/health`, `/jobs`, `/jobs/{job_id}`, and `/jobs/{job_id}/audio`. It submits work to a RunPod queue endpoint and proxies completed WAV audio to the desktop client.

Run locally:

```powershell
cd backend/api
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:RUNPOD_ENDPOINT_ID = "your-endpoint-id"
$env:RUNPOD_API_KEY = "your-runpod-key"
$env:API_SECRET = "your-desktop-backend-secret"
uvicorn app.main:app --reload
```

Do not place RunPod credentials in the desktop `.env` file.

## Worker

The worker uses the official `VoxCPM.from_pretrained()` API and supports:

- `design`: optional natural-language voice prompt
- `controllable_clone`: reference audio plus style prompt
- `hi_fidelity_clone`: reference audio plus exact transcript

Build and deploy the worker image to a RunPod Serverless queue endpoint. Configure `MODEL_ID` and any Hugging Face authentication in RunPod environment settings. Test the actual CUDA/PyTorch image before production deployment.

The worker explicitly sets `optimize=False`. This avoids PyTorch Inductor trying to compile native code in the container, which can fail with `Failed to find C compiler`. Keep this setting for the first deployment; optimize only after adding and validating a compatible compiler toolchain.

The initial API stores completed audio in process memory for development only. Do not run multiple API instances or restart the API in production until this is replaced with private object storage and a persistent job store. A RunPod worker restart is expected and does not preserve API memory.

## RunPod GitHub deployment

RunPod can deploy the worker directly from GitHub:

1. Push the backend repository contents, including the root `Dockerfile`, `worker/handler.py`, and `worker/requirements.txt`.
2. In RunPod, open **Settings -> Connections -> GitHub** and authorize access to this repository.
3. Choose **Serverless -> New Endpoint -> Import Git Repository**.
4. Select `csan79446-oss/khmer-tts-backend`, branch `main`, and Dockerfile path `Dockerfile`.
5. Choose a Queue endpoint, configure a compatible GPU, and add `MODEL_ID`, `HF_TOKEN` if needed, and worker settings.
6. Deploy and test the endpoint with a small Khmer request before connecting the desktop app.

RunPod's GitHub integration rebuilds when a new GitHub release is created; a normal commit alone does not automatically update the endpoint. The API service is separate and should run on a normal HTTPS host; do not deploy the FastAPI control service as the queue worker.

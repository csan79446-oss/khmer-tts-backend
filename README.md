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

## Audio quality settings

Quality-first defaults are baked into the worker and FastAPI API:

| Setting | Default | Effect |
| --- | --- | --- |
| `cfg_value` | `2.0` | Classifier-free guidance. Higher = crisper adherence to the text/voice prompt; >2.5 can sound harsh. |
| `inference_timesteps` | `30` (API allows `4..40`) | LocDiT flow-matching steps. More steps = smoother audio, linear speed cost; 30 is the quality/cost sweet spot. |
| `normalize` | `false` | VoxCPM's built-in normalizer is English/Chinese and would rewrite Khmer digits/dates into English words. The Khmer text-prep engine (14 rules) is the authoritative normalizer, so built-in normalization is off by default. |
| `denoise` | `true` | Enhances clone reference audio when the denoiser is loaded (see `LOAD_DENOISER` below). |
| `retry_badcase` | `true` (max 3 retries) | Re-rolls the sample when the audio/text length ratio looks wrong, avoiding garbled output. |

Every generated WAV also passes through a worker-side mastering chain before encoding: DC-offset removal, 5 ms edge fades (no boundary clicks), a gentle loudness lift for very quiet output, and a true-peak guard at -0.45 dBFS so 16-bit output never clips.

### Worker environment variables

- `LOAD_DENOISER=1` loads the ZipEnhancer denoiser (ModelScope `iic/speech_zipenhancer_ans_multiloss_16k_base`) so `denoise=true` actually cleans noisy reference audio before cloning. Off by default because it downloads an extra model at cold start; enable it on a GPU with spare VRAM for the best clone quality.
- `MAX_REFERENCE_AUDIO_SECONDS` (default `10`) caps prompt audio length to keep VoxCPM2's 8192-token KV cache from overflowing.
- `OUTPUT_SAMPLE_RATE` overrides the forced 48 kHz VoxCPM2 output rate.

### Desktop quality controls

The desktop TTS page adds a **VoxCPM2 audio quality** selector: **Best (30 steps)** or **Standard (24 steps)**. Voice Design and Voice Clone pages let you store a per-voice CFG value (default 2.0) and a denoise flag per clone.

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

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field


app = FastAPI(title="Khmer TTS Backend", version="0.1.0")
RUNPOD_API_URL = os.getenv("RUNPOD_API_URL", "https://api.runpod.ai/v2").rstrip("/")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
MAX_REFERENCE_AUDIO_BASE64_LENGTH = int(os.getenv("MAX_REFERENCE_AUDIO_BASE64_LENGTH", "33554432"))


@dataclass
class JobRecord:
    runpod_id: str
    audio: bytes | None = None
    sample_rate: int = 48000
    updated_at: float = 0.0


jobs: dict[str, JobRecord] = {}


class GenerationRequest(BaseModel):
    mode: str = Field(default="design", pattern="^(design|controllable_clone|hi_fidelity_clone)$")
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    voice_prompt: str = Field(default="", max_length=1000)
    reference_audio_base64: str | None = Field(default=None, max_length=MAX_REFERENCE_AUDIO_BASE64_LENGTH)
    prompt_text: str | None = Field(default=None, max_length=5000)
    text_preparation: bool = True
    cfg_value: float = Field(default=1.2, ge=1.0, le=3.0)
    inference_timesteps: int = Field(default=20, ge=4, le=30)
    normalize: bool = True
    denoise: bool = True
    seed: int | None = Field(default=None, ge=0)


async def authenticate(authorization: str | None = Header(default=None)) -> None:
    if not API_SECRET:
        return
    expected = f"Bearer {API_SECRET}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid backend credentials.")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "khmer-tts-backend", "runpod_configured": bool(RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY)}


@app.post("/jobs", dependencies=[Depends(authenticate)])
async def create_job(request: GenerationRequest) -> dict[str, Any]:
    if not API_SECRET:
        raise HTTPException(status_code=503, detail="API_SECRET is required before accepting generation jobs.")
    if not RUNPOD_ENDPOINT_ID or not RUNPOD_API_KEY:
        raise HTTPException(status_code=503, detail="RunPod backend is not configured.")
    if request.mode == "hi_fidelity_clone" and (not request.reference_audio_base64 or not request.prompt_text):
        raise HTTPException(status_code=400, detail="Hi-fidelity cloning requires reference audio and its transcript.")
    if request.mode == "controllable_clone" and not request.reference_audio_base64:
        raise HTTPException(status_code=400, detail="Controllable cloning requires reference audio.")

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{RUNPOD_API_URL}/{RUNPOD_ENDPOINT_ID}/run",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            json={"input": request.model_dump(exclude_none=True)},
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="Unable to submit the VoxCPM2 job to RunPod.")
    data = response.json()
    job_id = str(data.get("id", ""))
    if not job_id:
        raise HTTPException(status_code=502, detail="RunPod did not return a job ID.")
    jobs[job_id] = JobRecord(runpod_id=job_id, updated_at=time.time())
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}", dependencies=[Depends(authenticate)])
async def get_job(job_id: str) -> dict[str, Any]:
    record = jobs.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found.")
    if record.audio is not None:
        return {"job_id": job_id, "status": "completed", "output_url": f"/jobs/{job_id}/audio", "sample_rate": record.sample_rate}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{RUNPOD_API_URL}/{RUNPOD_ENDPOINT_ID}/status/{record.runpod_id}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        )
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail="Unable to read the RunPod job status.")
    data = response.json()
    status = str(data.get("status", "")).lower()
    if status == "completed":
        output = data.get("output") or {}
        encoded = output.get("audio_base64")
        if not encoded:
            raise HTTPException(status_code=502, detail="VoxCPM2 completed without audio output.")
        record.audio = base64.b64decode(encoded)
        record.sample_rate = int(output.get("sample_rate", 48000))
        record.updated_at = time.time()
        return {"job_id": job_id, "status": "completed", "output_url": f"/jobs/{job_id}/audio", "sample_rate": record.sample_rate}
    if status in {"failed", "cancelled", "timed_out"}:
        detail = (data.get("error") or {}).get("message", "VoxCPM2 job failed.") if isinstance(data.get("error"), dict) else data.get("error", "VoxCPM2 job failed.")
        return {"job_id": job_id, "status": status, "error": detail}
    return {"job_id": job_id, "status": status or "processing"}


@app.get("/jobs/{job_id}/audio", dependencies=[Depends(authenticate)])
async def get_audio(job_id: str) -> Response:
    record = jobs.get(job_id)
    if not record or record.audio is None:
        raise HTTPException(status_code=404, detail="Audio is not ready.")
    return Response(content=record.audio, media_type="audio/wav")

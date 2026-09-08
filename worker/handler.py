from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import numpy as np
import runpod
import soundfile as sf
from voxcpm import VoxCPM


MODEL_ID = os.getenv("MODEL_ID", "openbmb/VoxCPM2")
MODEL = VoxCPM.from_pretrained(
    MODEL_ID,
    load_denoiser=False,
    device="auto",
    optimize=False,
)


def handler(event: dict) -> dict:
    request = event.get("input", {})
    text = str(request.get("text", "")).strip()
    mode = request.get("mode", "design")
    if not text:
        raise ValueError("text is required")

    reference_path: str | None = None
    encoded_reference = request.get("reference_audio_base64")
    if encoded_reference:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as reference:
            reference.write(base64.b64decode(encoded_reference))
            reference_path = reference.name

    try:
        control_text = str(request.get("voice_prompt", "")).strip()
        if mode == "design" and control_text:
            text = f"({control_text}){text}"
        kwargs = {
            "text": text,
            "cfg_value": float(request.get("cfg_value", 2.0)),
            "inference_timesteps": int(request.get("inference_timesteps", 10)),
            "normalize": bool(request.get("normalize", False)),
            "denoise": bool(request.get("denoise", False)),
        }
        if request.get("seed") is not None:
            kwargs["seed"] = request.get("seed")
        if mode == "controllable_clone":
            kwargs["reference_wav_path"] = reference_path
        elif mode == "hi_fidelity_clone":
            kwargs["prompt_wav_path"] = reference_path
            kwargs["prompt_text"] = request.get("prompt_text")
            kwargs["reference_wav_path"] = reference_path

        wav = MODEL.generate(**kwargs)
        sample_rate = int(MODEL.tts_model.sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
            output_path = Path(output.name)
        try:
            sf.write(output_path, np.asarray(wav), sample_rate, subtype="PCM_16")
            audio = base64.b64encode(output_path.read_bytes()).decode("ascii")
        finally:
            output_path.unlink(missing_ok=True)
        return {"audio_base64": audio, "sample_rate": sample_rate}
    finally:
        if reference_path:
            Path(reference_path).unlink(missing_ok=True)


runpod.serverless.start({"handler": handler})

from __future__ import annotations

import base64
import inspect
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


def _accepted_params(*functions) -> set[str]:
    """Union of parameter names accepted by the given functions, ignoring *args/**kwargs catch-alls."""
    names: set[str] = set()
    for fn in functions:
        if fn is None:
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        for name, param in params.items():
            if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                names.add(name)
    return names


_GENERATE = getattr(MODEL, "generate", None)
ACCEPTED_KWARGS = _accepted_params(_GENERATE, getattr(MODEL, "_generate", None))


def _filter_kwargs(kwargs: dict) -> dict:
    """Drop parameters the installed VoxCPM build does not accept (e.g. seed on
    older releases) instead of crashing the whole worker with a TypeError."""
    if not ACCEPTED_KWARGS:
        return kwargs
    filtered = {k: v for k, v in kwargs.items() if k in ACCEPTED_KWARGS}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(f"[handler] dropped unsupported generation params: {dropped}", flush=True)
    return filtered


def handler(event: dict) -> dict:
    request = event.get("input", {})
    text = str(request.get("text", "")).strip()
    mode = str(request.get("mode", "design"))
    if not text:
        raise ValueError("text is required")
    if mode not in {"design", "controllable_clone", "hi_fidelity_clone"}:
        mode = "design"

    reference_path: str | None = None
    encoded_reference = request.get("reference_audio_base64")
    if encoded_reference:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as reference:
            reference.write(base64.b64decode(encoded_reference))
            reference_path = reference.name

    if mode in {"controllable_clone", "hi_fidelity_clone"} and not reference_path:
        raise ValueError(f"{mode} requires reference_audio_base64 with a readable audio file")
    if mode == "hi_fidelity_clone" and not str(request.get("prompt_text") or "").strip():
        raise ValueError("hi_fidelity_clone requires prompt_text (the exact transcript of the reference audio)")

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
        if mode in {"controllable_clone", "hi_fidelity_clone"}:
            # VoxCPM's zero-shot cloning API is prompt_wav_path (+ optional prompt_text).
            # Prefer whichever reference parameter the installed build actually supports.
            if "reference_wav_path" in ACCEPTED_KWARGS:
                kwargs["reference_wav_path"] = reference_path
            elif "prompt_wav_path" in ACCEPTED_KWARGS or not ACCEPTED_KWARGS:
                kwargs["prompt_wav_path"] = reference_path
            transcript = str(request.get("prompt_text") or "").strip()
            if transcript and ("prompt_text" in ACCEPTED_KWARGS or not ACCEPTED_KWARGS):
                kwargs["prompt_text"] = transcript
        kwargs = _filter_kwargs(kwargs)

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

from __future__ import annotations

import base64
import inspect
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import runpod
import soundfile as sf
from voxcpm import VoxCPM

# Ensure the worker directory is importable when handler.py is loaded from a
# different working directory (e.g. tests/test_worker.py loads it via
# importlib from the repo root).
import sys

_WORKER_DIR = str(Path(__file__).resolve().parent)
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from khmer_text_preparation import prepare_khmer_text


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

# VoxCPM2's LM has an 8192-token KV cache. Long reference audios overflow it
# during prompt prefill (RuntimeError: expanded size of the tensor (8192) must
# match the existing size (9656)...), so cap how much prompt audio we keep.
MAX_REFERENCE_AUDIO_SECONDS = float(os.getenv("MAX_REFERENCE_AUDIO_SECONDS", "10"))


def normalize_khmer_text(text: str) -> str:
    """Clean and normalize Khmer text for better VoxCPM pronunciation.
    
    Common issues in Khmer text that affect TTS:
    - Extra/repeated spaces
    - Zero-width joiners/non-joiners (U+200C, U+200D, U+FEFF)
    - Mixed Unicode normalization forms
    - Extra newlines
    - Mixed fullwidth/halfwidth characters
    - Leading/trailing whitespace
    
    This normalization ensures VoxCPM receives clean text and produces
    better pronunciation quality, similar to using external AI tools to
    prepare the text.
    """
    import unicodedata
    
    # 1. Normalize Unicode to NFC (composed form — most stable)
    text = unicodedata.normalize("NFC", text)
    
    # 2. Remove zero-width characters (silent but can break tokenization)
    text = text.replace("\u200c", "")  # zero-width non-joiner
    text = text.replace("\u200d", "")  # zero-width joiner
    text = text.replace("\ufeff", "")  # zero-width no-break space (BOM)
    text = text.replace("\u00ad", "")  # soft hyphen
    
    # 3. Normalize fullwidth to halfwidth for ASCII chars
    normalized = ""
    for ch in text:
        if "\uff01" <= ch <= "\uff5e":  # fullwidth ! to ~
            normalized += chr(ord(ch) - 0xFEE0)
        else:
            normalized += ch
    text = normalized
    
    # 4. Replace mixed whitespace (tabs, non-breaking spaces, etc.) with regular space
    import re
    text = re.sub(r"[ \t\u00a0\u2000-\u200b\u202f\u205f]", " ", text)
    
    # 5. Collapse multiple spaces into one (but preserve newlines)
    text = re.sub(r" {2,}", " ", text)
    
    # 6. Collapse multiple newlines into max 2 (one blank line)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # 7. Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    
    # 8. Final strip
    text = text.strip()
    
    return text


def _materialize_reference_wav(encoded_reference: str) -> str:
    """Decode an arbitrary base64 audio payload into a plain PCM WAV temp file.

    The desktop client uploads the reference file's raw bytes in whatever
    container it was recorded/saved in (WAV, MP3, M4A, FLAC, WebM...). libsndfile
    only parses a subset of those, and VoxCPM's loader needs a real WAV, so
    re-mux everything here instead of trusting the bytes to be WAV.
    Overly long references are trimmed to MAX_REFERENCE_AUDIO_SECONDS to keep
    the prompt cache inside the LM's 8192-token KV cache.
    """
    raw = base64.b64decode(encoded_reference)
    if not raw:
        raise ValueError("reference_audio_base64 is empty")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as reference:
        wav_path = Path(reference.name)
    try:
        try:
            data, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        except Exception:
            # libsndfile could not parse it (e.g. M4A/AAC/WebM). Fall back to
            # audioread via librosa, which shells out to ffmpeg when installed.
            raw_path = wav_path.with_suffix(".raw")
            try:
                raw_path.write_bytes(raw)
                import librosa

                audio, sample_rate = librosa.load(str(raw_path), sr=None, mono=False)
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim == 2:
                    audio = audio.T  # (channels, samples) -> (samples, channels) for soundfile
                data = audio
            except Exception as exc:
                raise ValueError(
                    "reference_audio_base64 could not be decoded as audio. "
                    "WAV, FLAC, MP3 and OGG are supported natively; M4A/AAC/WebM "
                    "additionally require ffmpeg in the worker image."
                ) from exc
            finally:
                raw_path.unlink(missing_ok=True)

        sample_rate = int(sample_rate)
        # VoxCPM2's audio VAE operates at 48kHz. If the reference audio is at a
        # different rate, VoxCPM's loader interprets the samples as 48kHz, which
        # speeds up or slows down the cloned speech proportionally. Resample to
        # 48kHz so the reference's timing is preserved during cloning.
        target_rate = 48000
        if sample_rate != target_rate:
            try:
                import librosa
                data = librosa.resample(data.T if data.ndim == 2 else data, orig_sr=sample_rate, target_sr=target_rate).T
            except Exception:
                # librosa unavailable — fall back to a simple linear interp.
                # x positions are normalized onto [0, 1) so xp and fp always
                # have matching lengths regardless of the resample ratio.
                old_len = data.shape[0]
                new_len = int(round(old_len * target_rate / sample_rate))
                if new_len < 1:
                    new_len = 1
                x_old = np.linspace(0.0, 1.0, old_len, endpoint=False)
                x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
                if data.ndim == 2:
                    data = np.column_stack(
                        [np.interp(x_new, x_old, data[:, ch]) for ch in range(data.shape[1])]
                    )
                else:
                    data = np.interp(x_new, x_old, data)
            sample_rate = target_rate
        frames = data.shape[0]
        max_frames = int(MAX_REFERENCE_AUDIO_SECONDS * sample_rate)
        if frames > max_frames > 0:
            print(
                f"[handler] reference audio too long: {frames / sample_rate:.1f}s exceeds "
                f"{MAX_REFERENCE_AUDIO_SECONDS:g}s limit (VoxCPM2 8192-token prompt cache); "
                f"keeping the first {MAX_REFERENCE_AUDIO_SECONDS:g}s",
                flush=True,
            )
            data = data[:max_frames]
        sf.write(wav_path, data, sample_rate, subtype="PCM_16")
    except Exception:
        wav_path.unlink(missing_ok=True)
        raise
    return str(wav_path)


def handler(event: dict) -> dict:
    request = event.get("input", {})
    text = str(request.get("text", "")).strip()
    text = prepare_khmer_text(text)  # VOXCPM text preparation engine
    mode = str(request.get("mode", "design"))
    if not text:
        raise ValueError("text is required")
    if mode not in {"design", "controllable_clone", "hi_fidelity_clone"}:
        mode = "design"

    reference_path: str | None = None
    encoded_reference = request.get("reference_audio_base64")
    if encoded_reference:
        reference_path = _materialize_reference_wav(encoded_reference)

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

        try:
            generated = MODEL.generate(**kwargs)
        except ValueError as exc:
            if "KV cache is full" in str(exc):
                raise ValueError(
                    "VoxCPM2's 8192-token KV cache overflowed. Use a shorter "
                    "reference clip (or lower MAX_REFERENCE_AUDIO_SECONDS) and/or "
                    "shorter text, then try again."
                ) from exc
            raise
        # voxcpm>=2.1 returns a generator that yields the waveform, while
        # voxcpm 2.0.x returns the numpy array directly. Consume whichever
        # we get so playback speed is never doubled by a wrapped generator.
        wav = next(generated) if hasattr(generated, "__next__") else generated
        wav = np.asarray(wav).squeeze()
        reported_rate = int(MODEL.tts_model.sample_rate)
        # VoxCPM2's audio VAE always outputs at 48kHz. Some wrapper builds report
        # a different rate via tts_model.sample_rate (e.g. 16000 or 24000); if we
        # trust that and write the WAV header with the wrong rate, playback sounds
        # proportionally fast/slow. Force 48000 for VoxCPM2 unless OUTPUT_SAMPLE_RATE
        # explicitly overrides it.
        override = int(os.getenv("OUTPUT_SAMPLE_RATE", "0"))
        if override > 0:
            sample_rate = override
            print(f"[handler] OUTPUT_SAMPLE_RATE override: {reported_rate} -> {override}", flush=True)
        elif "voxcpm2" in MODEL_ID.lower():
            sample_rate = 48000
            if reported_rate != 48000:
                print(f"[handler] wrapper reported {reported_rate} Hz but VoxCPM2 decoder outputs at 48000 Hz; forcing 48000 Hz (set OUTPUT_SAMPLE_RATE to override)", flush=True)
        else:
            sample_rate = reported_rate
        duration = float(len(wav)) / float(sample_rate)
        print(
            f"[handler] output: sample_rate={sample_rate} Hz, samples={len(wav)}, duration={duration:.2f}s",
            flush=True,
        )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
            output_path = Path(output.name)
        try:
            sf.write(output_path, wav, sample_rate, subtype="PCM_16")
            audio = base64.b64encode(output_path.read_bytes()).decode("ascii")
        finally:
            output_path.unlink(missing_ok=True)
        return {"audio_base64": audio, "sample_rate": sample_rate, "duration": round(duration, 3)}
    finally:
        if reference_path:
            Path(reference_path).unlink(missing_ok=True)


runpod.serverless.start({"handler": handler})

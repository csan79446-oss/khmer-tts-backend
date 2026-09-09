"""Local simulation tests for the RunPod worker handler (all deps stubbed)."""
import base64
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

captured: dict = {}


class FakeTTSModel:
    # Real voxcpm 2.0.3 sets this to audio_vae_config.out_sample_rate (48000 for VoxCPM2).
    sample_rate = 48000


class FakeModel:
    """Mimics official VoxCPM: no seed, no reference_wav_path params."""

    def __init__(self) -> None:
        self.tts_model = FakeTTSModel()

    def generate(self, text, prompt_wav_path=None, prompt_text=None, cfg_value=2.0,
                 inference_timesteps=10, normalize=False, denoise=False):
        captured["generate"] = dict(text=text, prompt_wav_path=prompt_wav_path,
                                    prompt_text=prompt_text, cfg_value=cfg_value,
                                    inference_timesteps=inference_timesteps, normalize=normalize, denoise=denoise)
        return [0.0, 0.1, 0.2]

    def _generate(self, text, prompt_wav_path=None, prompt_text=None, cfg_value=2.0,
                  inference_timesteps=10, normalize=False, denoise=False, streaming=False):
        raise RuntimeError("should not be called directly")


class VoxCPM2StyleModel(FakeModel):
    """Mimics a build that accepts reference_wav_path instead."""

    def generate(self, text, reference_wav_path=None, prompt_wav_path=None, prompt_text=None,
                 cfg_value=2.0, inference_timesteps=10, normalize=False, denoise=False):
        captured["generate"] = dict(text=text, reference_wav_path=reference_wav_path,
                                    prompt_wav_path=prompt_wav_path, prompt_text=prompt_text)
        return [0.0, 0.1]


class GeneratorModel(FakeModel):
    """Mimics voxcpm>=2.1 where generate() yields the waveform instead of returning it."""

    def generate(self, text, prompt_wav_path=None, prompt_text=None, cfg_value=2.0,
                 inference_timesteps=10, normalize=False, denoise=False):
        captured["generate"] = dict(text=text, prompt_wav_path=prompt_wav_path,
                                    prompt_text=prompt_text, cfg_value=cfg_value,
                                    inference_timesteps=inference_timesteps, normalize=normalize, denoise=denoise)
        yield np.array([0.0, 0.1, 0.2, 0.3], dtype="float32")


def install_stubs(model_cls):
    captured.clear()
    vox = types.ModuleType("voxcpm")

    class VoxCPM:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return model_cls()

    vox.VoxCPM = VoxCPM
    sys.modules["voxcpm"] = vox

    rp = types.ModuleType("runpod")
    serverless = types.ModuleType("runpod.serverless")
    serverless.start = lambda *a, **k: None
    rp.serverless = serverless
    sys.modules["runpod"] = rp
    sys.modules["runpod.serverless"] = serverless

    sf = types.ModuleType("soundfile")

    def fake_write(path, data, rate, subtype=None):
        Path(path).write_bytes(b"RIFFfake")
        captured["last_write"] = (data, rate)

    def fake_read(buf, dtype=None, always_2d=False):
        import numpy as np

        payload = buf.read() if hasattr(buf, "read") else Path(buf).read_bytes()
        if not payload.startswith(b"RIFF"):
            raise RuntimeError("stub: format not recognised")
        return (np.zeros((48000, 1), dtype="float32"), 24000)  # 2 seconds of audio at 24 kHz

    sf.write = fake_write
    sf.read = fake_read
    sys.modules["soundfile"] = sf


def load_handler():
    spec = importlib.util.spec_from_file_location("worker_handler", "backend/worker/handler.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wav_b64():
    return base64.b64encode(b"RIFF fake wav bytes").decode("ascii")


def main() -> None:
    # --- Official VoxCPM-style model (prompt_wav_path, no seed) ---
    install_stubs(FakeModel)
    mod = load_handler()

    # 1. design mode: seed must be dropped silently; handler uses the model's
    #    reported sample rate (48000 for VoxCPM2 under real voxcpm 2.0.3).
    result = mod.handler({"input": {"mode": "design", "text": "សួស្តី", "seed": 5}})
    assert result["sample_rate"] == 48000, f"expected 48000, got {result['sample_rate']}"
    assert result["audio_base64"]
    assert "seed" not in captured["generate"], f"seed leaked: {captured['generate']}"
    print("PASS 1: design mode drops unsupported seed and uses reported 48000 rate")

    # 2. design mode with voice_prompt prepends style text
    mod.handler({"input": {"mode": "design", "text": "hello", "voice_prompt": "warm male"}})
    assert captured["generate"]["text"] == "(warm male)hello", captured["generate"]["text"]
    print("PASS 2: design mode prepends voice_prompt")

    # 3. controllable_clone with reference -> prompt_wav_path (not reference_wav_path)
    mod.handler({"input": {"mode": "controllable_clone", "text": "hello", "reference_audio_base64": wav_b64()}})
    g = captured["generate"]
    assert g.get("prompt_wav_path"), f"reference not mapped: {g}"
    assert "reference_wav_path" not in g and g.get("prompt_text") is None, g
    print("PASS 3: controllable_clone maps reference -> prompt_wav_path, no prompt_text")

    # 4. hi_fidelity_clone with transcript -> prompt_wav_path + prompt_text
    mod.handler({"input": {"mode": "hi_fidelity_clone", "text": "hello",
                           "reference_audio_base64": wav_b64(), "prompt_text": "exact transcript"}})
    g = captured["generate"]
    assert g.get("prompt_wav_path") and g.get("prompt_text") == "exact transcript", g
    print("PASS 4: hi_fidelity_clone passes reference + transcript")

    # 5. controllable_clone without reference -> clear error
    try:
        mod.handler({"input": {"mode": "controllable_clone", "text": "hello"}})
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "reference_audio_base64" in str(exc)
    print("PASS 5: clone without reference raises clear error")

    # 6. hi_fidelity_clone without transcript -> clear error
    try:
        mod.handler({"input": {"mode": "hi_fidelity_clone", "text": "hello", "reference_audio_base64": wav_b64()}})
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "prompt_text" in str(exc)
    print("PASS 6: hi_fidelity without transcript raises clear error")

    # 7. unknown mode falls back to design
    mod.handler({"input": {"mode": "weird_mode", "text": "hello"}})
    assert captured["generate"].get("prompt_wav_path") is None, captured["generate"]
    print("PASS 7: unknown mode falls back to design")

    # --- VoxCPM2-style model that accepts reference_wav_path ---
    install_stubs(VoxCPM2StyleModel)
    mod = load_handler()
    mod.handler({"input": {"mode": "controllable_clone", "text": "hello", "reference_audio_base64": wav_b64()}})
    g = captured["generate"]
    assert g.get("reference_wav_path"), f"VoxCPM2 param not used: {g}"
    print("PASS 8: VoxCPM2-style builds get reference_wav_path")

    # 9. OUTPUT_SAMPLE_RATE env override wins over the spec
    import os
    os.environ["OUTPUT_SAMPLE_RATE"] = "44100"
    try:
        result = mod.handler({"input": {"mode": "design", "text": "hello"}})
    finally:
        del os.environ["OUTPUT_SAMPLE_RATE"]
    assert result["sample_rate"] == 44100, f"override failed: {result['sample_rate']}"
    print("PASS 9: OUTPUT_SAMPLE_RATE env override wins")

    # 10. undecodable reference audio -> clear ValueError, not a libsndfile traceback
    try:
        mod.handler({"input": {"mode": "controllable_clone", "text": "hello",
                               "reference_audio_base64": base64.b64encode(b"garbage bytes").decode("ascii")}})
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "could not be decoded as audio" in str(exc)
    print("PASS 10: undecodable reference raises a clear ValueError")

    # 11. overly long reference audio gets trimmed to MAX_REFERENCE_AUDIO_SECONDS
    #     (VoxCPM2's 8192-token KV cache overflows on long prompt audio otherwise)
    import os
    os.environ["MAX_REFERENCE_AUDIO_SECONDS"] = "1"
    try:
        install_stubs(FakeModel)
        mod = load_handler()  # re-import so the env var is picked up
        mod.handler({"input": {"mode": "controllable_clone", "text": "hello",
                               "reference_audio_base64": wav_b64()}})
        data, rate = captured["last_write"]
        assert len(data) <= rate, f"not trimmed: {len(data)} frames at {rate} Hz"
    finally:
        del os.environ["MAX_REFERENCE_AUDIO_SECONDS"]
    print("PASS 11: long reference audio trimmed to MAX_REFERENCE_AUDIO_SECONDS")

    # 12. voxcpm>=2.1 returns a generator from generate(); handler must consume it
    #     instead of wrapping it in a 0-d object array (which would crash sf.write).
    install_stubs(GeneratorModel)
    mod = load_handler()
    result = mod.handler({"input": {"mode": "design", "text": "hello"}})
    assert result["sample_rate"] == 48000, f"generator path: expected 48000, got {result['sample_rate']}"
    data, rate = captured["last_write"]
    assert len(data) == 4, f"generator path: expected 4 samples, got {len(data)}"
    assert rate == 48000, f"generator path: expected rate 48000, got {rate}"
    print("PASS 12: generate() generator is consumed (no 0-d object array wrap)")

    print("\nALL 12 WORKER TESTS PASSED")


if __name__ == "__main__":
    main()

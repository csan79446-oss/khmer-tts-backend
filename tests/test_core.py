"""End-to-end simulation of the direct-RunPod flow (no network)."""
import base64
from pathlib import Path
import sys

sys.path.insert(0, ".")

from app.core.config import AppConfig
from app.database.database import Database
from app.services.voice_prompt import build_voice_prompt
from app.services.backend_client import BackendClient, BackendError


def test_prompt_builder_ignores_empty_fields() -> None:
    result = build_voice_prompt({"gender": "male", "tone": "", "custom": "clear narrator"})
    assert "male" in result
    assert "clear narrator" in result
    assert "voice" in result.lower()


def test_database_round_trip(tmp_path: Path) -> None:
    database = Database(tmp_path / "test.db")
    voice_id = database.add_voice("Test", "Description", "VoxCPM2")
    assert voice_id > 0
    assert database.list_voices()[0]["name"] == "Test"
    database.set_setting("sample", {"enabled": True})
    assert database.get_setting("sample")["enabled"] is True
    database.close()


def test_backend_generation_polls_and_downloads(tmp_path: Path) -> None:
    class FakeBackend(BackendClient):
        def __init__(self) -> None:
            super().__init__("http://backend", "")
            self.polls = 0

        def submit_job(self, payload: dict) -> dict:
            return {"job_id": "job-1", "status": "queued"}

        def get_job(self, job_id: str) -> dict:
            self.polls += 1
            return {"status": "completed", "output_url": "https://backend/audio/job-1"}

        def download_result(self, job: dict, output_path: Path) -> Path:
            output_path.write_bytes(b"audio")
            return output_path

    output = tmp_path / "result.mp3"
    result = FakeBackend().generate_and_download({"mode": "design", "text": "សួស្តី"}, output, polling_interval=0)
    assert result == output
    assert output.read_bytes() == b"audio"


def test_runpod_completed_immediately(tmp_path: Path) -> None:
    """runsync returns COMPLETED with base64 audio -> download_result decodes it."""
    wav = b"RIFF....WAVEfmt fake audio bytes"
    encoded = base64.b64encode(wav).decode("ascii")

    class FakeRunPod(BackendClient):
        def __init__(self) -> None:
            super().__init__("https://api.runpod.ai/v2/testend/runsync", "key")

        def submit_job(self, payload):
            return {"id": "job-1", "status": "COMPLETED", "output": {"audio_base64": encoded, "sample_rate": 48000}}

    output = tmp_path / "result.wav"
    result = FakeRunPod().generate_and_download({"mode": "design", "text": "test"}, output, polling_interval=0)
    assert result == output
    assert output.read_bytes() == wav


def test_runpod_polling(tmp_path: Path) -> None:
    """runsync returns IN_QUEUE -> poll /status until COMPLETED."""
    wav = b"RIFF polled audio"
    encoded = base64.b64encode(wav).decode("ascii")

    class FakeRunPod(BackendClient):
        def __init__(self) -> None:
            super().__init__("https://api.runpod.ai/v2/testend/runsync", "key")
            self.polls = 0

        def submit_job(self, payload):
            return {"id": "job-2", "status": "IN_QUEUE"}

        def get_job(self, job_id):
            self.polls += 1
            return {"id": job_id, "status": "COMPLETED", "output": {"audio_base64": encoded}}

    output = tmp_path / "result2.wav"
    FakeRunPod().generate_and_download({"mode": "design", "text": "test"}, output, polling_interval=0)
    assert output.read_bytes() == wav


def test_runpod_failed_job(tmp_path: Path) -> None:
    class FakeRunPod(BackendClient):
        def __init__(self) -> None:
            super().__init__("https://api.runpod.ai/v2/testend/runsync", "key")

        def submit_job(self, payload):
            return {"id": "job-4", "status": "IN_QUEUE"}

        def get_job(self, job_id):
            return {"id": job_id, "status": "FAILED", "error": "Generation crashed"}

    try:
        FakeRunPod().generate_and_download({"mode": "design", "text": "test"}, tmp_path / "x.wav", polling_interval=0)
        raise AssertionError("should have raised")
    except BackendError as exc:
        assert "Generation crashed" in str(exc)


def test_runpod_top_level_audio_base64(tmp_path: Path) -> None:
    """Some wrappers put audio_base64 at top level instead of under output."""
    wav = b"RIFF top-level audio"
    encoded = base64.b64encode(wav).decode("ascii")

    class FakeRunPod(BackendClient):
        def __init__(self) -> None:
            super().__init__("https://api.runpod.ai/v2/testend/runsync", "key")

        def submit_job(self, payload):
            return {"id": "job-5", "status": "COMPLETED", "audio_base64": encoded}

    output = tmp_path / "result3.wav"
    FakeRunPod().generate_and_download({"mode": "design", "text": "test"}, output, polling_interval=0)
    assert output.read_bytes() == wav



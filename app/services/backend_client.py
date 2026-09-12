from __future__ import annotations

import base64
from typing import Any
import time
from pathlib import Path

import httpx


class BackendError(RuntimeError):
    pass


class BackendTimeout(BackendError):
    pass


class BackendClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_error:
            detail = ""
            try:
                payload = response.json()
                detail = str(payload.get("detail", "")) if isinstance(payload, dict) else ""
            except ValueError:
                detail = response.text.strip()
            message = detail or f"Backend request failed with HTTP {response.status_code}."
            raise BackendError(message)

    def health(self) -> dict[str, Any]:
        if not self.base_url:
            raise BackendError("Backend URL is not configured.")
        if "api.runpod.ai" in self.base_url:
            try:
                response = httpx.get("https://api.runpod.ai", headers=self.headers, timeout=self.timeout)
                if response.status_code == 200:
                    return {"status": "reachable", "backend": "RunPod"}
            except Exception:
                pass
            return {"status": "configured", "backend": "RunPod", "note": "API key will be validated on first job submission"}
        response = httpx.get(f"{self.base_url}/health", headers=self.headers, timeout=self.timeout)
        self._raise_for_status(response)
        return response.json()

    def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.base_url:
            raise BackendError("Backend URL is not configured. Configure it in Settings first.")
        if "api.runpod.ai" in self.base_url:
            response = httpx.post(self.base_url, json={"input": payload}, headers=self.headers, timeout=600.0)
        else:
            response = httpx.post(f"{self.base_url}/jobs", json=payload, headers=self.headers, timeout=self.timeout)
        self._raise_for_status(response)
        return response.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        if not self.base_url:
            raise BackendError("Backend URL is not configured. Configure it in Settings first.")
        if "api.runpod.ai" in self.base_url:
            status_url = self.base_url.replace("/runsync", "/status").replace("/run", "/status")
            response = httpx.get(f"{status_url}/{job_id}", headers=self.headers, timeout=self.timeout)
            self._raise_for_status(response)
            return response.json()
        response = httpx.get(f"{self.base_url}/jobs/{job_id}", headers=self.headers, timeout=self.timeout)
        self._raise_for_status(response)
        return response.json()

    def generate_and_download(
        self,
        payload: dict[str, Any],
        output_path: Path,
        polling_interval: float = 2.0,
        max_wait: float = 900.0,
    ) -> Path:
        job = self.submit_job(payload)
        state = str(job.get("status", "")).lower()
        if state == "completed" or job.get("output_url") or job.get("audio_url"):
            return self.download_result(job, output_path)

        job_id = job.get("job_id") or job.get("id")
        if not job_id:
            raise BackendError("The backend did not return a job ID.")

        started = time.monotonic()
        while time.monotonic() - started < max_wait:
            time.sleep(polling_interval)
            status = self.get_job(str(job_id))
            state = str(status.get("status", "")).lower()
            if state in {"failed", "cancelled", "timed_out"}:
                raise BackendError(status.get("error", f"VoxCPM2 job {state}."))
            if state == "completed" or status.get("output_url") or status.get("audio_url"):
                return self.download_result(status, output_path)
        raise BackendTimeout("VoxCPM2 generation timed out. Check backend and RunPod status.")

    def download_result(self, job: dict[str, Any], output_path: Path) -> Path:
        # RunPod workers return inline base64 audio under "audio_base64" (top level
        # or nested inside "output"/"result"). Decode it directly when present.
        result = job.get("result") or job.get("output") or {}
        if not isinstance(result, dict):
            result = {}
        audio_b64 = job.get("audio_base64") or result.get("audio_base64")
        if audio_b64:
            output_path.write_bytes(base64.b64decode(audio_b64))
            return output_path

        result_url = job.get("output_url") or job.get("audio_url")
        if not result_url:
            result_url = result.get("output_url") or result.get("audio_url")
        if not result_url:
            raise BackendError("The backend completed the job without an audio result.")
        if result_url.startswith("/"):
            result_url = f"{self.base_url}{result_url}"
        response = httpx.get(result_url, headers=self.headers, timeout=self.timeout)
        self._raise_for_status(response)
        output_path.write_bytes(response.content)
        return output_path

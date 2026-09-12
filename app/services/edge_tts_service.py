from __future__ import annotations

import asyncio
from pathlib import Path


class EdgeTTSUnavailable(RuntimeError):
    pass


class EdgeTTSTimeout(RuntimeError):
    pass


class EdgeTTSService:
    async def list_voices(self) -> list[dict]:
        try:
            import edge_tts
        except ImportError as exc:
            raise EdgeTTSUnavailable("Install edge-tts to use Edge-TTS.") from exc
        return await edge_tts.list_voices()

    async def synthesize(
        self,
        text: str,
        voice: str,
        output_path: Path,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> Path:
        try:
            import edge_tts
        except ImportError as exc:
            raise EdgeTTSUnavailable("Install edge-tts to use Edge-TTS.") from exc
        if not text.strip():
            raise ValueError("Text cannot be empty.")
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
        try:
            await asyncio.wait_for(communicate.save(str(output_path)), timeout=45)
        except asyncio.TimeoutError as exc:
            raise EdgeTTSTimeout("Edge-TTS took too long to respond. Check your internet connection and try again.") from exc
        return output_path

    def synthesize_sync(
        self,
        text: str,
        voice: str,
        output_path: Path,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
    ) -> Path:
        return asyncio.run(self.synthesize(text, voice, output_path, rate, pitch, volume))

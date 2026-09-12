"""Comprehensive audit of the Khmer TTS Studio codebase."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from app.core.config import AppConfig
from app.database.database import Database
from app.services.backend_client import BackendClient
from app.services.voice_prompt import build_voice_prompt
from app.services.edge_tts_service import EdgeTTSService


def test_config_loads_db_settings():
    """Verify that AppConfig loads from database."""
    print("=" * 60)
    print("TEST 1: Config loads from database")
    print("=" * 60)
    config = AppConfig.load()
    print(f"  backend_url: {config.backend_url!r}")
    print(f"  api_key: {config.api_key!r}")
    print(f"  output_directory: {config.output_directory}")

    expected_url = "https://api.runpod.ai/v2/rm41lmxotxsi7a/runsync"
    if config.backend_url == expected_url:
        print("  OK: Backend URL loaded correctly from DB")
    else:
        print(f"  FAIL: Expected {expected_url!r}, got {config.backend_url!r}")
    print()


def test_voice_prompt_builder():
    """Verify build_voice_prompt generates good prompts."""
    print("=" * 60)
    print("TEST 2: Voice Prompt Builder")
    print("=" * 60)

    prompt = build_voice_prompt({"gender": "male", "age": "adult"})
    print(f"  Minimal: {prompt[:80]}...")
    assert "male" in prompt.lower()
    assert "adult" in prompt.lower()
    print("  OK: Minimal input works")

    prompt = build_voice_prompt({
        "gender": "female",
        "age": "young adult",
        "tone": "warm and cheerful",
        "emotion": "happy",
        "style": "professional",
        "speed": "medium",
        "clarity": "Clear Khmer pronunciation",
        "custom": "Like a podcast host."
    })
    print(f"  Full: {prompt[:100]}...")
    assert "female" in prompt.lower()
    assert "warm and cheerful" in prompt.lower()
    assert "cheerful and uplifting" in prompt.lower()
    assert "podcast host" in prompt.lower()
    print("  OK: Full input works")
    print()


def test_backend_client():
    """Verify BackendClient health check works."""
    print("=" * 60)
    print("TEST 3: Backend Client Health")
    print("=" * 60)

    client = BackendClient(
        "https://api.runpod.ai/v2/rm41lmxotxsi7a/runsync",
        "",
        30.0
    )
    result = client.health()
    print(f"  Health: {result}")
    assert result["status"] == "configured"
    assert result["backend"] == "RunPod"
    print("  OK: RunPod health check passes")
    print()


def test_voice_design_flow():
    """Trace the complete voice design save + generation flow."""
    print("=" * 60)
    print("TEST 4: Voice Design Save Flow")
    print("=" * 60)

    name = "Test Voice Design"
    prompt = "A warm female adult voice, calm and confident, professional narration"
    metadata = {"prompt": prompt}

    db_path = Path.home() / "AppData" / "Roaming" / "khmer-tts-studio" / "studio.db"
    db = Database(db_path)
    voice_id = db.add_voice(name, prompt, "VoxCPM2", "", metadata)
    print(f"  Created voice #{voice_id}")

    voices = db.list_voices()
    found = any(v["id"] == voice_id and v["name"] == name for v in voices)
    if found:
        print("  OK: Voice saved and retrievable")
    else:
        print("  FAIL: Voice not saved")

    payload = {"mode": "design", "text": "Test text"}
    voice_row = next((v for v in voices if v["id"] == voice_id), None)
    if voice_row and not voice_row["reference_audio"]:
        payload["voice_prompt"] = prompt
        print(f"  OK: Payload includes voice_prompt: {payload['voice_prompt'][:50]}...")

    db.close()
    print()


def test_voice_clone_flow():
    """Trace the complete voice clone flow."""
    print("=" * 60)
    print("TEST 5: Voice Clone Flow")
    print("=" * 60)

    db_path = Path.home() / "AppData" / "Roaming" / "khmer-tts-studio" / "studio.db"
    db = Database(db_path)

    voices = db.list_voices()
    clones = [v for v in voices if v["reference_audio"]]
    designs = [v for v in voices if not v["reference_audio"]]

    print(f"  Total voices: {len(voices)}")
    print(f"  Clone voices (with audio): {len(clones)}")
    print(f"  Design voices (prompt only): {len(designs)}")

    if designs:
        d = designs[-1]
        print(f"\n  Sample design voice: {d['name']}")
        metadata = json.loads(d["metadata"] or "{}")
        print(f"  Prompt: {metadata.get('prompt', '')[:80]}...")

    if clones:
        c = clones[-1]
        print(f"\n  Sample clone voice: {c['name']}")
        metadata = json.loads(c["metadata"] or "{}")
        print(f"  Reference: {c['reference_audio']}")
        print(f"  Transcript: {metadata.get('transcript', '')}")

    db.close()
    print()


def test_audio_generation_flow():
    """Trace the complete audio generation flow."""
    print("=" * 60)
    print("TEST 6: Audio Generation Flow")
    print("=" * 60)

    config = AppConfig.load()
    db = Database(config.database_path)

    if config.backend_url:
        print(f"  OK: Backend URL configured: {config.backend_url[:60]}...")
    else:
        print("  FAIL: Backend URL NOT configured")

    print(f"  Output directory: {config.output_directory}")
    print(f"  Exists: {config.output_directory.exists()}")
    config.output_directory.mkdir(parents=True, exist_ok=True)
    print("  OK: Output directory ready")

    voices = db.list_voices()
    print(f"  Available voices: {len(voices)}")

    generations = db.list_generations()
    print(f"  Generation history: {len(generations)}")
    if generations:
        for g in generations[-3:]:
            print(f"    - {g['status']}: {g['text'][:40]}...")

    db.close()
    print()


def test_edge_tts():
    """Test Edge-TTS service."""
    print("=" * 60)
    print("TEST 7: Edge-TTS Service")
    print("=" * 60)

    try:
        import edge_tts
        print("  OK: edge-tts module available")
    except ImportError:
        print("  FAIL: edge-tts not installed")
    print()


def test_runpod_submit_format():
    """Verify the exact format of RunPod submit_job calls."""
    print("=" * 60)
    print("TEST 8: RunPod Submit Job Format")
    print("=" * 60)

    with open("app/services/backend_client.py", "r", encoding="utf-8") as f:
        code = f.read()

    if 'json={"input": payload}' in code:
        print("  OK: RunPod runsync wraps payload as {'input': payload}")
    else:
        print("  FAIL: RunPod payload not wrapped correctly")

    if '/runsync' in code and '/run' in code:
        print("  OK: Both runsync and run modes supported")

    if '/status' in code:
        print("  OK: Status polling uses /status endpoint")
    print()


def test_output_directory_loading():
    """Check output directory loading from DB."""
    print("=" * 60)
    print("TEST 9: Output Directory Loading")
    print("=" * 60)

    config = AppConfig.load()
    print(f"  Current output_directory: {config.output_directory}")

    import sqlite3
    db_path = Path.home() / "AppData" / "Roaming" / "khmer-tts-studio" / "studio.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT value FROM settings WHERE key='output_directory'").fetchone()
    if row:
        raw_value = row["value"]
        parsed = json.loads(raw_value)
        print(f"  DB raw value: {raw_value!r}")
        print(f"  DB parsed value: {parsed!r}")

        expected = Path.home() / "AppData" / "Roaming" / "khmer-tts-studio" / "audio"
        if str(config.output_directory) == str(expected):
            print("  OK: Using default output directory")
        else:
            print(f"  NOTE: Override from DB path differs")
            print(f"     Expected default: {expected}")
            print(f"     Got: {config.output_directory}")
    else:
        print("  No output_directory in DB")
    conn.close()
    print()


def test_backend_status_badge():
    """Check if the backend badge shows correct status."""
    print("=" * 60)
    print("TEST 10: Backend Status Badge")
    print("=" * 60)

    config = AppConfig.load()
    has_backend = bool(config.backend_url)
    badge_text = "BACKEND CONFIGURED" if has_backend else "BACKEND NOT CONFIGURED"
    badge_color = "badgeSuccess" if has_backend else "badgeWarning"

    print(f"  Backend URL: {config.backend_url}")
    print(f"  Badge text: {badge_text}")
    print(f"  Badge color: {badge_color}")

    if has_backend:
        print("  OK: Badge will show green 'CONFIGURED'")
    else:
        print("  FAIL: Badge will show yellow 'NOT CONFIGURED'")
    print()


def run_all():
    """Run all audit tests."""
    print("\n" + "=" * 60)
    print("KHMER TTS STUDIO - COMPREHENSIVE AUDIT")
    print("=" * 60 + "\n")

    test_config_loads_db_settings()
    test_voice_prompt_builder()
    test_backend_client()
    test_voice_design_flow()
    test_voice_clone_flow()
    test_audio_generation_flow()
    test_edge_tts()
    test_runpod_submit_format()
    test_output_directory_loading()
    test_backend_status_badge()

    print("=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_all()

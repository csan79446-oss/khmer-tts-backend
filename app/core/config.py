from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


APP_NAME = "KHMER TTS STUDIO"
APP_SLUG = "khmer-tts-studio"
DATA_DIR = Path(os.getenv("APPDATA", Path.home())) / APP_SLUG


@dataclass(slots=True)
class AppConfig:
    backend_url: str = ""
    api_key: str = ""
    default_engine: str = "VoxCPM2"
    output_directory: Path = DATA_DIR / "audio"
    connection_timeout: float = 30.0
    polling_interval: float = 2.0
    save_history: bool = True

    @classmethod
    def load(cls) -> "AppConfig":
        load_dotenv()
        output = Path(os.getenv("OUTPUT_DIRECTORY", str(DATA_DIR / "audio")))
        config = cls(
            backend_url=os.getenv("BACKEND_URL", "").rstrip("/"),
            api_key=os.getenv("API_KEY", ""),
            default_engine=os.getenv("DEFAULT_ENGINE", "VoxCPM2"),
            output_directory=output,
            connection_timeout=float(os.getenv("CONNECTION_TIMEOUT", "30")),
            polling_interval=float(os.getenv("POLLING_INTERVAL", "2")),
            save_history=os.getenv("SAVE_GENERATION_HISTORY", "true").lower() != "false",
        )
        config.ensure_directories()

        # Merge database settings as fallback when env vars are not set
        try:
            from app.database.database import Database

            db = Database(config.database_path)
            if not config.backend_url:
                db_url = db.get_setting("backend_url")
                if db_url:
                    config.backend_url = str(db_url)
            if not config.api_key:
                db_key = db.get_setting("api_key")
                if db_key:
                    config.api_key = str(db_key)
            if not str(output).strip() or str(output).strip().startswith('"'):
                db_dir = db.get_setting("output_directory")
                if db_dir:
                    config.output_directory = Path(str(db_dir).strip('"'))
            db.close()
        except Exception:
            # Database may not exist yet on first launch; env vars are sufficient
            pass

        return config

    def ensure_directories(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "references").mkdir(parents=True, exist_ok=True)

    @property
    def database_path(self) -> Path:
        return DATA_DIR / "studio.db"

    @property
    def references_directory(self) -> Path:
        return DATA_DIR / "references"

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS voices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                engine TEXT NOT NULL,
                reference_audio TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                engine TEXT NOT NULL,
                voice TEXT DEFAULT '',
                output_file TEXT,
                created_at TEXT NOT NULL,
                duration REAL DEFAULT 0,
                status TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def set_setting(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.connection.commit()

    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.connection.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row["value"])

    def add_voice(self, name: str, description: str, engine: str, reference_audio: str = "", metadata: dict | None = None) -> int:
        now = self.now()
        cursor = self.connection.execute(
            "INSERT INTO voices(name, description, engine, reference_audio, created_at, updated_at, metadata) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (name, description, engine, reference_audio, now, now, json.dumps(metadata or {})),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_voices(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM voices ORDER BY updated_at DESC"))

    def delete_voice(self, voice_id: int) -> None:
        self.connection.execute("DELETE FROM voices WHERE id=?", (voice_id,))
        self.connection.commit()

    def add_generation(self, text: str, engine: str, voice: str, output_file: str, status: str, metadata: dict | None = None) -> int:
        cursor = self.connection.execute(
            "INSERT INTO generations(text, engine, voice, output_file, created_at, status, metadata) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (text, engine, voice, output_file, self.now(), status, json.dumps(metadata or {})),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_generations(self) -> list[sqlite3.Row]:
        return list(self.connection.execute("SELECT * FROM generations ORDER BY created_at DESC"))

    def cleanup_old_generations(self, keep: int = 10) -> None:
        """Remove generations beyond the top `keep` most recent ones."""
        result = self.connection.execute(
            "SELECT COUNT(*) as cnt FROM generations"
        ).fetchone()
        total = result["cnt"]
        if total <= keep:
            return
        to_delete = total - keep
        ids_to_delete = self.connection.execute(
            "SELECT id FROM generations ORDER BY created_at ASC LIMIT ?",
            (to_delete,),
        ).fetchall()
        for row in ids_to_delete:
            self.connection.execute("DELETE FROM generations WHERE id=?", (row["id"],))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

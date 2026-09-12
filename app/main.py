from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.core.config import AppConfig
from app.database.database import Database
from app.ui.main_window import MainWindow


def main() -> int:
    config = AppConfig.load()
    database = Database(config.database_path)
    application = QApplication(sys.argv)
    application.setApplicationName("KHMER TTS STUDIO")
    window = MainWindow(config, database)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())

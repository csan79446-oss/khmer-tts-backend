from __future__ import annotations

import asyncio
import base64
import json
import shutil
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QPlainTextEdit,
    QProgressBar, QScrollArea, QSlider, QSpinBox, QStackedWidget, QVBoxLayout,
    QWidget,
)

from app.core.config import APP_NAME, AppConfig
from app.database.database import Database
from app.services.backend_client import BackendClient, BackendError, BackendTimeout
from app.services.edge_tts_service import EdgeTTSService, EdgeTTSUnavailable, EdgeTTSTimeout
from app.services.voice_prompt import build_voice_prompt


class GenerationWorker(QObject):
    finished = Signal(str, str)
    failed = Signal(str)

    def __init__(self, engine: str, text: str, output: Path, edge_voice: str, edge_rate: str, edge_pitch: str, backend: BackendClient, payload: dict | None = None) -> None:
        super().__init__()
        self.engine = engine
        self.text = text
        self.output = output
        self.edge_voice = edge_voice
        self.edge_rate = edge_rate
        self.edge_pitch = edge_pitch
        self.backend = backend
        self.payload = payload or {"mode": "design", "text": text}

    def run(self) -> None:
        try:
            if self.engine == "Edge-TTS":
                EdgeTTSService().synthesize_sync(self.text, self.edge_voice, self.output, self.edge_rate, self.edge_pitch)
                self.finished.emit(str(self.output), "Edge-TTS generation completed")
                return
            self.backend.generate_and_download(self.payload, self.output)
            self.finished.emit(str(self.output), "VoxCPM2 generation completed")
        except (BackendError, BackendTimeout, EdgeTTSUnavailable, EdgeTTSTimeout, ValueError, OSError) as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Generation failed: {exc}")


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, database: Database) -> None:
        super().__init__()
        self.config = config
        self.database = database
        self.backend = BackendClient(config.backend_url, config.api_key, config.connection_timeout)
        self.worker_thread: QThread | None = None
        self.generation_worker: GenerationWorker | None = None
        self.current_output: Path | None = None
        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 650)
        self.setMinimumSize(900, 600)
        self.build_ui()
        self.apply_theme()
        self.show_page(0)

    def build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.sidebar = self.build_sidebar()
        root_layout.addWidget(self.sidebar)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(32, 24, 32, 24)
        content_layout.setSpacing(20)
        content_layout.addWidget(self.build_topbar())
        self.pages = QStackedWidget()
        for page in (self.dashboard_page(), self.tts_page(), self.clone_page(), self.design_page(), self.voices_page(), self.history_page(), self.settings_page()):
            self.pages.addWidget(page)
        content_layout.addWidget(self.pages, 1)
        root_layout.addWidget(content, 1)
        self.setCentralWidget(root)
        self.refresh_voice_presets()

    def build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(245)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(22, 28, 18, 22)
        brand = QLabel("KHMER\nTTS STUDIO")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        subtitle = QLabel("Voice creation workspace")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)
        layout.addSpacing(32)
        labels = ["Dashboard", "Text to Speech", "Voice Clone", "Voice Design", "My Voices", "History", "Settings"]
        self.nav_buttons: list[QPushButton] = []
        for index, label in enumerate(labels):
            button = QPushButton(label)
            button.setProperty("nav", True)
            button.clicked.connect(lambda checked=False, i=index: self.show_page(i))
            layout.addWidget(button)
            self.nav_buttons.append(button)
        layout.addStretch()
        status = QLabel("LOCAL STORAGE\nSQLite ready")
        status.setObjectName("sidebarStatus")
        layout.addWidget(status)
        return sidebar

    def build_topbar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        self.page_title = QLabel("Dashboard")
        self.page_title.setObjectName("pageTitle")
        layout.addWidget(self.page_title)
        layout.addStretch()
        self.backend_badge = QLabel("BACKEND  NOT CONFIGURED")
        self.backend_badge.setObjectName("badgeWarning")
        layout.addWidget(self.backend_badge)
        settings = QPushButton("Settings")
        settings.clicked.connect(lambda: self.show_page(6))
        layout.addWidget(settings)
        return bar

    def page_header(self, title: str, description: str) -> QVBoxLayout:
        layout = QVBoxLayout()
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        detail = QLabel(description)
        detail.setObjectName("muted")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        return layout

    def card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("card")
        return frame

    def dashboard_page(self) -> QWidget:
        page = QWidget()
        layout = self.page_header("Good to hear from you.", "Create clear, natural Khmer speech from a focused workspace.")
        metrics = QGridLayout()
        metrics.setSpacing(14)
        generations = len(self.database.list_generations())
        voices = len(self.database.list_voices())
        for column, (label, value, accent) in enumerate([("GENERATED AUDIO", str(generations), "#75d6bd"), ("SAVED VOICES", str(voices), "#f5c46a"), ("CURRENT ENGINE", self.config.default_engine, "#98a9ff"), ("LOCAL LIBRARY", "READY", "#e18b9b")]):
            frame = self.card()
            frame_layout = QVBoxLayout(frame)
            small = QLabel(label)
            small.setObjectName("metricLabel")
            number = QLabel(value)
            number.setObjectName("metricValue")
            number.setStyleSheet(f"color: {accent};")
            frame_layout.addWidget(small)
            frame_layout.addWidget(number)
            metrics.addWidget(frame, 0, column)
        layout.addLayout(metrics)
        row = QHBoxLayout()
        status_card = self.card()
        status_layout = QVBoxLayout(status_card)
        status_layout.addWidget(QLabel("SYSTEM STATUS", objectName="metricLabel"))
        for name, value in [("Backend", "Not configured" if not self.config.backend_url else "Configured"), ("VoxCPM2", "Remote GPU"), ("Edge-TTS", "Available when installed"), ("Storage", "SQLite ready")]:
            line = QHBoxLayout()
            line.addWidget(QLabel(name))
            line.addStretch()
            value_label = QLabel(value)
            value_label.setObjectName("statusValue")
            line.addWidget(value_label)
            status_layout.addLayout(line)
        row.addWidget(status_card, 1)
        quick = self.card()
        quick_layout = QVBoxLayout(quick)
        quick_layout.addWidget(QLabel("QUICK ACTIONS", objectName="metricLabel"))
        for text, page_index in [("Open text workspace", 1), ("Clone a reference voice", 2), ("Design a new voice", 3)]:
            action = QPushButton(text)
            action.clicked.connect(lambda checked=False, i=page_index: self.show_page(i))
            quick_layout.addWidget(action)
        row.addWidget(quick, 1)
        layout.addLayout(row)
        layout.addStretch()
        page.setLayout(layout)
        return page

    def tts_page(self) -> QWidget:
        page = QWidget()
        layout = self.page_header("Text to Speech", "Write or paste Khmer text, choose an engine, and generate an audio file.")
        workspace = QHBoxLayout()
        editor_card = self.card()
        editor_layout = QVBoxLayout(editor_card)
        editor_layout.addWidget(QLabel("SCRIPT", objectName="metricLabel"))
        self.text_editor = QPlainTextEdit()
        self.text_editor.setObjectName("ttsEditor")
        self.text_editor.setFont(QFont("Khmer OS Battambang", 24))
        self.text_editor.setPlaceholderText("សរសេរអត្ថបទខ្មែររបស់អ្នកនៅទីនេះ...\nYou can mix Khmer, English, and numbers.")
        self.text_editor.setMinimumHeight(300)
        self.text_editor.textChanged.connect(self.update_counts)
        editor_layout.addWidget(self.text_editor)
        counts = QHBoxLayout()
        self.character_count = QLabel("0 characters")
        self.word_count = QLabel("0 words")
        counts.addWidget(self.character_count)
        counts.addStretch()
        counts.addWidget(self.word_count)
        editor_layout.addLayout(counts)
        workspace.addWidget(editor_card, 2)
        controls = self.card()
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(QLabel("GENERATION", objectName="metricLabel"))
        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["VoxCPM2", "Edge-TTS"])
        self.engine_combo.setCurrentText(self.config.default_engine)
        controls_layout.addWidget(QLabel("Engine"))
        controls_layout.addWidget(self.engine_combo)
        self.voice_preset = QComboBox()
        controls_layout.addWidget(QLabel("Voice preset (design or clone)"))
        controls_layout.addWidget(self.voice_preset)
        self.tts_quality = QComboBox()
        self.tts_quality.addItem("Best (30 steps, slower)", 30)
        self.tts_quality.addItem("Standard (24 steps)", 24)
        self.tts_quality.setCurrentIndex(0)
        self.tts_quality.setToolTip(
            "LocDiT flow-matching diffusion steps for VoxCPM2. "
            "More steps → smoother, higher-quality audio but slower generation. "
            "Best (30) is the recommended default."
        )
        controls_layout.addWidget(QLabel("VoxCPM2 audio quality"))
        controls_layout.addWidget(self.tts_quality)
        self.edge_voice = QComboBox()
        self.edge_voice.addItems(["km-KH-SreymomNeural", "km-KH-PisethNeural", "en-US-AriaNeural"])
        controls_layout.addWidget(QLabel("Edge-TTS voice"))
        controls_layout.addWidget(self.edge_voice)
        self.edge_style = QComboBox()
        self.edge_style.addItems(["Natural", "Warm", "Clear"])
        controls_layout.addWidget(QLabel("Voice style"))
        controls_layout.addWidget(self.edge_style)
        self.progress = QProgressBar()
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        controls_layout.addWidget(self.progress)
        self.generation_status = QLabel("Idle")
        self.generation_status.setObjectName("muted")
        controls_layout.addWidget(self.generation_status)
        player_row = QHBoxLayout()
        play = QPushButton("Play / pause")
        play.clicked.connect(self.toggle_playback)
        open_file = QPushButton("Open file")
        open_file.clicked.connect(self.open_current_output)
        player_row.addWidget(play)
        player_row.addWidget(open_file)
        controls_layout.addLayout(player_row)
        generate = QPushButton("Generate audio")
        generate.setObjectName("primaryButton")
        generate.clicked.connect(self.generate_audio)
        controls_layout.addWidget(generate)
        clear = QPushButton("Clear editor")
        clear.clicked.connect(self.text_editor.clear)
        controls_layout.addWidget(clear)
        controls_layout.addStretch()
        workspace.addWidget(controls, 1)
        layout.addLayout(workspace)
        page.setLayout(layout)
        return page

    def clone_page(self) -> QWidget:
        page = QWidget()
        layout = self.page_header("Voice Clone", "Use a clean 5 to 30 second reference recording. The file is sent only when you submit a backend job.")
        card = self.card()
        form = QFormLayout(card)
        self.clone_path = QLineEdit()
        browse = QPushButton("Choose audio")
        browse.clicked.connect(self.choose_reference)
        path_row = QHBoxLayout()
        path_row.addWidget(self.clone_path)
        path_row.addWidget(browse)
        form.addRow("Reference audio", path_row)
        self.clone_name = QLineEdit()
        form.addRow("Voice name", self.clone_name)
        self.clone_description = QPlainTextEdit()
        self.clone_description.setMaximumHeight(110)
        form.addRow("Description", self.clone_description)
        self.clone_transcript = QPlainTextEdit()
        self.clone_transcript.setPlaceholderText("Optional exact transcript for high-fidelity cloning")
        self.clone_transcript.setMaximumHeight(110)
        form.addRow("Transcript", self.clone_transcript)
        self.clone_cfg = QDoubleSpinBox()
        self.clone_cfg.setRange(1.0, 3.0)
        self.clone_cfg.setSingleStep(0.1)
        self.clone_cfg.setDecimals(2)
        self.clone_cfg.setValue(float(self.database.get_setting("voxcpm_cfg", 2.0)))
        self.clone_cfg.setToolTip(
            "Guidance scale for VoxCPM2. Higher = follows this reference voice more strictly."
        )
        form.addRow("CFG (guidance scale)", self.clone_cfg)
        self.clone_denoise = QCheckBox("Enhance reference audio (denoise)")
        self.clone_denoise.setChecked(bool(self.database.get_setting("voxcpm_denoise", True)))
        self.clone_denoise.setToolTip(
            "Denoises this reference audio on the worker before cloning. "
            "Enable for noisy recordings; disable for clean studio takes."
        )
        form.addRow("", self.clone_denoise)
        submit = QPushButton("Validate and save profile")
        submit.setObjectName("primaryButton")
        submit.clicked.connect(self.save_clone_profile)
        form.addRow("", submit)
        layout.addWidget(card)
        layout.addStretch()
        page.setLayout(layout)
        return page

    def design_page(self) -> QWidget:
        page = QWidget()
        layout = self.page_header("Voice Design", "Paste or type your voice design prompt here and give it a name. This prompt will be sent to VoxCPM2 to generate your desired voice.")
        
        # Single prompt text box
        card = self.card()
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(15)
        card_layout.setContentsMargins(20, 20, 20, 20)
        
        instruction = QLabel("Enter your voice design prompt below. This should describe the voice characteristics you want, such as gender, age, tone, emotion, speaking style, speed, and clarity.")
        instruction.setObjectName("muted")
        instruction.setWordWrap(True)
        card_layout.addWidget(instruction)
        
        # Name field
        name_row = QHBoxLayout()
        name_label = QLabel("Voice Name:")
        name_label.setStyleSheet("font-weight: 600; color: #eef1f2;")
        self.design_name = QLineEdit()
        self.design_name.setPlaceholderText("e.g., Warm Female Narrator, Professional Male Announcer")
        name_row.addWidget(name_label)
        name_row.addWidget(self.design_name)
        card_layout.addLayout(name_row)
        
        self.design_prompt = QPlainTextEdit()
        self.design_prompt.setPlaceholderText("Example: A warm female adult voice with a calm and confident tone, speaking naturally at a moderate pace with clear Khmer pronunciation...")
        self.design_prompt.setObjectName("ttsEditor")
        self.design_prompt.setMinimumHeight(250)
        self.design_prompt.textChanged.connect(self.update_prompt_count)
        card_layout.addWidget(self.design_prompt)
        
        # Stats row
        stats_row = QHBoxLayout()
        self.prompt_char_count = QLabel("0 characters")
        self.prompt_word_count = QLabel("0 words")
        stats_row.addWidget(self.prompt_char_count)
        stats_row.addWidget(self.prompt_word_count)
        stats_row.addStretch()
        card_layout.addLayout(stats_row)
        
        cfg_row = QHBoxLayout()
        cfg_label = QLabel("CFG (guidance scale):")
        self.design_cfg = QDoubleSpinBox()
        self.design_cfg.setRange(1.0, 3.0)
        self.design_cfg.setSingleStep(0.1)
        self.design_cfg.setDecimals(2)
        self.design_cfg.setValue(float(self.database.get_setting("voxcpm_cfg", 2.0)))
        self.design_cfg.setToolTip(
            "Guidance scale for VoxCPM2. Higher = follows this voice description more strictly. "
            "2.0 is the recommended default for crisp Khmer articulation."
        )
        cfg_row.addWidget(cfg_label)
        cfg_row.addWidget(self.design_cfg)
        cfg_row.addStretch()
        card_layout.addLayout(cfg_row)

        # Action buttons
        button_row = QHBoxLayout()
        
        self.clear_prompt_button = QPushButton("Clear")
        self.clear_prompt_button.clicked.connect(self.clear_design_form)
        button_row.addWidget(self.clear_prompt_button)
        
        self.save_design_button = QPushButton("Save Voice Profile")
        self.save_design_button.setObjectName("primaryButton")
        self.save_design_button.clicked.connect(self.save_design_profile)
        button_row.addWidget(self.save_design_button)
        
        button_row.addStretch()
        card_layout.addLayout(button_row)
        
        layout.addWidget(card)
        layout.addStretch()
        page.setLayout(layout)
        return page

    def voices_page(self) -> QWidget:
        page = QWidget()
        layout = self.page_header("My Voices", "Local profiles keep names and metadata. Audio remains in your configured local folders.")
        self.voice_list = QListWidget()
        layout.addWidget(self.voice_list)
        refresh = QPushButton("Refresh library")
        refresh.clicked.connect(self.refresh_voices)
        layout.addWidget(refresh)
        self.refresh_voices()
        page.setLayout(layout)
        return page

    def history_page(self) -> QWidget:
        page = QWidget()
        layout = self.page_header("Generation History", "Every completed local generation is recorded here without storing audio blobs in SQLite.")
        self.history_list = QListWidget()
        layout.addWidget(self.history_list)
        refresh = QPushButton("Refresh history")
        refresh.clicked.connect(self.refresh_history)
        layout.addWidget(refresh)
        self.refresh_history()
        page.setLayout(layout)
        return page

    def settings_page(self) -> QWidget:
        page = QWidget()
        layout = self.page_header("Settings", "Use your FastAPI backend URL here, not the raw api.runpod.ai URL. The FastAPI service keeps RunPod credentials private.")
        card = self.card()
        form = QFormLayout(card)
        self.backend_url = QLineEdit(self.config.backend_url)
        self.backend_url.setPlaceholderText("Example: http://127.0.0.1:8000")
        self.api_key = QLineEdit(self.config.api_key)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.output_dir = QLineEdit(str(self.config.output_directory))
        form.addRow("Backend URL", self.backend_url)
        form.addRow("API key", self.api_key)
        form.addRow("Output directory", self.output_dir)
        buttons = QHBoxLayout()
        save = QPushButton("Save settings")
        save.setObjectName("primaryButton")
        save.clicked.connect(self.save_settings)
        test = QPushButton("Test connection")
        test.clicked.connect(self.test_connection)
        buttons.addWidget(save)
        buttons.addWidget(test)
        form.addRow("", buttons)
        layout.addWidget(card)
        layout.addStretch()
        page.setLayout(layout)
        return page

    def show_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        titles = ["Dashboard", "Text to Speech", "Voice Clone", "Voice Design", "My Voices", "History", "Settings"]
        self.page_title.setText(titles[index])
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)

    def update_counts(self) -> None:
        text = self.text_editor.toPlainText()
        self.character_count.setText(f"{len(text)} characters")
        self.word_count.setText(f"{len(text.split())} words")

    def update_prompt_count(self) -> None:
        text = self.design_prompt.toPlainText()
        self.prompt_char_count.setText(f"{len(text)} characters")
        self.prompt_word_count.setText(f"{len(text.split())} words")
    
    def clear_design_form(self) -> None:
        self.design_name.clear()
        self.design_prompt.clear()
        self.prompt_char_count.setText("0 characters")
        self.prompt_word_count.setText("0 words")

    def choose_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose reference audio", "", "Audio (*.wav *.mp3 *.flac *.m4a)")
        if path:
            self.clone_path.setText(path)

    def save_clone_profile(self) -> None:
        path = Path(self.clone_path.text())
        name = self.clone_name.text().strip()
        if not path.is_file():
            QMessageBox.warning(self, "Reference audio", "Choose a readable audio file first.")
            return
        if not name:
            QMessageBox.warning(self, "Voice name", "Enter a name for this voice profile.")
            return
        destination = self.config.references_directory / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{path.name}"
        shutil.copy2(path, destination)
        metadata = {
            "transcript": self.clone_transcript.toPlainText().strip(),
            "cfg": float(self.clone_cfg.value()),
            "denoise": bool(self.clone_denoise.isChecked()),
        }
        self.database.add_voice(name, self.clone_description.toPlainText().strip(), "VoxCPM2", str(destination), metadata)
        self.refresh_voices()
        QMessageBox.information(self, "Voice saved", "The local voice profile was saved.")

    def save_design_profile(self) -> None:
        name = self.design_name.text().strip()
        prompt = self.design_prompt.toPlainText().strip()
        if not name:
            QMessageBox.warning(self, "Voice name", "Please enter a name for this voice profile.")
            return
        if not prompt:
            QMessageBox.warning(self, "Voice design", "Enter a voice design prompt first.")
            return
        self.database.add_voice(name, prompt, "VoxCPM2", "", {"prompt": prompt, "cfg": float(self.design_cfg.value())})
        self.refresh_voices()
        QMessageBox.information(self, "Voice saved", f'Voice profile "{name}" has been saved.')
        self.clear_design_form()

    def generate_audio(self) -> None:
        if self.worker_thread and self.worker_thread.isRunning():
            self.generation_status.setText("A generation is already in progress...")
            return
        text = self.text_editor.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Text required", "Enter Khmer or mixed-language text before generating.")
            return
        engine = self.engine_combo.currentText()
        if engine == "VoxCPM2" and not self.config.backend_url:
            QMessageBox.warning(
                self,
                "Backend not configured",
                "Configure the Backend URL in Settings before generating with VoxCPM2."
            )
            return
        edge_styles = {
            "Natural": ("-5%", "+0Hz"),
            "Warm": ("-8%", "-2Hz"),
            "Clear": ("-2%", "+1Hz"),
        }
        edge_rate, edge_pitch = edge_styles[self.edge_style.currentText()]
        payload = self.build_generation_payload(text) if engine == "VoxCPM2" else None
        mode_note = f" ({payload['mode']})" if payload and payload.get("mode") != "design" else ""
        output_suffix = ".mp3" if engine == "Edge-TTS" else ".wav"
        self.current_output = self.config.output_directory / f"khmer_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output_suffix}"
        self.progress.show()
        self.generation_status.setText(f"Generating with {engine}{mode_note}...")
        self.worker_thread = QThread(self)
        self.generation_worker = GenerationWorker(engine, text, self.current_output, self.edge_voice.currentText(), edge_rate, edge_pitch, self.backend, payload)
        self.generation_worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.generation_worker.run)
        self.generation_worker.finished.connect(self.generation_finished)
        self.generation_worker.failed.connect(self.generation_failed)
        self.generation_worker.finished.connect(self.worker_thread.quit)
        self.generation_worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.generation_worker.deleteLater)
        self.worker_thread.finished.connect(self.generation_thread_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def generation_thread_finished(self) -> None:
        self.generation_worker = None
        self.worker_thread = None

    def generation_finished(self, output: str, message: str) -> None:
        self.progress.hide()
        self.generation_status.setText(message)
        if output:
            self.database.add_generation(self.text_editor.toPlainText(), self.engine_combo.currentText(), self.edge_voice.currentText(), output, "completed")
            self.database.cleanup_old_generations(keep=10)
            self.refresh_history()
            self.current_output = Path(output)
            self.player.setSource(QUrl.fromLocalFile(output))
            details = ""
            if output.lower().endswith(".wav"):
                try:
                    import wave
                    with wave.open(output, "rb") as w:
                        rate = w.getframerate()
                        seconds = w.getnframes() / float(rate)
                    details = f"\n\nWAV header: {rate} Hz, duration {seconds:.2f}s\n(If speech sounds fast/slow, this rate doesn't match the model's true output rate.)"
                except (wave.Error, OSError, EOFError):
                    pass
            QMessageBox.information(self, "Generation complete", f"Audio saved to:\n{output}{details}")

    def generation_failed(self, message: str) -> None:
        self.progress.hide()
        self.generation_status.setText("Failed")
        self.database.add_generation(self.text_editor.toPlainText(), self.engine_combo.currentText(), "", "", "failed", {"error": message})
        QMessageBox.warning(self, "Generation unavailable", message)

    def toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        elif self.current_output and self.current_output.is_file():
            self.player.play()
        else:
            QMessageBox.information(self, "No audio", "Generate or select an audio file first.")

    def open_current_output(self) -> None:
        if self.current_output and self.current_output.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_output)))
        else:
            QMessageBox.information(self, "No audio", "Generate an audio file first.")

    def refresh_voices(self) -> None:
        if not hasattr(self, "voice_list"):
            return
        self.voice_list.clear()
        for row in self.database.list_voices():
            item = QListWidgetItem(f"{row['name']}   |   {row['engine']}\n{row['description']}")
            item.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.voice_list.addItem(item)
        self.refresh_voice_presets()

    def refresh_voice_presets(self) -> None:
        if not hasattr(self, "voice_preset"):
            return
        current = self.voice_preset.currentData()
        self.voice_preset.blockSignals(True)
        self.voice_preset.clear()
        self.voice_preset.addItem("Default VoxCPM2 voice", None)
        for row in self.database.list_voices():
            kind = "clone" if row["reference_audio"] else "design"
            self.voice_preset.addItem(f"{row['name']} ({kind})", row["id"])
        if current is not None:
            index = self.voice_preset.findData(current)
            if index >= 0:
                self.voice_preset.setCurrentIndex(index)
        self.voice_preset.blockSignals(False)

    def build_generation_payload(self, text: str) -> dict:
        """Build the VoxCPM2 job payload from the selected voice preset.

        - No preset: plain `design` mode with just the text.
        - Design preset (saved prompt, no audio): adds `voice_prompt`.
        - Clone preset (reference audio): sends the audio as base64; uses
          `hi_fidelity_clone` when a transcript exists, else `controllable_clone`.
        - Khmer text preparation is always enabled for TTS-page generation.
        - CFG / denoise defaults come from the backend; per-voice metadata
          overrides are applied when a voice preset is selected.
        """
        payload: dict = {"mode": "design", "text": text}
        # Khmer text preparation is essential — always enable it.
        payload["text_preparation"] = True
        # Diffusion-step quality from the TTS page selector (Best=30, Standard=24).
        if hasattr(self, "tts_quality"):
            payload["inference_timesteps"] = int(self.tts_quality.currentData())

        if not hasattr(self, "voice_preset"):
            return payload
        voice_id = self.voice_preset.currentData()
        if voice_id is None:
            return payload
        row = next((r for r in self.database.list_voices() if r["id"] == voice_id), None)
        if row is None:
            return payload
        try:
            metadata = json.loads(row["metadata"] or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except (TypeError, ValueError):
            metadata = {}

        # Per-voice advanced settings override backend defaults.
        if "cfg" in metadata:
            try:
                payload["cfg_value"] = round(min(max(float(metadata["cfg"]), 1.0), 3.0), 2)
            except (TypeError, ValueError):
                pass
        if "denoise" in metadata:
            payload["denoise"] = bool(metadata["denoise"])

        reference = str(row["reference_audio"] or "").strip()
        reference_path = Path(reference) if reference else None
        if reference_path and reference_path.is_file():
            audio_b64 = base64.b64encode(reference_path.read_bytes()).decode("ascii")
            transcript = str(metadata.get("transcript") or "").strip()
            if transcript:
                payload.update({
                    "mode": "hi_fidelity_clone",
                    "reference_audio_base64": audio_b64,
                    "prompt_text": transcript,
                })
            else:
                payload.update({"mode": "controllable_clone", "reference_audio_base64": audio_b64})
            return payload
        if metadata.get("prompt"):
            payload["voice_prompt"] = str(metadata["prompt"])
        return payload

    def refresh_history(self) -> None:
        if not hasattr(self, "history_list"):
            return
        self.history_list.clear()
        for row in self.database.list_generations():
            text = row["text"].replace("\n", " ")[:90]
            self.history_list.addItem(f"{row['status'].upper()}   |   {row['engine']}   |   {text}")

    def save_settings(self) -> None:
        self.config.backend_url = self.backend_url.text().strip().rstrip("/")
        self.config.api_key = self.api_key.text()
        self.config.output_directory = Path(self.output_dir.text()).expanduser()
        self.config.ensure_directories()
        self.backend = BackendClient(self.config.backend_url, self.config.api_key, self.config.connection_timeout)
        self.database.set_setting("backend_url", self.config.backend_url)
        self.database.set_setting("output_directory", str(self.config.output_directory))
        self.backend_badge.setText("BACKEND  CONFIGURED" if self.config.backend_url else "BACKEND  NOT CONFIGURED")
        self.backend_badge.setObjectName("badgeSuccess" if self.config.backend_url else "badgeWarning")
        self.backend_badge.style().unpolish(self.backend_badge)
        self.backend_badge.style().polish(self.backend_badge)
        QMessageBox.information(self, "Settings saved", "Your desktop settings were saved.")

    def test_connection(self) -> None:
        try:
            backend_url = self.backend_url.text().strip().rstrip("/")
            api_key = self.api_key.text()
            self.backend = BackendClient(backend_url, api_key, self.config.connection_timeout)
            result = self.backend.health()
            QMessageBox.information(self, "Backend online", str(result))
        except Exception as exc:
            QMessageBox.warning(self, "Backend unavailable", str(exc))

    def apply_theme(self) -> None:
        self.setStyleSheet("""
        QWidget { background: #101318; color: #eef1f5; font-family: 'Segoe UI'; font-size: 12px; }
        #sidebar { background: #171b22; border-right: 1px solid #29303a; }
        #brand { color: #f5c46a; font-size: 21px; font-weight: 700; letter-spacing: 1px; }
        #pageTitle { font-size: 28px; font-weight: 650; color: #ffffff; }
        #sectionTitle { font-size: 24px; font-weight: 650; color: #ffffff; }
        #muted, QLabel#muted { color: #8f9aa8; }
        #sidebarStatus { color: #75d6bd; background: #20272a; border-radius: 6px; padding: 10px; }
        QPushButton { background: #232a33; color: #e9edf2; border: 1px solid #343e49; border-radius: 6px; padding: 10px 14px; }
        QPushButton:hover { background: #2c3744; border-color: #75d6bd; }
        QPushButton:checked, QPushButton[nav="true"]:checked { background: #31413f; color: #75d6bd; border-left: 3px solid #75d6bd; }
        QPushButton[nav="true"] { text-align: left; background: transparent; border: 0; padding: 12px 14px; }
        #primaryButton { background: #75d6bd; color: #10201e; font-weight: 700; border: 0; }
        #primaryButton:hover { background: #91e4ce; }
        #card { background: #191f27; border: 1px solid #2b3541; border-radius: 8px; }
        #metricLabel { color: #8f9aa8; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
        #metricValue { font-size: 26px; font-weight: 700; padding-top: 8px; }
        #statusValue { color: #75d6bd; }
        #badgeWarning, #badgeSuccess { border-radius: 12px; padding: 7px 12px; font-size: 11px; font-weight: 700; }
        #badgeWarning { color: #f5c46a; background: #403624; }
        #badgeSuccess { color: #75d6bd; background: #203c37; }
        QLineEdit, QPlainTextEdit, QComboBox, QSpinBox { background: #11161c; border: 1px solid #36404c; border-radius: 5px; padding: 8px; selection-background-color: #496e67; }
        #ttsEditor { font-size: 22px; }
        QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus { border-color: #75d6bd; }
        QListWidget { background: #191f27; border: 1px solid #2b3541; border-radius: 8px; padding: 8px; }
        QListWidget::item { padding: 14px; border-bottom: 1px solid #2b3541; }
        QListWidget::item:selected { background: #263b39; color: #ffffff; }
        QProgressBar { border: 0; background: #2b3541; height: 8px; border-radius: 4px; }
        QProgressBar::chunk { background: #75d6bd; border-radius: 4px; }
        QScrollBar:vertical { background: #161b21; width: 10px; }
        QScrollBar::handle:vertical { background: #3a4652; border-radius: 5px; }
        """)

    def closeEvent(self, event) -> None:
        self.database.close()
        event.accept()

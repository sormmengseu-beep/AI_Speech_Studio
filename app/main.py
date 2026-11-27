import logging
import sys
from pathlib import Path

from PySide6.QtCore import QIODevice, QSaveFile, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

if __package__:
    from .speech import synthesize, transcribe
    from .voice_clone import KHMER_LANGUAGE, clone_voice
else:
    from speech import synthesize, transcribe
    from voice_clone import KHMER_LANGUAGE, clone_voice


PROJECT_DIR = Path(__file__).resolve().parents[1]
LANGUAGES = (
    ("Detect automatically", None),
    ("English", "en"),
    ("Khmer", "km"),
    ("Thai", "th"),
    ("Vietnamese", "vi"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("Chinese", "zh"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
)
VOICES = (
    ("English (US) · Aria", "en-US-AriaNeural"),
    ("English (UK) · Sonia", "en-GB-SoniaNeural"),
    ("Khmer · Sreymom", "km-KH-SreymomNeural"),
    ("Thai · Premwadee", "th-TH-PremwadeeNeural"),
    ("Vietnamese · HoaiMy", "vi-VN-HoaiMyNeural"),
    ("French · Denise", "fr-FR-DeniseNeural"),
    ("Spanish · Elvira", "es-ES-ElviraNeural"),
)
CLONE_LANGUAGES = (
    "English",
    KHMER_LANGUAGE,
    "Chinese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
)


def label(text, style=""):
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    widget.setWordWrap(True)
    if style:
        widget.setObjectName(style)
    return widget


def button(text, action, primary=False):
    widget = QPushButton(text)
    if primary:
        widget.setObjectName("primary")
    widget.clicked.connect(action)
    return widget


def card(title):
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 20, 22, 22)
    layout.setSpacing(12)
    layout.addWidget(label(title, "cardTitle"))
    return frame, layout


def choices(name, options):
    combo = QComboBox()
    combo.setAccessibleName(name)
    for text, value in options:
        combo.addItem(text, value)
    return combo


def save_file(parent, title, filename, file_filter):
    output = PROJECT_DIR / "output"
    suggested = (output if output.is_dir() else Path.home()) / filename
    dialog = QFileDialog(parent, title, str(suggested), file_filter)
    dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    dialog.setDefaultSuffix(suggested.suffix[1:])
    if dialog.exec():
        return Path(dialog.selectedFiles()[0])
    return None


class SpeechTask(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, action, arguments, parent):
        super().__init__(parent)
        self.action = action
        self.arguments = arguments

    def run(self):
        try:
            result = self.action(*self.arguments)
        except Exception as error:
            logging.getLogger(__name__).exception("Speech task failed")
            self.failed.emit(str(error).strip() or type(error).__name__)
        else:
            self.succeeded.emit(result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Speech Studio")
        self.resize(1120, 720)
        self.setMinimumSize(900, 600)
        self._task = None
        self._audio_file = None
        self._cloned_file = None

        root = QWidget()
        root.setObjectName("studio")
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        rail = QFrame()
        rail.setObjectName("navigation")
        rail.setFixedWidth(216)
        navigation = QVBoxLayout(rail)
        navigation.setContentsMargins(18, 26, 18, 20)
        navigation.setSpacing(8)
        navigation.addWidget(label("AI Speech Studio", "brand"))
        navigation.addSpacing(25)
        self._navigation = QButtonGroup(self)
        for index, title in enumerate(
            ("Speech to text", "Text to speech", "Voice clone")
        ):
            tab = QPushButton(title)
            tab.setObjectName("navigationButton")
            tab.setCheckable(True)
            self._navigation.addButton(tab, index)
            navigation.addWidget(tab)
        navigation.addStretch()
        shell.addWidget(rail)

        workspace = QVBoxLayout()
        workspace.setContentsMargins(30, 25, 30, 20)
        workspace.setSpacing(18)
        shell.addLayout(workspace, 1)
        self._heading = label("", "heading")
        self._description = label("", "muted")
        header = QVBoxLayout()
        header.setSpacing(5)
        header.addWidget(self._heading)
        header.addWidget(self._description)
        workspace.addLayout(header)

        self._pages = QStackedWidget()
        self._pages.addWidget(self._transcription_page())
        self._pages.addWidget(self._synthesis_page())
        self._pages.addWidget(self._voice_clone_page())
        workspace.addWidget(self._pages, 1)

        footer = QHBoxLayout()
        footer.setSpacing(14)
        self._status = label("Ready", "status")
        self._status.setAccessibleName("Task status")
        footer.addWidget(self._status, 1)
        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 0)
        self._progress.setFixedSize(104, 5)
        self._progress.hide()
        footer.addWidget(self._progress)
        workspace.addLayout(footer)

        self._navigation.idClicked.connect(self._show_page)
        self._navigation.button(0).setChecked(True)
        self._show_page(0)
        self._update_actions()
        sheet = Path(__file__).with_name("ui") / "studio.qss"
        self.setStyleSheet(sheet.read_text(encoding="utf-8"))

    def _show_page(self, index):
        titles = ("Speech to text", "Text to speech", "Voice clone")
        descriptions = (
            "Turn an audio or video file into a transcript.",
            "Generate an MP3 from text with an online voice.",
            "Generate new speech in the voice of a short recording.",
        )
        self._heading.setText(titles[index])
        self._description.setText(descriptions[index])
        self._pages.setCurrentIndex(index)

    def _transcription_page(self):
        page = QWidget()
        columns = QHBoxLayout(page)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(16)

        source_card, source = card("Audio source")
        columns.addWidget(source_card, 4)
        self._source_path = QLineEdit()
        self._source_path.setAccessibleName("Selected audio or video file")
        self._source_path.setPlaceholderText("No file selected")
        self._source_path.setReadOnly(True)
        file_row = QHBoxLayout()
        file_row.addWidget(self._source_path, 1)
        file_row.addWidget(button("Browse…", self._choose_source))
        source.addLayout(file_row)
        source.addSpacing(6)
        source.addWidget(label("Spoken language", "fieldLabel"))
        self._language = choices("Spoken language", LANGUAGES)
        source.addWidget(self._language)
        source.addWidget(label("Whisper model", "fieldLabel"))
        self._model = choices(
            "Whisper model",
            (
                ("Tiny · Fastest", "tiny"),
                ("Base · Balanced", "base"),
                ("Small · Higher accuracy", "small"),
            ),
        )
        source.addWidget(self._model)
        source.addWidget(label("The first run downloads the selected model.", "muted"))
        source.addStretch()
        self._transcribe_button = button("Transcribe", self._transcribe, primary=True)
        source.addWidget(self._transcribe_button)

        transcript_card, transcript = card("Transcript")
        columns.addWidget(transcript_card, 6)
        self._transcript = QPlainTextEdit()
        self._transcript.setAccessibleName("Transcript")
        self._transcript.setPlaceholderText("Your transcript will appear here.")
        self._transcript.setReadOnly(True)
        transcript.addWidget(self._transcript, 1)
        actions = QHBoxLayout()
        actions.addStretch()
        self._copy_button = button("Copy", self._copy_transcript)
        self._save_button = button("Save text…", self._save_transcript)
        actions.addWidget(self._copy_button)
        actions.addWidget(self._save_button)
        transcript.addLayout(actions)
        return page

    def _synthesis_page(self):
        page = QWidget()
        columns = QHBoxLayout(page)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(16)

        text_card, text = card("Script")
        columns.addWidget(text_card, 6)
        self._script = QPlainTextEdit()
        self._script.setAccessibleName("Script to read aloud")
        self._script.setPlaceholderText("Type or paste the text to read aloud.")
        self._script.textChanged.connect(self._update_actions)
        text.addWidget(self._script, 1)

        voice_card, voice = card("Voice and audio")
        columns.addWidget(voice_card, 4)
        voice.addWidget(label("Voice", "fieldLabel"))
        self._voice = choices("Voice", VOICES)
        voice.addWidget(self._voice)
        voice.addWidget(
            label("Edge TTS sends your script to Microsoft to generate audio.", "muted")
        )
        voice.addSpacing(12)
        voice.addWidget(label("Latest export", "fieldLabel"))
        self._audio_output = label("No audio generated yet.", "muted")
        voice.addWidget(self._audio_output)
        self._play_button = button(
            "Open MP3", lambda: self._open_audio(self._audio_file)
        )
        voice.addWidget(self._play_button)
        voice.addStretch()
        self._generate_button = button("Generate MP3…", self._generate, primary=True)
        voice.addWidget(self._generate_button)
        return page

    def _voice_clone_page(self):
        page = QWidget()
        columns = QHBoxLayout(page)
        columns.setContentsMargins(0, 0, 0, 0)
        columns.setSpacing(16)

        reference_card, reference = card("Reference voice")
        columns.addWidget(reference_card, 4)
        self._reference_path = QLineEdit()
        self._reference_path.setAccessibleName("Reference recording")
        self._reference_path.setPlaceholderText("No recording selected")
        self._reference_path.setReadOnly(True)
        file_row = QHBoxLayout()
        file_row.addWidget(self._reference_path, 1)
        file_row.addWidget(button("Browse…", self._choose_reference))
        reference.addLayout(file_row)
        self._reference_hint = label(
            "For the clearest result, use 5–15 seconds of one speaker in a quiet room.",
            "muted",
        )
        reference.addWidget(self._reference_hint)
        self._reference_label = label("Reference transcript (optional)", "fieldLabel")
        reference.addWidget(self._reference_label)
        self._reference_text = QPlainTextEdit()
        self._reference_text.setAccessibleName("Reference transcript")
        self._reference_text.setPlaceholderText(
            "Enter the exact words in the recording."
        )
        self._reference_text.setMinimumHeight(72)
        self._reference_text.setMaximumHeight(116)
        self._reference_text.textChanged.connect(self._update_actions)
        reference.addWidget(self._reference_text)
        reference.addWidget(
            label("Matching the words helps preserve the speaker’s delivery.", "muted")
        )
        reference.addStretch()
        reference.addWidget(label("Output language", "fieldLabel"))
        self._clone_language = choices(
            "Output language",
            (
                (f"{name} (ភាសាខ្មែរ)" if name == KHMER_LANGUAGE else name, name)
                for name in CLONE_LANGUAGES
            ),
        )
        self._clone_language.currentIndexChanged.connect(self._change_clone_language)
        reference.addWidget(self._clone_language)

        script_card, script = card("Script")
        columns.addWidget(script_card, 6)
        self._clone_script = QPlainTextEdit()
        self._clone_script.setAccessibleName("Script for the cloned voice")
        self._clone_script.setPlaceholderText(
            "Type the new words for the voice to speak."
        )
        self._clone_script.textChanged.connect(self._update_actions)
        script.addWidget(self._clone_script, 1)
        self._clone_output = label("Exports as lossless WAV.", "muted")
        script.addWidget(self._clone_output)
        actions = QHBoxLayout()
        actions.addStretch()
        self._open_clone_button = button(
            "Open WAV", lambda: self._open_audio(self._cloned_file)
        )
        actions.addWidget(self._open_clone_button)
        self._clone_button = button("Generate WAV…", self._generate_clone, primary=True)
        actions.addWidget(self._clone_button)
        script.addLayout(actions)
        return page

    def _change_clone_language(self):
        khmer = self._clone_language.currentData() == KHMER_LANGUAGE
        self._reference_label.setText(
            "Reference transcript (required)"
            if khmer
            else "Reference transcript (optional)"
        )
        self._reference_hint.setText(
            "Use a quiet 4–9 second recording of one Khmer speaker."
            if khmer
            else "For the clearest result, use 5–15 seconds of one speaker in a quiet room."
        )
        self._update_actions()

    def _choose_source(self):
        self._choose_file(
            self._source_path,
            "Choose audio or video",
            "Media files (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus "
            "*.wma *.mp4 *.mkv *.mov *.webm);;All files (*)",
        )

    def _choose_reference(self):
        self._choose_file(
            self._reference_path,
            "Choose a reference recording",
            "Audio files (*.wav *.mp3 *.flac *.ogg *.aif *.aiff);;All files (*)",
        )

    def _choose_file(self, field, title, file_filter):
        file_path, _ = QFileDialog.getOpenFileName(
            self, title, field.text(), file_filter
        )
        if file_path:
            field.setText(file_path)
            field.setToolTip(file_path)
            self._update_actions()

    def _transcribe(self):
        self._start_task(
            "Loading the Whisper model and transcribing…",
            transcribe,
            (
                Path(self._source_path.text()),
                self._model.currentData(),
                self._language.currentData(),
            ),
            self._show_transcript,
        )

    @Slot(object)
    def _show_transcript(self, result):
        self._transcript.setPlainText(result)
        self._show_status("Transcript ready." if result else "No speech detected.")

    def _copy_transcript(self):
        QApplication.clipboard().setText(self._transcript.toPlainText())
        self._show_status("Transcript copied.")

    def _save_transcript(self):
        destination = save_file(
            self, "Save transcript", "transcript.txt", "Text (*.txt)"
        )
        if destination is None:
            return
        content = self._transcript.toPlainText().encode("utf-8")
        file = QSaveFile(str(destination))
        if (
            not file.open(QIODevice.OpenModeFlag.WriteOnly)
            or file.write(content) != len(content)
            or not file.commit()
        ):
            self._show_failure(file.errorString())
            return
        self._show_status(f"Saved {destination.name}.")

    def _generate(self):
        destination = save_file(
            self, "Save generated speech", "speech.mp3", "MP3 (*.mp3)"
        )
        if destination is None:
            return
        self._start_task(
            "Generating speech with Edge TTS…",
            synthesize,
            (
                self._script.toPlainText().strip(),
                self._voice.currentData(),
                destination,
            ),
            self._show_audio,
        )

    @Slot(object)
    def _show_audio(self, destination):
        self._audio_file = destination
        self._audio_output.setText(destination.name)
        self._audio_output.setToolTip(str(destination))
        self._show_status(f"Saved {destination.name}.")

    def _generate_clone(self):
        destination = save_file(
            self, "Save cloned speech", "cloned-voice.wav", "WAV audio (*.wav)"
        )
        if destination is None:
            return
        language = self._clone_language.currentData()
        status = (
            "Generating Khmer speech. This can take a few minutes on the CPU."
            if language == KHMER_LANGUAGE
            else "Preparing the voice model and generating speech…"
        )
        self._start_task(
            status,
            clone_voice,
            (
                Path(self._reference_path.text()),
                self._reference_text.toPlainText(),
                self._clone_script.toPlainText(),
                language,
                destination,
            ),
            self._show_clone,
        )

    @Slot(object)
    def _show_clone(self, destination):
        self._cloned_file = destination
        self._clone_output.setText(destination.name)
        self._clone_output.setToolTip(str(destination))
        self._show_status(f"Saved {destination.name}.")

    def _open_audio(self, path):
        if not path.is_file():
            self._show_failure("The audio file could not be found.")
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self._show_failure("Could not open the audio file.")

    def _update_actions(self):
        idle = self._task is None
        requires_transcript = self._clone_language.currentData() == KHMER_LANGUAGE
        self._transcribe_button.setEnabled(idle and bool(self._source_path.text()))
        self._generate_button.setEnabled(
            idle and bool(self._script.toPlainText().strip())
        )
        self._clone_button.setEnabled(
            idle
            and bool(self._reference_path.text())
            and bool(self._clone_script.toPlainText().strip())
            and (
                not requires_transcript
                or bool(self._reference_text.toPlainText().strip())
            )
        )
        has_transcript = bool(self._transcript.toPlainText())
        self._copy_button.setEnabled(has_transcript)
        self._save_button.setEnabled(has_transcript)
        self._play_button.setEnabled(self._audio_file is not None)
        self._open_clone_button.setEnabled(self._cloned_file is not None)

    def _start_task(self, status, action, arguments, completed):
        if self._task is not None:
            return
        self._task = SpeechTask(action, arguments, self)
        self._task.succeeded.connect(completed)
        self._task.failed.connect(self._show_failure)
        self._task.finished.connect(self._task_finished)
        self._task.finished.connect(self._task.deleteLater)
        self._show_status(status)
        self._progress.show()
        self._update_actions()
        self._task.start()

    @Slot()
    def _task_finished(self):
        self._task = None
        self._progress.hide()
        self._update_actions()

    def _show_status(self, message):
        visible = message.splitlines()[0]
        self._status.setText(visible if len(visible) <= 170 else visible[:167] + "…")
        self._status.setToolTip(message)
        self._status.setStyleSheet("")

    @Slot(str)
    def _show_failure(self, message):
        self._show_status(f"Could not complete the action. {message}")
        self._status.setStyleSheet("color: #ffb4a9;")

    def closeEvent(self, event):
        if self._task is not None:
            self._show_status(
                "The current task is still running. Try closing when it finishes."
            )
            event.ignore()
            return
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

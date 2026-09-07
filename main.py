from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from device_utils import ProcessingDevice, resolve_processing_device
from audio_utils import get_audio_duration
from speaker_diarizer import (
    LocalSpeakerDiarizer,
    assign_speakers,
    prepare_transcript_for_diarization,
)
from stt_engine import AnalysisCancelled, LocalSTTEngine, TranscriptSegment
from text_importer import ImportedTranscript, load_transcript_txt


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


STT_QUALITY_OPTIONS = {
    "빠름 (small)": "small",
    "균형 (medium)": "medium",
    "정확 (large-v3)": "large-v3",
}

LANGUAGE_OPTIONS = {
    "한국어": "ko",
    "자동 감지": None,
    "영어": "en",
    "중국어": "zh",
    "일본어": "ja",
}


class ModelLoadWorker(QThread):
    status = pyqtSignal(str)
    completed = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(self, kind: str, engine, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.engine = engine

    def run(self):
        try:
            self.engine.load_model(status_callback=self.status.emit)
            self.completed.emit(self.kind, self.engine)
        except Exception as exc:
            self.failed.emit(self.kind, str(exc))


class STTWorker(QThread):
    status = pyqtSignal(str)
    progress = pyqtSignal(str, int, str)
    completed = pyqtSignal(object)
    cancelled = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        audio_path: str,
        model_size: str,
        language: str | None,
        device: str,
        compute_type: str,
        stt_engine: LocalSTTEngine | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.audio_path = audio_path
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.stt_engine = stt_engine
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self):
        try:
            if (
                self.stt_engine is None
                or self.stt_engine.model_size != self.model_size
                or self.stt_engine.device != self.device
                or self.stt_engine.compute_type != self.compute_type
            ):
                self.stt_engine = LocalSTTEngine(
                    model_size=self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    language=self.language,
                )
            else:
                self.stt_engine.language = self.language

            def stt_progress(percent: int, current: float, total: float):
                detail = f"{format_time(current)} / {format_time(total)}"
                self.progress.emit("STT", percent, detail)

            segments = self.stt_engine.transcribe(
                self.audio_path,
                status_callback=self.status.emit,
                progress_callback=stt_progress,
                cancel_check=self._is_cancelled,
            )

            if self._is_cancelled():
                raise AnalysisCancelled()

            self.completed.emit(segments)
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class DiarizationWorker(QThread):
    status = pyqtSignal(str)
    progress = pyqtSignal(str, int, str)
    completed = pyqtSignal(object)
    cancelled = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        audio_path: str,
        transcript: list[TranscriptSegment],
        audio_duration: float,
        hf_token: str,
        device: str,
        diarizer: LocalSpeakerDiarizer | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.audio_path = audio_path
        self.transcript = transcript
        self.audio_duration = audio_duration
        self.hf_token = hf_token.strip()
        self.device = device
        self.diarizer = diarizer
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self):
        try:
            # Validate and normalize timestamps again inside the worker so
            # diarization can never silently label an incompatible transcript.
            transcript = prepare_transcript_for_diarization(
                self.transcript,
                self.audio_duration,
            )

            if self.diarizer is None or self.diarizer.device != self.device:
                self.diarizer = LocalSpeakerDiarizer(
                    hf_token=self.hf_token or os.getenv("HF_TOKEN"),
                    local_model_path=os.getenv("PYANNOTE_MODEL_PATH"),
                    device=self.device,
                )

            def diar_progress(step: str, percent: int):
                self.progress.emit(f"화자분리 · {step}", percent, "")

            turns = self.diarizer.diarize(
                self.audio_path,
                status_callback=self.status.emit,
                progress_callback=diar_progress,
                cancel_check=self._is_cancelled,
            )
            transcript = assign_speakers(transcript, turns)

            if self._is_cancelled():
                raise AnalysisCancelled()

            self.completed.emit(transcript)
        except AnalysisCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class SpeakerMappingPanel(QWidget):
    mapping_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.detected_speakers: list[str] = []
        self.visible_slots = 4
        self.name_inputs: dict[str, QLineEdit] = {}
        self.saved_names: dict[str, str] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("화자 이름 지정")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        self.info = QLabel("음성 분석 또는 구조화 TXT를 불러오면 화자가 표시됩니다.")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        content = QWidget()
        self.form = QFormLayout(content)
        self.scroll.setWidget(content)
        layout.addWidget(self.scroll, 1)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("+ 지정 슬롯")
        self.remove_button = QPushButton("마지막 슬롯 숨기기")
        self.add_button.clicked.connect(self.add_slot)
        self.remove_button.clicked.connect(self.remove_slot)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        layout.addLayout(buttons)

        self.refresh()

    def set_speakers(
        self,
        speakers: list[str],
        initial_names: dict[str, str] | None = None,
    ):
        self.detected_speakers = list(speakers)
        self.saved_names = dict(initial_names or {})
        self.visible_slots = 4
        self.refresh()
        self.mapping_changed.emit()

    def clear_form(self):
        while self.form.rowCount():
            self.form.removeRow(0)

    def _capture_names(self):
        for speaker_id, edit in self.name_inputs.items():
            value = edit.text().strip()
            if value:
                self.saved_names[speaker_id] = value
            elif speaker_id in self.saved_names:
                del self.saved_names[speaker_id]

    def refresh(self):
        self._capture_names()
        self.clear_form()
        self.name_inputs = {}

        count = len(self.detected_speakers)
        if count:
            self.info.setText(f"확인된 화자: {count}명 · 이름 지정 UI는 기본 4명부터 표시")
        else:
            self.info.setText("현재 화자 정보가 없습니다.")

        for index in range(self.visible_slots):
            if index < count:
                speaker_id = self.detected_speakers[index]
                edit = QLineEdit(self.saved_names.get(speaker_id, ""))
                edit.setPlaceholderText("이름 또는 직책 입력")
                edit.textChanged.connect(self._name_changed)
                self.name_inputs[speaker_id] = edit
                self.form.addRow(f"Speaker {index + 1}", edit)
            else:
                edit = QLineEdit()
                edit.setEnabled(False)
                edit.setPlaceholderText("확인된 화자 없음")
                self.form.addRow(f"Speaker {index + 1}", edit)

        self.add_button.setEnabled(self.visible_slots < count)
        self.remove_button.setEnabled(self.visible_slots > 4)

    def _name_changed(self):
        self._capture_names()
        self.mapping_changed.emit()

    def add_slot(self):
        if self.visible_slots < len(self.detected_speakers):
            self.visible_slots += 1
            self.refresh()
            self.mapping_changed.emit()

    def remove_slot(self):
        if self.visible_slots > 4:
            self.visible_slots -= 1
            self.refresh()
            self.mapping_changed.emit()

    def mapping(self) -> dict[str, str]:
        self._capture_names()
        return dict(self.saved_names)


class TranscriptionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Audio and transcript are intentionally independent inputs.
        # STT creates/replaces the transcript; diarization only labels whatever
        # timestamped transcript is currently loaded.
        self.audio_path = ""
        self.audio_duration: float | None = None
        self.transcript_source = "none"  # none | stt | txt_structured | txt_plain
        self.transcript_source_path = ""
        self.segments: list[TranscriptSegment] = []
        self.speaker_ids: list[str] = []
        self.worker: STTWorker | DiarizationWorker | None = None
        self.worker_kind = ""
        self.model_loader: ModelLoadWorker | None = None
        self.stt_engine: LocalSTTEngine | None = None
        self.diarizer: LocalSpeakerDiarizer | None = None
        self.processing_device: ProcessingDevice = resolve_processing_device(True)

        root = QVBoxLayout(self)

        source_title = QLabel("입력")
        source_title.setStyleSheet("font-weight: 600;")
        root.addWidget(source_title)

        audio_row = QHBoxLayout()
        self.audio_button = QPushButton("음성 파일 선택")
        self.audio_button.clicked.connect(self.choose_audio)
        self.audio_file_edit = QLineEdit()
        self.audio_file_edit.setReadOnly(True)
        self.audio_file_edit.setPlaceholderText("화자 라벨링의 기준이 될 음성 파일을 선택하세요.")
        audio_row.addWidget(self.audio_button)
        audio_row.addWidget(self.audio_file_edit, 1)
        root.addLayout(audio_row)

        txt_row = QHBoxLayout()
        self.txt_button = QPushButton("기존 TXT 불러오기")
        self.txt_button.clicked.connect(self.choose_txt)
        self.transcript_file_edit = QLineEdit()
        self.transcript_file_edit.setReadOnly(True)
        self.transcript_file_edit.setPlaceholderText(
            "선택 사항: 기존 타임라인 TXT를 불러오면 STT 없이 화자 라벨링할 수 있습니다."
        )
        txt_row.addWidget(self.txt_button)
        txt_row.addWidget(self.transcript_file_edit, 1)
        root.addLayout(txt_row)

        stt_title = QLabel("STT · 텍스트 추출")
        stt_title.setStyleSheet("font-weight: 600; margin-top: 10px;")
        root.addWidget(stt_title)

        stt_row = QHBoxLayout()
        stt_row.addWidget(QLabel("음성 언어"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(LANGUAGE_OPTIONS.keys())
        self.language_combo.setCurrentText("한국어")
        stt_row.addWidget(self.language_combo)

        stt_row.addWidget(QLabel("STT 품질"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(STT_QUALITY_OPTIONS.keys())
        self.quality_combo.setCurrentText("빠름 (small)")
        self.quality_combo.currentTextChanged.connect(self.on_stt_quality_changed)
        stt_row.addWidget(self.quality_combo)

        self.stt_model_status = QLabel("STT 모델: 준비 안 됨")
        self.stt_load_button = QPushButton("STT 모델 불러오기")
        self.stt_load_button.clicked.connect(self.preload_stt_model)
        self.stt_button = QPushButton("STT 실행")
        self.stt_button.clicked.connect(self.start_stt)
        self.stt_button.setEnabled(False)
        stt_row.addWidget(self.stt_model_status)
        stt_row.addWidget(self.stt_load_button)
        stt_row.addWidget(self.stt_button)
        stt_row.addStretch(1)
        root.addLayout(stt_row)

        performance_title = QLabel("성능 설정")
        performance_title.setStyleSheet("font-weight: 600; margin-top: 10px;")
        root.addWidget(performance_title)

        performance_row = QHBoxLayout()
        self.gpu_auto_check = QCheckBox("GPU 사용 가능하면 자동으로 사용")
        self.gpu_auto_check.setChecked(True)
        self.gpu_auto_check.toggled.connect(self.on_processing_device_changed)
        self.device_status = QLabel("")
        self.device_status.setStyleSheet("color: #555;")
        performance_row.addWidget(self.gpu_auto_check)
        performance_row.addSpacing(16)
        performance_row.addWidget(self.device_status)
        performance_row.addStretch(1)
        root.addLayout(performance_row)
        self.refresh_processing_device()

        diar_title = QLabel("화자 라벨링 · STT와 독립 실행")
        diar_title.setStyleSheet("font-weight: 600; margin-top: 10px;")
        root.addWidget(diar_title)

        diar_row = QHBoxLayout()
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("HF Token - 최초 pyannote 모델 다운로드 시 필요할 수 있음")
        self.diar_model_status = QLabel("화자 모델: 준비 안 됨")
        self.diar_load_button = QPushButton("화자 모델 불러오기")
        self.diar_load_button.clicked.connect(self.preload_diarization_model)
        self.diarize_button = QPushButton("현재 텍스트에 화자 라벨링")
        self.diarize_button.clicked.connect(self.start_diarization)
        self.diarize_button.setEnabled(False)
        self.cancel_button = QPushButton("작업 중지")
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.cancel_button.setEnabled(False)

        diar_row.addWidget(QLabel("HF Token"))
        diar_row.addWidget(self.token_edit, 1)
        diar_row.addWidget(self.diar_model_status)
        diar_row.addWidget(self.diar_load_button)
        diar_row.addWidget(self.diarize_button)
        diar_row.addWidget(self.cancel_button)
        root.addLayout(diar_row)

        self.diar_hint = QLabel(
            "화자 라벨링 조건: ① 음성 파일 존재 ② 텍스트 존재 ③ 각 발언에 타임라인 존재 "
            "④ 텍스트 타임라인이 음성 길이를 초과하지 않음. 일반 TXT(시간 없음)는 라벨링할 수 없습니다."
        )
        self.diar_hint.setWordWrap(True)
        self.diar_hint.setStyleSheet("color: #666;")
        root.addWidget(self.diar_hint)

        privacy = QLabel(
            "STT와 화자 라벨링은 서로 독립적으로 실행됩니다. STT가 성공한 뒤 화자 모델에서 오류가 나더라도 "
            "추출된 텍스트는 유지됩니다. 기존 타임라인 TXT + 원본 음성 조합으로도 화자 라벨링할 수 있습니다."
        )
        privacy.setWordWrap(True)
        privacy.setStyleSheet("color: #666;")
        root.addWidget(privacy)

        self.status_label = QLabel("음성 파일을 선택하거나 기존 TXT를 불러오세요.")
        root.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("대기 중")
        root.addWidget(self.progress_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.speaker_panel = SpeakerMappingPanel()
        self.speaker_panel.mapping_changed.connect(self.refresh_table_labels)
        splitter.addWidget(self.speaker_panel)

        transcript_widget = QWidget()
        transcript_layout = QVBoxLayout(transcript_widget)
        transcript_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["시간", "화자", "발언"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        transcript_layout.addWidget(self.table)

        save_row = QHBoxLayout()
        self.mode_label = QLabel("")
        self.mode_label.setStyleSheet("color: #666;")
        save_row.addWidget(self.mode_label)
        save_row.addStretch(1)
        self.save_button = QPushButton("TXT 저장")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_txt)
        save_row.addWidget(self.save_button)
        transcript_layout.addLayout(save_row)

        splitter.addWidget(transcript_widget)
        splitter.setSizes([280, 820])
        root.addWidget(splitter, 1)
        self.update_action_controls()

    def _default_open_dir(self) -> str:
        desktop = Path.home() / "Desktop"
        return str(desktop if desktop.exists() else Path.home())

    def choose_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "회의 음성 선택",
            self._default_open_dir(),
            "Audio Files (*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.wma);;All Files (*)",
        )
        if not path:
            return

        try:
            duration = get_audio_duration(path)
        except Exception as exc:
            QMessageBox.critical(self, "음성 파일 오류", str(exc))
            return

        self.audio_path = path
        self.audio_duration = duration
        self.audio_file_edit.setText(f"{path}   ({format_time(duration)})")
        self.status_label.setText(
            f"음성 파일 준비 완료 · 길이 {format_time(duration)}. STT 실행 또는 기존 타임라인 텍스트 화자 라벨링이 가능합니다."
        )
        self.update_mode_label()
        self.update_action_controls()

    def choose_txt(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "기존 Transcript TXT 선택",
            self._default_open_dir(),
            "Text Files (*.txt);;All Files (*)",
        )
        if not path:
            return

        try:
            imported = load_transcript_txt(path)
        except Exception as exc:
            QMessageBox.critical(self, "TXT 불러오기 오류", str(exc))
            return

        self.load_imported_txt(imported)

    def load_imported_txt(self, imported: ImportedTranscript):
        # Loading text never removes the independently selected audio file.
        self.transcript_source_path = imported.source_path
        self.transcript_source = "txt_structured" if imported.is_structured else "txt_plain"
        self.transcript_file_edit.setText(imported.source_path)
        self.segments = list(imported.segments)

        self.speaker_ids = []
        for seg in self.segments:
            if seg.speaker_id and seg.speaker_id not in self.speaker_ids:
                self.speaker_ids.append(seg.speaker_id)

        self.speaker_panel.set_speakers(self.speaker_ids, imported.speaker_names)
        self.populate_table()
        self.save_button.setEnabled(True)

        if imported.is_structured:
            self.status_label.setText(
                f"타임라인 TXT 불러오기 완료 · 발언 {len(self.segments)}개. "
                "음성 파일이 있으면 STT 없이 화자 라벨링할 수 있습니다."
            )
        else:
            self.status_label.setText(
                f"일반 TXT 불러오기 완료 · 텍스트 {len(self.segments)}줄. "
                "시간 정보가 없어 화자 라벨링은 사용할 수 없습니다."
            )

        self.update_mode_label()
        self.update_action_controls()

    def update_mode_label(self):
        audio_text = "음성 있음" if self.audio_path else "음성 없음"
        source_names = {
            "none": "텍스트 없음",
            "stt": "STT 텍스트",
            "txt_structured": "타임라인 TXT",
            "txt_plain": "일반 TXT(시간 없음)",
        }
        self.mode_label.setText(f"{audio_text} · {source_names.get(self.transcript_source, '')}")

    def refresh_processing_device(self):
        self.processing_device = resolve_processing_device(self.gpu_auto_check.isChecked())
        if self.processing_device.device == "cuda":
            self.device_status.setText(
                f"처리 장치: {self.processing_device.display_name} · GPU 사용"
            )
        elif self.processing_device.gpu_available:
            self.device_status.setText("처리 장치: CPU · GPU 자동 사용 꺼짐")
        else:
            self.device_status.setText("처리 장치: CPU · NVIDIA GPU 감지 안 됨")

    def on_processing_device_changed(self):
        previous_device = (
            self.processing_device.device,
            self.processing_device.compute_type,
        )
        self.refresh_processing_device()
        current_device = (
            self.processing_device.device,
            self.processing_device.compute_type,
        )
        if previous_device != current_device:
            self.stt_engine = None
            self.diarizer = None
            self.stt_model_status.setText("STT 모델: 준비 안 됨")
            self.diar_model_status.setText("화자 모델: 준비 안 됨")

    def selected_model_size(self) -> str:
        return STT_QUALITY_OPTIONS[self.quality_combo.currentText()]

    def selected_language(self) -> str | None:
        return LANGUAGE_OPTIONS[self.language_combo.currentText()]

    def on_stt_quality_changed(self):
        selected = self.selected_model_size()
        if self.stt_engine is not None and self.stt_engine.model_size != selected:
            self.stt_engine = None
            self.stt_model_status.setText("STT 모델: 준비 안 됨")

    def _model_loader_busy(self) -> bool:
        return self.model_loader is not None and self.model_loader.isRunning()

    def _worker_busy(self) -> bool:
        return self.worker is not None

    def _segments_have_timeline(self) -> bool:
        return bool(self.segments) and all(seg.start is not None for seg in self.segments)

    def update_action_controls(self):
        busy = self._worker_busy()
        loading = self._model_loader_busy()
        available = not busy and not loading

        self.audio_button.setEnabled(available)
        self.txt_button.setEnabled(available)
        self.stt_button.setEnabled(available and bool(self.audio_path))
        self.language_combo.setEnabled(available)
        self.quality_combo.setEnabled(available)
        self.stt_load_button.setEnabled(available)
        self.gpu_auto_check.setEnabled(available)
        self.token_edit.setEnabled(available)
        self.diar_load_button.setEnabled(available)
        self.diarize_button.setEnabled(
            available
            and bool(self.audio_path)
            and bool(self.segments)
            and self._segments_have_timeline()
        )
        self.cancel_button.setEnabled(busy)
        self.save_button.setEnabled(available and bool(self.segments))

    def preload_stt_model(self):
        if self._model_loader_busy() or self._worker_busy():
            QMessageBox.information(self, "모델 로딩", "다른 작업이 진행 중입니다.")
            return

        model_size = self.selected_model_size()
        if (
            self.stt_engine is not None
            and self.stt_engine.model_size == model_size
            and self.stt_engine.is_loaded
        ):
            actual_device = "GPU" if self.stt_engine.device == "cuda" else "CPU"
            self.stt_model_status.setText(f"STT 모델: 준비됨 ({model_size} · {actual_device})")
            return

        self.refresh_processing_device()
        engine = LocalSTTEngine(
            model_size=model_size,
            device=self.processing_device.device,
            compute_type=self.processing_device.compute_type,
            language=self.selected_language(),
        )
        self.stt_model_status.setText(f"STT 모델: 불러오는 중 ({model_size})...")
        self.model_loader = ModelLoadWorker("stt", engine, self)
        self.model_loader.status.connect(self.status_label.setText)
        self.model_loader.completed.connect(self.model_load_completed)
        self.model_loader.failed.connect(self.model_load_failed)
        self.model_loader.finished.connect(self.model_load_finished)
        self.model_loader.start()
        self.update_action_controls()

    def preload_diarization_model(self):
        if self._model_loader_busy() or self._worker_busy():
            QMessageBox.information(self, "모델 로딩", "다른 작업이 진행 중입니다.")
            return
        if self.diarizer is not None and self.diarizer.is_loaded:
            actual_device = "GPU" if self.diarizer.device == "cuda" else "CPU"
            self.diar_model_status.setText(f"화자 모델: 준비됨 ({actual_device})")
            return

        self.refresh_processing_device()
        engine = LocalSpeakerDiarizer(
            hf_token=self.token_edit.text().strip() or os.getenv("HF_TOKEN"),
            local_model_path=os.getenv("PYANNOTE_MODEL_PATH"),
            device=self.processing_device.device,
        )
        self.diar_model_status.setText("화자 모델: 불러오는 중...")
        self.model_loader = ModelLoadWorker("diarization", engine, self)
        self.model_loader.status.connect(self.status_label.setText)
        self.model_loader.completed.connect(self.model_load_completed)
        self.model_loader.failed.connect(self.model_load_failed)
        self.model_loader.finished.connect(self.model_load_finished)
        self.model_loader.start()
        self.update_action_controls()

    def model_load_completed(self, kind: str, engine):
        if kind == "stt":
            self.stt_engine = engine
            actual_device = "GPU" if self.stt_engine.device == "cuda" else "CPU"
            self.stt_model_status.setText(
                f"STT 모델: 준비됨 ({self.stt_engine.model_size} · {actual_device})"
            )
            self.status_label.setText("STT 모델을 메모리에 불러왔습니다.")
        else:
            self.diarizer = engine
            actual_device = "GPU" if self.diarizer.device == "cuda" else "CPU"
            self.diar_model_status.setText(f"화자 모델: 준비됨 ({actual_device})")
            self.status_label.setText("화자분리 모델을 메모리에 불러왔습니다.")

    def model_load_failed(self, kind: str, message: str):
        if kind == "stt":
            self.stt_engine = None
            self.stt_model_status.setText("STT 모델: 로딩 실패")
        else:
            self.diarizer = None
            self.diar_model_status.setText("화자 모델: 로딩 실패")
        QMessageBox.critical(self, "모델 로딩 오류", message)

    def model_load_finished(self):
        self.model_loader = None
        self.refresh_processing_device()
        self.update_action_controls()

    def _sync_table_text_to_segments(self):
        for row, segment in enumerate(self.segments):
            text_item = self.table.item(row, 2)
            if text_item is not None:
                segment.text = text_item.text().strip()

    def start_stt(self):
        if not self.audio_path:
            QMessageBox.warning(self, "음성 파일", "먼저 음성 파일을 선택하세요.")
            return

        self.refresh_processing_device()
        self.worker_kind = "stt"
        self.worker = STTWorker(
            audio_path=self.audio_path,
            model_size=self.selected_model_size(),
            language=self.selected_language(),
            device=self.processing_device.device,
            compute_type=self.processing_device.compute_type,
            stt_engine=self.stt_engine,
            parent=self,
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("STT 준비 중")
        self.worker.status.connect(self.status_label.setText)
        self.worker.progress.connect(self.update_progress)
        self.worker.completed.connect(self.stt_completed)
        self.worker.cancelled.connect(self.analysis_cancelled)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.update_action_controls()

    def start_diarization(self):
        if not self.audio_path or self.audio_duration is None:
            QMessageBox.warning(self, "화자 라벨링", "먼저 화자 기준이 될 음성 파일을 선택하세요.")
            return
        if not self.segments:
            QMessageBox.warning(self, "화자 라벨링", "먼저 STT를 실행하거나 기존 TXT를 불러오세요.")
            return

        self._sync_table_text_to_segments()
        try:
            normalized = prepare_transcript_for_diarization(
                self.segments,
                self.audio_duration,
            )
        except Exception as exc:
            QMessageBox.warning(self, "타임라인 확인 필요", str(exc))
            return

        self.refresh_processing_device()
        self.worker_kind = "diarization"
        self.worker = DiarizationWorker(
            audio_path=self.audio_path,
            transcript=normalized,
            audio_duration=self.audio_duration,
            hf_token=self.token_edit.text(),
            device=self.processing_device.device,
            diarizer=self.diarizer,
            parent=self,
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("화자 라벨링 준비 중")
        self.worker.status.connect(self.status_label.setText)
        self.worker.progress.connect(self.update_progress)
        self.worker.completed.connect(self.diarization_completed)
        self.worker.cancelled.connect(self.analysis_cancelled)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.update_action_controls()

    def cancel_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("작업 중지 요청 중...")
            self.progress_bar.setFormat("중지 요청 중")

    def update_progress(self, stage: str, percent: int, detail: str):
        percent = max(0, min(100, percent))
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{stage} · %p%")
        if detail:
            self.status_label.setText(f"{stage} 진행 중 · {detail}")
        else:
            self.status_label.setText(f"{stage} 진행 중")

    def _sync_model_status_from_worker(self):
        if self.worker is None:
            return
        engine = getattr(self.worker, "stt_engine", None)
        if engine is not None and engine.is_loaded:
            self.stt_engine = engine
            actual_device = "GPU" if engine.device == "cuda" else "CPU"
            self.stt_model_status.setText(
                f"STT 모델: 준비됨 ({engine.model_size} · {actual_device})"
            )
        diarizer = getattr(self.worker, "diarizer", None)
        if diarizer is not None and diarizer.is_loaded:
            self.diarizer = diarizer
            actual_device = "GPU" if diarizer.device == "cuda" else "CPU"
            self.diar_model_status.setText(f"화자 모델: 준비됨 ({actual_device})")

    def stt_completed(self, segments):
        self._sync_model_status_from_worker()
        self.segments = list(segments)
        self.transcript_source = "stt"
        self.transcript_source_path = self.audio_path
        self.transcript_file_edit.setText("STT 결과 (현재 음성에서 생성)")
        self.speaker_ids = []
        self.speaker_panel.set_speakers([])
        self.populate_table()
        self.status_label.setText(
            f"STT 완료 · 발언 {len(self.segments)}개. 텍스트는 유지되며 화자 라벨링을 별도로 실행할 수 있습니다."
        )
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("STT 완료 · 100%")
        self.update_mode_label()

    def diarization_completed(self, segments):
        self._sync_model_status_from_worker()
        self.segments = list(segments)
        self.speaker_ids = []
        for seg in self.segments:
            if seg.speaker_id and seg.speaker_id not in self.speaker_ids:
                self.speaker_ids.append(seg.speaker_id)
        self.speaker_panel.set_speakers(self.speaker_ids)
        self.populate_table()
        self.status_label.setText(
            f"화자 라벨링 완료 · 발언 {len(self.segments)}개 · 감지 화자 {len(self.speaker_ids)}명"
        )
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("화자 라벨링 완료 · 100%")
        self.update_mode_label()

    def analysis_cancelled(self):
        self._sync_model_status_from_worker()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("작업 중지됨")
        self.status_label.setText("작업이 중지되었습니다. 기존 텍스트는 유지됩니다.")

    def analysis_failed(self, message: str):
        self._sync_model_status_from_worker()
        stage = "STT" if self.worker_kind == "stt" else "화자 라벨링"
        self.progress_bar.setFormat(f"{stage} 실패")
        self.status_label.setText(f"{stage} 실패 · 기존 텍스트는 유지됩니다.")
        QMessageBox.critical(self, f"{stage} 오류", message)

    def worker_finished(self):
        self.worker = None
        self.worker_kind = ""
        self.refresh_processing_device()
        self.update_action_controls()

    def display_speaker_id(self, speaker_id: str) -> str:
        if not speaker_id:
            return ""
        try:
            return f"Speaker {self.speaker_ids.index(speaker_id) + 1}"
        except ValueError:
            return speaker_id

    def populate_table(self):
        self.table.setRowCount(len(self.segments))
        mapping = self.speaker_panel.mapping()
        for row, seg in enumerate(self.segments):
            time_item = QTableWidgetItem(format_time(seg.start))
            time_item.setFlags(time_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            speaker = mapping.get(seg.speaker_id) or self.display_speaker_id(seg.speaker_id)
            speaker_item = QTableWidgetItem(speaker)
            speaker_item.setFlags(speaker_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            text_item = QTableWidgetItem(seg.text)
            self.table.setItem(row, 0, time_item)
            self.table.setItem(row, 1, speaker_item)
            self.table.setItem(row, 2, text_item)

    def refresh_table_labels(self):
        mapping = self.speaker_panel.mapping()
        for row, seg in enumerate(self.segments):
            name = mapping.get(seg.speaker_id) or self.display_speaker_id(seg.speaker_id)
            item = self.table.item(row, 1)
            if item:
                item.setText(name)

    def _edited_text(self, row: int, segment: TranscriptSegment) -> str:
        text_item = self.table.item(row, 2)
        return text_item.text().strip() if text_item else segment.text.strip()

    def save_txt(self):
        if not self.segments:
            return

        if self.transcript_source_path:
            source = Path(self.transcript_source_path)
            default_name = f"{source.stem}_edited.txt"
            initial_path = str(source.with_name(default_name))
        else:
            initial_path = "transcript.txt"

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Transcript 저장",
            initial_path,
            "Text Files (*.txt)",
        )
        if not path:
            return

        if self.transcript_source == "txt_plain":
            lines = [
                self._edited_text(row, seg)
                for row, seg in enumerate(self.segments)
                if self._edited_text(row, seg)
            ]
            output = "\n".join(lines)
        else:
            mapping = self.speaker_panel.mapping()
            lines: list[str] = []
            for row, seg in enumerate(self.segments):
                text = self._edited_text(row, seg)
                speaker = mapping.get(seg.speaker_id) or self.display_speaker_id(seg.speaker_id)
                timecode = format_time(seg.start)

                if timecode and speaker:
                    lines.append(f"[{timecode}] {speaker}")
                elif timecode:
                    lines.append(f"[{timecode}]")
                elif speaker:
                    lines.append(speaker)

                lines.append(text)
                lines.append("")

            output = "\n".join(lines).rstrip() + "\n"

        Path(path).write_text(output, encoding="utf-8")
        self.status_label.setText(f"TXT 저장 완료: {path}")


class FutureMinutesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(
            "2. 회의록 생성\n\n"
            "추후 AI Provider / API Key / 회의록 양식 / Preview 기능을 추가할 영역입니다."
        )
        label.setWordWrap(True)
        label.setStyleSheet("color: #777; font-size: 14px;")
        layout.addWidget(label)
        layout.addStretch(1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Meeting Transcriber v0.6")
        self.resize(1120, 740)

        tabs = QTabWidget()
        tabs.addTab(TranscriptionTab(), "1. Transcript 준비")
        tabs.addTab(FutureMinutesTab(), "2. 회의록 생성")
        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

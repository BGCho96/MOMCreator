from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
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

from speaker_diarizer import LocalSpeakerDiarizer, assign_speakers
from stt_engine import AnalysisCancelled, LocalSTTEngine, TranscriptSegment
from text_importer import ImportedTranscript, load_transcript_txt


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class AnalysisWorker(QThread):
    status = pyqtSignal(str)
    progress = pyqtSignal(str, int, str)
    completed = pyqtSignal(object)
    cancelled = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, audio_path: str, use_diarization: bool, hf_token: str, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path
        self.use_diarization = use_diarization
        self.hf_token = hf_token.strip()
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    def run(self):
        try:
            stt = LocalSTTEngine(
                model_size="small",
                device="cpu",
                compute_type="int8",
                language=None,
            )

            def stt_progress(percent: int, current: float, total: float):
                detail = f"{format_time(current)} / {format_time(total)}"
                self.progress.emit("STT", percent, detail)

            segments = stt.transcribe(
                self.audio_path,
                status_callback=self.status.emit,
                progress_callback=stt_progress,
                cancel_check=self._is_cancelled,
            )

            if self.use_diarization:
                if self._is_cancelled():
                    raise AnalysisCancelled()

                diarizer = LocalSpeakerDiarizer(
                    hf_token=self.hf_token or os.getenv("HF_TOKEN"),
                    local_model_path=os.getenv("PYANNOTE_MODEL_PATH"),
                )

                def diar_progress(step: str, percent: int):
                    self.progress.emit(f"화자분리 · {step}", percent, "")

                turns = diarizer.diarize(
                    self.audio_path,
                    status_callback=self.status.emit,
                    progress_callback=diar_progress,
                    cancel_check=self._is_cancelled,
                )
                segments = assign_speakers(segments, turns)

            if self._is_cancelled():
                raise AnalysisCancelled()

            self.completed.emit(segments)
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

        self.audio_path = ""
        self.source_path = ""
        self.source_mode = "none"  # none | audio | txt_structured | txt_plain
        self.segments: list[TranscriptSegment] = []
        self.speaker_ids: list[str] = []
        self.worker: AnalysisWorker | None = None

        root = QVBoxLayout(self)

        source_title = QLabel("입력 방식")
        source_title.setStyleSheet("font-weight: 600;")
        root.addWidget(source_title)

        source_buttons = QHBoxLayout()
        self.audio_button = QPushButton("음성 해석하기")
        self.txt_button = QPushButton("기존 TXT 불러오기")
        self.audio_button.clicked.connect(self.choose_audio)
        self.txt_button.clicked.connect(self.choose_txt)
        source_buttons.addWidget(self.audio_button)
        source_buttons.addWidget(self.txt_button)
        source_buttons.addStretch(1)
        root.addLayout(source_buttons)

        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("음성 파일 또는 기존 TXT를 선택하세요.")
        root.addWidget(self.file_edit)

        option_row = QHBoxLayout()
        self.diarization_check = QCheckBox("화자 분리 사용")
        self.diarization_check.setChecked(True)
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("HF Token - 최초 pyannote 모델 다운로드 시 필요할 수 있음")
        self.analyze_button = QPushButton("분석 시작")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setEnabled(False)

        self.cancel_button = QPushButton("분석 중지")
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.cancel_button.setEnabled(False)

        option_row.addWidget(self.diarization_check)
        option_row.addWidget(QLabel("HF Token"))
        option_row.addWidget(self.token_edit, 1)
        option_row.addWidget(self.analyze_button)
        option_row.addWidget(self.cancel_button)
        root.addLayout(option_row)

        privacy = QLabel(
            "음성/STT 분석은 로컬에서 수행합니다. 모델이 PC에 없으면 최초 모델 다운로드에는 인터넷이 사용될 수 있습니다. "
            "TXT 불러오기는 파일 내용을 외부로 전송하지 않습니다."
        )
        privacy.setWordWrap(True)
        privacy.setStyleSheet("color: #666;")
        root.addWidget(privacy)

        self.status_label = QLabel("입력 파일을 선택하세요.")
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

        self.audio_path = path
        self.source_path = path
        self.source_mode = "audio"
        self.file_edit.setText(path)
        self.status_label.setText("음성 분석 준비 완료.")
        self.mode_label.setText("입력: 음성 파일")
        self.analyze_button.setEnabled(True)
        self._set_audio_options_enabled(True)

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
        self.audio_path = ""
        self.source_path = imported.source_path
        self.source_mode = "txt_structured" if imported.is_structured else "txt_plain"
        self.file_edit.setText(imported.source_path)
        self.segments = list(imported.segments)

        self.speaker_ids = []
        for seg in self.segments:
            if seg.speaker_id and seg.speaker_id not in self.speaker_ids:
                self.speaker_ids.append(seg.speaker_id)

        self.speaker_panel.set_speakers(self.speaker_ids, imported.speaker_names)
        self.populate_table()

        self.analyze_button.setEnabled(False)
        self._set_audio_options_enabled(False)
        self.save_button.setEnabled(True)

        if imported.is_structured:
            self.status_label.setText(
                f"구조화 TXT 불러오기 완료 · 발언 {len(self.segments)}개 · 화자 {len(self.speaker_ids)}명"
            )
            self.mode_label.setText("입력: 구조화 TXT · 시간/화자 정보 복원")
        else:
            self.status_label.setText(f"일반 TXT 불러오기 완료 · 텍스트 {len(self.segments)}줄")
            self.mode_label.setText("입력: 일반 TXT · 시간/화자 정보 없음")

    def _set_audio_options_enabled(self, enabled: bool):
        self.diarization_check.setEnabled(enabled)
        self.token_edit.setEnabled(enabled)

    def start_analysis(self):
        if self.source_mode != "audio" or not self.audio_path:
            QMessageBox.warning(self, "음성 파일", "먼저 '음성 해석하기'에서 음성 파일을 선택하세요.")
            return

        self.analyze_button.setEnabled(False)
        self.audio_button.setEnabled(False)
        self.txt_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.table.setRowCount(0)
        self.segments = []
        self.speaker_ids = []
        self.speaker_panel.set_speakers([])

        self.worker = AnalysisWorker(
            self.audio_path,
            self.diarization_check.isChecked(),
            self.token_edit.text(),
            self,
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("분석 준비 중")
        self.cancel_button.setEnabled(True)

        self.worker.status.connect(self.status_label.setText)
        self.worker.progress.connect(self.update_progress)
        self.worker.completed.connect(self.analysis_completed)
        self.worker.cancelled.connect(self.analysis_cancelled)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.start()

    def cancel_analysis(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("분석 중지 요청 중...")
            self.progress_bar.setFormat("중지 요청 중")

    def update_progress(self, stage: str, percent: int, detail: str):
        percent = max(0, min(100, percent))
        self.progress_bar.setValue(percent)
        self.progress_bar.setFormat(f"{stage} · %p%")
        if detail:
            self.status_label.setText(f"{stage} 진행 중 · {detail}")
        else:
            self.status_label.setText(f"{stage} 진행 중")

    def _restore_analysis_controls(self):
        self.analyze_button.setEnabled(True)
        self.audio_button.setEnabled(True)
        self.txt_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def analysis_completed(self, segments):
        self.segments = list(segments)
        self.speaker_ids = []
        for seg in self.segments:
            if seg.speaker_id and seg.speaker_id not in self.speaker_ids:
                self.speaker_ids.append(seg.speaker_id)

        self.speaker_panel.set_speakers(self.speaker_ids)
        self.populate_table()
        self.status_label.setText(
            f"분석 완료 · 발언 {len(self.segments)}개 · 감지 화자 {len(self.speaker_ids)}명"
        )
        self.mode_label.setText("입력: 음성 분석 결과")
        self._restore_analysis_controls()
        self.save_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("분석 완료 · 100%")

    def analysis_cancelled(self):
        self._restore_analysis_controls()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("분석 중지됨")
        self.status_label.setText("분석이 중지되었습니다.")

    def analysis_failed(self, message: str):
        self._restore_analysis_controls()
        self.progress_bar.setFormat("분석 실패")
        self.status_label.setText("분석 실패")
        QMessageBox.critical(self, "분석 오류", message)

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

            # The text column stays editable for manual correction in every input mode.
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

        if self.source_path:
            source = Path(self.source_path)
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

        if self.source_mode == "txt_plain":
            # No metadata existed in the source, so do not invent any on save.
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
        self.setWindowTitle("Meeting Transcriber v0.2")
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


class AnalysisCancelled(Exception):
    """Raised when the user requests cancellation of a running analysis."""


@dataclass
class TranscriptSegment:
    start: float | None
    end: float | None
    text: str
    speaker_id: str = ""


class LocalSTTEngine:
    """Local STT backend isolated from the PyQt UI."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = "ko",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(
        self,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        """Load the selected Whisper model into memory ahead of analysis."""
        if self._model is not None:
            return self._model

        if status_callback:
            status_callback(f"STT 모델 불러오는 중 ({self.model_size})...")

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper가 설치되어 있지 않습니다.\n"
                "pip install faster-whisper"
            ) from exc

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._model

    def _load_model(self):
        return self.load_model()

    def transcribe(
        self,
        audio_path: str | Path,
        status_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, float, float], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> list[TranscriptSegment]:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_path}")

        model = self.load_model(status_callback=status_callback)

        if cancel_check and cancel_check():
            raise AnalysisCancelled()

        if status_callback:
            status_callback("음성을 텍스트로 변환 중...")

        segments, info = model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=True,
            beam_size=5,
        )

        total_duration = float(getattr(info, "duration", 0.0) or 0.0)
        result: list[TranscriptSegment] = []

        # faster-whisper performs most transcription work while this generator
        # is iterated. This gives us a natural place for cooperative cancellation
        # and timeline-based progress reporting.
        for seg in segments:
            if cancel_check and cancel_check():
                raise AnalysisCancelled()

            current = float(seg.end)
            if progress_callback and total_duration > 0:
                percent = int(max(0, min(100, current / total_duration * 100)))
                progress_callback(percent, current, total_duration)

            text = (seg.text or "").strip()
            if not text:
                continue

            result.append(
                TranscriptSegment(
                    start=float(seg.start),
                    end=current,
                    text=text,
                )
            )

        if cancel_check and cancel_check():
            raise AnalysisCancelled()

        if progress_callback and total_duration > 0:
            progress_callback(100, total_duration, total_duration)

        if not result:
            raise RuntimeError("STT 결과가 비어 있습니다.")

        return result

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass
class TranscriptSegment:
    start: float
    end: float
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

    def _load_model(self):
        if self._model is not None:
            return self._model

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

    def transcribe(
        self,
        audio_path: str | Path,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> list[TranscriptSegment]:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_path}")

        if status_callback:
            status_callback(f"STT 모델 로딩 중 ({self.model_size})...")

        model = self._load_model()

        if status_callback:
            status_callback("음성을 텍스트로 변환 중...")

        segments, _info = model.transcribe(
            str(audio_path),
            language=self.language,
            vad_filter=True,
            beam_size=5,
        )

        result: list[TranscriptSegment] = []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            result.append(
                TranscriptSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                )
            )

        if not result:
            raise RuntimeError("STT 결과가 비어 있습니다.")

        return result

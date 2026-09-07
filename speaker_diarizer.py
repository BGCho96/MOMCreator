from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from stt_engine import AnalysisCancelled, TranscriptSegment


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker_id: str


class LocalSpeakerDiarizer:
    """Local speaker-diarization backend isolated from the UI."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        local_model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.hf_token = hf_token or None
        self.local_model_path = local_model_path or None
        self.device = device
        self._pipeline = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load_model(
        self,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        """Load the diarization pipeline into memory ahead of analysis."""
        if self._pipeline is not None:
            return self._pipeline

        if status_callback:
            status_callback("화자분리 모델 불러오는 중...")

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio가 설치되어 있지 않습니다.\n"
                "pip install pyannote.audio"
            ) from exc

        source = self.local_model_path or "pyannote/speaker-diarization-community-1"
        kwargs = {}
        if not self.local_model_path and self.hf_token:
            kwargs["token"] = self.hf_token

        try:
            self._pipeline = Pipeline.from_pretrained(source, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "화자분리 모델 로딩 실패: "
                f"{exc}\n처음 모델을 받는 경우 Hugging Face 모델 사용 조건 동의와 토큰이 필요할 수 있습니다."
            ) from exc

        if self.device == "cuda":
            try:
                import torch
                self._pipeline.to(torch.device("cuda"))
            except Exception:
                if status_callback:
                    status_callback(
                        "화자 모델 GPU 전환에 실패하여 CPU 모드로 사용합니다..."
                    )
                self.device = "cpu"

        return self._pipeline

    def _load_pipeline(self):
        return self.load_model()

    def diarize(
        self,
        audio_path: str | Path,
        status_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> list[SpeakerTurn]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {audio_path}")

        pipeline = self.load_model(status_callback=status_callback)

        if cancel_check and cancel_check():
            raise AnalysisCancelled()

        if status_callback:
            status_callback("화자 구간 분석 중...")

        def hook(step_name, _artifact=None, **kwargs):
            if cancel_check and cancel_check():
                raise AnalysisCancelled()

            if progress_callback:
                completed = kwargs.get("completed")
                total = kwargs.get("total")
                if completed is not None and total:
                    percent = int(max(0, min(100, completed / total * 100)))
                else:
                    percent = 0
                progress_callback(str(step_name), percent)

        output = pipeline(str(audio_path), hook=hook)

        if cancel_check and cancel_check():
            raise AnalysisCancelled()

        if progress_callback:
            progress_callback("화자분리 완료", 100)

        annotation = getattr(output, "exclusive_speaker_diarization", None)
        if annotation is None:
            annotation = getattr(output, "speaker_diarization", None)
        if annotation is None:
            annotation = output

        turns: list[SpeakerTurn] = []
        if hasattr(annotation, "itertracks"):
            for turn, _track, speaker in annotation.itertracks(yield_label=True):
                turns.append(SpeakerTurn(float(turn.start), float(turn.end), str(speaker)))
        else:
            for item in annotation:
                if len(item) == 2:
                    turn, speaker = item
                else:
                    turn, _track, speaker = item
                turns.append(SpeakerTurn(float(turn.start), float(turn.end), str(speaker)))

        if not turns:
            raise RuntimeError("화자분리 결과가 비어 있습니다.")
        return turns


def assign_speakers(
    transcript: list[TranscriptSegment],
    turns: list[SpeakerTurn],
) -> list[TranscriptSegment]:
    """Assign the speaker with the largest time overlap to each STT segment."""
    if not transcript or not turns:
        return transcript

    for segment in transcript:
        if segment.start is None or segment.end is None:
            continue

        best_speaker = ""
        best_overlap = 0.0

        for turn in turns:
            overlap = max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn.speaker_id

        if not best_speaker:
            segment_mid = (segment.start + segment.end) / 2
            nearest = min(
                turns,
                key=lambda turn: abs(((turn.start + turn.end) / 2) - segment_mid),
            )
            best_speaker = nearest.speaker_id

        segment.speaker_id = best_speaker

    return transcript

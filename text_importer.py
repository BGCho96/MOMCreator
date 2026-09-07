from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from stt_engine import TranscriptSegment


STRUCTURED_HEADER = re.compile(
    r"^\s*\[(?P<time>(?:\d{1,2}:)?\d{1,2}:\d{2})\]\s*(?P<speaker>.*?)\s*$"
)


@dataclass
class ImportedTranscript:
    segments: list[TranscriptSegment]
    is_structured: bool
    speaker_names: dict[str, str]
    source_path: str


def parse_timecode(value: str) -> float:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours * 3600 + minutes * 60 + seconds)
    raise ValueError(f"지원하지 않는 시간 형식입니다: {value}")


def _read_text_with_fallback(path: Path) -> str:
    # UTF-8 is the app's save format. cp949 fallback helps with older Korean TXT files.
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError("TXT 파일 인코딩을 해석할 수 없습니다. UTF-8 또는 CP949 파일을 사용하세요.")


def load_transcript_txt(file_path: str | Path) -> ImportedTranscript:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TXT 파일을 찾을 수 없습니다: {path}")

    raw = _read_text_with_fallback(path)
    lines = raw.splitlines()

    header_indexes = [index for index, line in enumerate(lines) if STRUCTURED_HEADER.match(line)]

    if header_indexes:
        return _parse_structured(lines, str(path))

    return _parse_plain(lines, str(path))


def _parse_structured(lines: list[str], source_path: str) -> ImportedTranscript:
    segments: list[TranscriptSegment] = []
    speaker_to_id: dict[str, str] = {}
    speaker_names: dict[str, str] = {}

    current_time: Optional[float] = None
    current_speaker = ""
    current_text: list[str] = []

    def flush_current():
        nonlocal current_time, current_speaker, current_text
        if current_time is None:
            current_text = []
            return

        text = "\n".join(current_text).strip()
        if not text:
            current_time = None
            current_speaker = ""
            current_text = []
            return

        speaker_label = current_speaker.strip()
        if speaker_label:
            if speaker_label not in speaker_to_id:
                speaker_id = f"TXT_SPEAKER_{len(speaker_to_id) + 1:03d}"
                speaker_to_id[speaker_label] = speaker_id
                speaker_names[speaker_id] = speaker_label
            speaker_id = speaker_to_id[speaker_label]
        else:
            speaker_id = ""

        segments.append(
            TranscriptSegment(
                start=current_time,
                end=None,
                text=text,
                speaker_id=speaker_id,
            )
        )
        current_time = None
        current_speaker = ""
        current_text = []

    for line in lines:
        match = STRUCTURED_HEADER.match(line)
        if match:
            flush_current()
            current_time = parse_timecode(match.group("time"))
            current_speaker = match.group("speaker").strip()
        else:
            if current_time is not None:
                current_text.append(line)

    flush_current()

    if not segments:
        # A malformed file may contain header-like text but no usable content.
        return _parse_plain(lines, source_path)

    # Recover a best-effort end time from the next segment start.
    for index, segment in enumerate(segments[:-1]):
        segment.end = segments[index + 1].start

    return ImportedTranscript(
        segments=segments,
        is_structured=True,
        speaker_names=speaker_names,
        source_path=source_path,
    )


def _parse_plain(lines: list[str], source_path: str) -> ImportedTranscript:
    # Keep each non-empty line as one editable row. This avoids inventing time/speaker metadata.
    segments = [
        TranscriptSegment(start=None, end=None, text=line.strip(), speaker_id="")
        for line in lines
        if line.strip()
    ]

    if not segments:
        raise RuntimeError("TXT 파일에 표시할 텍스트가 없습니다.")

    return ImportedTranscript(
        segments=segments,
        is_structured=False,
        speaker_names={},
        source_path=source_path,
    )

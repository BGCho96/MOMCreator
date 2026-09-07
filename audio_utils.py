from __future__ import annotations

from pathlib import Path


def get_audio_duration(audio_path: str | Path) -> float:
    """Return audio duration in seconds without relying on system FFmpeg PATH.

    WAV uses the Python standard library first. Other formats use PyAV, which is
    already part of the faster-whisper stack in a normal installation.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"음성 파일을 찾을 수 없습니다: {path}")

    if path.suffix.lower() == ".wav":
        try:
            import wave

            with wave.open(str(path), "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                if frames > 0 and rate > 0:
                    return frames / float(rate)
        except Exception:
            pass

    try:
        import av
    except ImportError as exc:
        raise RuntimeError(
            "음성 길이를 확인하려면 PyAV가 필요합니다. "
            "faster-whisper가 정상 설치된 환경인지 확인하세요."
        ) from exc

    try:
        with av.open(str(path)) as container:
            if container.duration is not None:
                duration = float(container.duration) / float(av.time_base)
                if duration > 0:
                    return duration

            durations: list[float] = []
            for stream in container.streams.audio:
                if stream.duration is not None and stream.time_base is not None:
                    durations.append(float(stream.duration * stream.time_base))
            if durations:
                duration = max(durations)
                if duration > 0:
                    return duration
    except Exception as exc:
        raise RuntimeError(f"음성 파일 길이를 읽지 못했습니다: {exc}") from exc

    raise RuntimeError("음성 파일 길이를 확인할 수 없습니다.")

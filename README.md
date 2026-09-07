# Meeting Transcriber v0.1

로컬 회의 음성 전사 + 화자분리 초안입니다.

## 현재 범위
- PyQt6 2탭 UI
- 음성 파일 선택
- faster-whisper 기반 로컬 STT
- pyannote 기반 로컬 화자분리
- 감지된 화자에 사용자가 직접 이름 지정
- 기본 이름 지정 슬롯 4개
- 감지 화자가 더 많으면 `+ 지정 슬롯`으로 추가
- Transcript 발언 텍스트 직접 수정
- TXT 저장
- 2번째 `회의록 생성` 탭은 placeholder

## 코드 구조
- `main.py`: PyQt UI, 화자 이름 매핑, Transcript 표시/저장
- `stt_engine.py`: STT 전용. 향후 잡음 제거, 포맷별 전처리, 모델 변경은 여기를 수정
- `speaker_diarizer.py`: 화자분리 전용. diarization과 STT 타임스탬프 결합

## 설치
```bash
pip install -r requirements.txt
```

회사 PC에서 pyannote/PyTorch 설치가 먼저 부담되면:
```bash
pip install PyQt6 faster-whisper
```
까지만 설치하고 UI에서 `화자 분리 사용`을 끈 채 STT부터 확인할 수 있습니다.

## 실행
```bash
python main.py
```

## 기본 STT 설정
- model: `small`
- device: `cpu`
- compute type: `int8`
- language: `ko`

초기 호환성을 위해 CPU 기준으로 잡았습니다.

## 모델 다운로드와 보안
음성 추론은 로컬에서 수행하도록 설계했습니다. 다만 모델이 PC에 없으면 최초 모델 다운로드에는 인터넷 연결이 사용될 수 있습니다.

pyannote `speaker-diarization-community-1`은 최초 접근 시 Hugging Face 사용 조건 동의 및 token이 필요할 수 있습니다. 완전 오프라인 사용 시 로컬 모델 폴더를 내려받고 환경변수 `PYANNOTE_MODEL_PATH`로 지정할 수 있습니다.

HF token은 UI 입력 또는 환경변수 `HF_TOKEN`으로 전달하며, 현재 초안에서는 저장하지 않습니다.

## 초안에서 의도적으로 제외
- 특정 사람 자동 실명 추론/목소리 추적
- 발언별 수동 화자 재지정
- 음성 플레이어/타임스탬프 클릭 재생
- 노이즈 제거/기기별 튜닝
- AI 회의록 생성
- EXE 최적화

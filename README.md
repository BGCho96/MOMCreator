# Meeting Transcriber v0.2

로컬 회의 음성 전사 + 화자분리 + 기존 TXT 재편집 초안입니다.

## v0.2 변경점

1번 탭의 입력 방식이 2개로 분리되었습니다.

### 1) 음성 해석하기
- 기존 기능 유지
- faster-whisper 로컬 STT
- pyannote 로컬 화자분리
- 감지된 화자에 사용자가 직접 이름 지정
- Transcript 발언 직접 수정
- TXT 저장

### 2) 기존 TXT 불러오기
TXT 내용을 보고 자동으로 두 방식 중 하나로 처리합니다.

#### 구조화 TXT
이 프로그램이 저장하는 아래 형식을 인식합니다.

```text
[00:03] 김대리
이번 설치 일정에 대해서 말씀드리겠습니다.

[00:11] 박과장
그러면 10월 둘째 주로 잡으면 될까요?
```

- 시간 복원
- 화자 복원
- 발언 복원
- 복원된 화자 이름을 다시 수정 가능
- 수정 후 같은 구조를 유지해 TXT 저장 가능

#### 일반 TXT
시간/화자 헤더가 없는 일반 메모라면:

```text
설치 일정은 10월 둘째 주로 논의
현장 일정 재확인 필요
금요일까지 자료 전달
```

- 각 비어 있지 않은 줄을 편집 가능한 발언 행으로 표시
- 시간/화자 정보는 억지로 생성하지 않음
- 저장 시에도 일반 TXT 형식을 유지

## 코드 구조

- `main.py`
  - PyQt6 UI
  - 음성/TXT 입력 선택
  - 화자 이름 매핑
  - Transcript 표시/수정/저장
- `stt_engine.py`
  - STT 전용
  - 향후 잡음 제거, 포맷별 전처리, 모델 변경을 이곳에서 확장
- `speaker_diarizer.py`
  - 화자분리 전용
  - diarization과 STT 타임스탬프 결합
- `text_importer.py`
  - TXT 형식 판별
  - 구조화 TXT의 시간/화자/발언 복원
  - 일반 TXT fallback

## 설치

전체 기능:

```bash
pip install -r requirements_full.txt
```

STT까지만 먼저 확인:

```bash
pip install -r requirements_stt_only.txt
```

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
음성 추론은 로컬에서 수행하도록 설계했습니다. 단 모델이 PC에 없으면 최초 모델 다운로드에는 인터넷 연결이 사용될 수 있습니다.

TXT 불러오기/수정은 로컬 파일 처리만 수행합니다.

pyannote `speaker-diarization-community-1`은 최초 접근 시 Hugging Face 사용 조건 동의 및 token이 필요할 수 있습니다. HF token은 UI 입력 또는 환경변수 `HF_TOKEN`으로 전달하며 현재 초안에서는 저장하지 않습니다.

## 아직 제외한 기능
- 특정 사람 자동 실명 추론/목소리 추적
- 발언별 수동 화자 재지정
- 음성 플레이어/타임스탬프 클릭 재생
- 노이즈 제거/기기별 튜닝
- AI 회의록 생성
- EXE 최적화

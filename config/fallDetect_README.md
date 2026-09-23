# 낙상 감지 AI 모듈 — fall_detector.py (v2: 판단근거 제공자)

`AiServer/fall_detector.py`는 이미 추출된 keypoints를 받아서 낙상 확률(`fall_prob`)만
계산해 돌려주는 모듈이다. state(normal/checking/alert/lost) 판정, risk/alert 전이,
EVENT 생성은 이제 이 모듈이 하지 않는다 — 전부 **Main Service의 정책 엔진**이 담당한다.

## 왜 이렇게 바뀌었나 (v1 → v2)

기존 구조는 `frame → YOLO Pose → LSTM 추론 → threshold 적용 → state 결정 → STATE/EVENT 생성`을
AI 모듈 하나가 전부 처리했다. 이 방식의 문제:

- threshold, window_sec, min_ratio 같은 **정책 값이 AI 코드에 박혀있어서**, 정책을 바꾸려면
  AI 모듈을 재배포해야 했음
- gait/stretch 같은 다른 AI 모듈과 Main 쪽에서 통합 판단하는 게 불가능한 구조였음

그래서 AI 모듈은 "판단근거(fall_prob)만 제공"하고, "그 값으로 실제 state를 어떻게 정의할지"는
Main Service가 갖는 정책 엔진으로 분리했다.

| | v1 (이전) | v2 (현재) |
|---|---|---|
| 입력 | 원본 프레임(frame) | 이미 추출된 keypoints |
| 포즈 추정 | AI 모듈이 직접 YOLO 실행 | 이 모듈 밖에서 이미 끝난 상태로 들어옴 |
| threshold/window_sec/min_ratio | AI 모듈이 소유 | Main Service가 소유 |
| state(normal/checking/alert/lost) 판정 | AI 모듈(`RoomIncidents`)이 함 | Main Service가 함 |
| EVENT(ALERT_OPENED 등) 생성 | AI 모듈이 함 | Main Service가 함 |
| AI 모듈 출력 | `StateMessage` + `EventMessage` | `AIOutput`(판단근거 dict) 하나 |

## 역할 경계

- **이 모듈이 하는 일**: keypoints(이미 추출됨) → 정규화 → LSTM 추론 → `fall_prob`(판단근거) 계산 → dict 반환
- **이 모듈이 안 하는 일**: 프레임에서 keypoints 뽑기(YOLO pose), threshold 적용, state 판정,
  risk/alert 상태 전이, EVENT 생성, Main Service와의 통신 프로토콜(HTTP 등)

## 사용법

```python
from fall_detector import load_models, FallDetectorSession, to_json

# load_yolo=False가 기본값 — keypoints를 직접 받으므로 YOLO는 로드 안 함
yolo, lstm_model, device = load_models()

session = FallDetectorSession(camera_id="CAM-03", yolo=yolo, lstm_model=lstm_model,
                               device=device, fps=25.0)

# keypoints_list: [(track_id, keypoints), ...], keypoints는 17 x 2 ([[x, y], ...])
keypoints_data = [
    (7, keypoints_of_person_7),
    (8, keypoints_of_person_8),
]

output = session.process_keypoints(frame_idx=1234, keypoints_list=keypoints_data)

json_str = to_json(output)
requests.post(f"{MAIN_URL}/api/fall-detection/raw", data=json_str)
```

## 함수/클래스별 요약

| 이름 | 역할 |
|---|---|
| `RawDetection` (dataclass) | 사람 한 명의 판단근거. `track_id`, `bbox`, `raw_data`(`fall_prob`, `confidence`) |
| `AIOutput` (dataclass) | 이 모듈의 최종 출력. `camera_id`, `timestamp`, `mode`, `detections`(`list[RawDetection]`) |
| `to_json(message)` | dataclass를 JSON 문자열로 직렬화 |
| `_now_iso()` | 현재 시각을 ISO8601(UTC) 문자열로 반환 |
| `FallLSTM` | 낙상 판별 LSTM 모델 구조 정의 (`fall_lstm.pt`가 이 구조의 학습된 가중치) |
| `load_models()` | `fall_lstm.pt` 로드 (기본). `load_yolo=True`로 주면 `yolov8n-pose.pt`도 로드하지만 지금 추론 경로에선 안 씀 |
| `normalize(kpts)` | keypoint 17개를 hip 중심 상대좌표로 정규화 (위치/크기 무관하게) |
| `Track` | track_id 하나당 LSTM 입력 시퀀스(최근 16프레임)를 들고 있는 버퍼. state 판정 필드 없음 |
| `FallDetectorSession` | 카메라 한 대에 대응하는 세션. `tracks` 딕셔너리(track_id → `Track`)를 들고 있음 |
| `FallDetectorSession.process_keypoints()` | 매 프레임 호출하는 메인 함수. keypoints → `fall_prob` 계산 → `AIOutput` 반환 |

## 입력 형식

`process_keypoints(frame_idx, keypoints_list)`:

| 인자 | 타입 | 설명 |
|---|---|---|
| `frame_idx` | int | 프레임 번호 (참고/로깅용) |
| `keypoints_list` | `[(track_id, keypoints), ...]` | `track_id`는 int, `keypoints`는 17 x 2 배열/리스트. `keypoints`가 `None`이거나 17개 미만이면 해당 track은 이번 프레임에서 건너뜀 |

`track_id`는 이 모듈이 만드는 게 아니라 **호출하는 쪽(포즈 추정 + 트래킹을 이미 끝낸 상위 시스템)이
매 프레임 넘겨준다.**

## 출력 형식

| 필드 | 타입 | 설명 |
|---|---|---|
| camera_id | string | 카메라 식별자 |
| timestamp | string (ISO8601, UTC) | 처리 시각 |
| mode | int | `0` = fall_detector 고정값. 다른 AI 모듈(gait=1, stretch=2 등)과 구분용 |
| detections | array | 이 프레임에서 판단한 사람 목록 |
| detections[].track_id | integer | 입력으로 받은 track_id 그대로 |
| detections[].bbox | [x1,y1,x2,y2] | 현재는 항상 `[0,0,0,0]` — 아래 "확정 필요 항목" 참고 |
| detections[].raw_data.fall_prob | float (0~1) | LSTM이 계산한 낙상 확률 (판단근거) |
| detections[].raw_data.confidence | float | 현재는 placeholder(`1.0`) — 아래 참고 |

예시:
```json
{
  "camera_id": "CAM-03",
  "timestamp": "2026-09-22T10:15:32.512Z",
  "mode": 0,
  "detections": [
    { "track_id": 7, "bbox": [0.0, 0.0, 0.0, 0.0], "raw_data": {"fall_prob": 0.81, "confidence": 1.0} }
  ]
}
```

state/event는 이 출력에 없다. Main Service가 `fall_prob` 시계열을 카메라·track별로
버퍼링해서 threshold/window_sec/min_ratio를 적용해 state를 직접 결정해야 한다.

## 확정 필요 항목 (TODO)

- **`bbox`가 항상 `[0,0,0,0]`**: 입력 `(track_id, keypoints)`엔 bbox 정보가 없어서 채울 방법이
  없음. keypoints의 min/max로 계산해서 채우는 방법이 있는데, 스펙에 명시되지 않아 일단 보류.
- **`confidence`가 placeholder(`1.0`)**: 입력 keypoints에 point별 confidence가 없어서 실제
  계산 방법이 없음. "confidence를 어떻게 정의할지" Main Service/팀 차원에서 확정 필요.
- **Main Service 쪽 정책 엔진 미구현**: threshold, window_sec, min_ratio 값과 state 판정 로직
  (이전 `RoomIncidents`가 하던 일)을 Main Service가 새로 구현해야 함. 상태 전이표(어떤 전이에서
  어떤 이벤트가 나야 하는지)는 참고용으로 아래에 남겨둠 — Main 구현 시 동일한 판단 기준을
  쓰고 싶다면 참고할 것.

### 참고용 상태 전이표 (v1에서 AI가 쓰던 판단 기준, 이제는 Main이 구현해야 함)

| 이전 상태 | 전이 | 판단 / 동작 |
|---|---|---|
| normal | → lost | 무시 (일상적인 화면 이탈로 간주) |
| checking | → lost | risk(위험) 사건 오픈 |
| alert | → lost | alert 유지, 추가 알림 없음 |
| lost | → normal (재탐지, 낙상 아님) | risk 자동 종료 |
| lost | → checking (재탐지, 낙상처럼 보임) | risk 유지 |
| lost | → alert (재탐지, 확정 조건 충족) | risk를 alert로 흡수 후 제거 |

## 제거된 것들 (v1 대비)

- `StateMessage`, `EventMessage`, `AckMessage` (Main이 만듦)
- `RoomIncidents` (state/event 판정 로직, Main으로 이동)
- `process_frame()` (프레임 직접 입력 버전 — keypoints 직접 입력으로 대체)
- `handle_ack()` (ACK로 닫아줄 alert 상태 자체가 이제 AI 모듈에 없음)
- `match_tracks()`, `MAX_MATCH_DIST`, `MAX_MISSED` (track_id를 호출하는 쪽이 이미 정해서 주므로
  거리 기반 매칭이 더 이상 필요 없어짐)
- `threshold`, `window_sec`, `min_ratio`, `recent_check_sec` 파라미터 (Main으로 이동)

## 실행 환경

기존과 동일 — `Config/fallDetect_requirements.txt` 참고 (`numpy`, `torch`, `ultralytics`).
`ultralytics`는 `load_yolo=True`를 쓸 때만 실제로 모델을 로드하지만, import 자체는 항상
일어나므로 의존성 목록엔 변화 없음.

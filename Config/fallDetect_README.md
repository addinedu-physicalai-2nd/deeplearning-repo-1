# 낙상 감지 AI Service — fall-detection-service

이 디렉토리는 실제 배포용 낙상 감지 모듈(`fall_detector.py`)과 그 모델 파일들을 담고 있다.
`~/fall-detection/`은 연구/학습 기록용으로 그대로 남겨두고, 여기는 프로덕션에 들어갈
최소 구성만 모아둔 폴더.

## 구성 파일

| 파일 | 설명 |
|---|---|
| `AiServer/fall_detector.py` | 핵심 모듈. `FallDetectorSession.process_frame()`에 프레임 한 장을 넣으면 STATE/EVENT 메시지를 반환 |
| `Models/fall_lstm.pt` | 직접 학습시킨 낙상 판별 LSTM (프로젝트 실제 산출물) |
| `Models/yolov8n-pose.pt` | 사전학습된 범용 포즈 추정 모델 (외부 입력 재료, 학습 안 함) |
| `Config/fallDetect_requirements.txt` | 이 모듈 실행에 필요한 최소 패키지 목록 |

## 역할 경계

- **이 모듈이 하는 일**: 프레임 → YOLO pose → LSTM 판단 → STATE/EVENT 메시지 생성 (`to_json()`으로 직렬화까지)
- **이 모듈이 안 하는 일**: 프레임을 어디서 받아오는지(UDP 수신 등), 만들어진 메시지를 어떻게 전송하는지(HTTP 등)
  — 둘 다 호출하는 쪽(아직 구조 미정)의 책임

## 실행 환경

- **Python**: 3.12.3
- **venv 생성**: 이 모듈 전용 venv를 새로 만드는 걸 권장 (requirements가 최소 구성이라 기존
  연구용 venv와 분리):

  ```bash
  python3 -m venv ~/venv/fall_detection_service_venv
  source ~/venv/fall_detection_service_venv/bin/activate
  pip install -r Config/fallDetect_requirements.txt
  ```

### 필요한 패키지

| 패키지 | 버전(현재 venv 기준) | 용도 |
|---|---|---|
| `numpy` | 2.3.5 | keypoint 정규화, feature 벡터 연산 |
| `torch` | 2.14.0 | LSTM 모델 (`FallLSTM`) |
| `ultralytics` | 8.4.157 | YOLO pose 추론 (내부적으로 `torchvision` 0.29.0을 종속성으로 같이 설치함) |

`json`, `collections`, `dataclasses`, `datetime`, `pathlib`, `typing`은 파이썬 표준 라이브러리라
설치 대상이 아니다. `opencv-python`과 `requests`도 여기엔 필요 없다 — 이 모듈은 이미 디코딩된
프레임(`np.ndarray`)을 받아서 판단만 하고, 전송은 호출하는 쪽 책임이라(위 "역할 경계" 참고)
영상 입출력/HTTP 코드 자체가 없기 때문.

## 사용법

```python
from fall_detector import load_models, FallDetectorSession, to_json

yolo, lstm_model, device = load_models()   # 세션 시작 시 1회만
session = FallDetectorSession(camera_id="CAM-03", yolo=yolo, lstm_model=lstm_model,
                               device=device, fps=25.0)

# frame: np.ndarray (BGR), 매 프레임마다 호출
state_msg, event_msgs = session.process_frame(frame)

send_state(to_json(state_msg))          # 매 프레임 전송
for e in event_msgs:
    send_event(to_json(e))              # 발생했을 때만 전송
```

### `FallDetectorSession` 생성자 파라미터

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `camera_id` | (필수) | 이 세션이 대응하는 카메라 식별자. STATE/EVENT 메시지에 그대로 실림 |
| `yolo`, `lstm_model`, `device` | (필수) | `load_models()`가 반환한 값을 그대로 전달 |
| `fps` | (필수) | 입력 프레임의 초당 프레임 수. `window_sec`/`recent_check_sec`을 프레임 개수로 환산하는 데 쓰임 |
| `threshold` | 0.5 | LSTM 확률이 이 값 이상이면 해당 프레임을 "낙상처럼 보임(is_fall)"으로 취급 |
| `window_sec` | 1.5 | 확정(alert) 판단 디바운스 창. 최근 이 시간(초) 동안의 판정 중 `min_ratio` 이상이 낙상이어야 확정 |
| `min_ratio` | 0.7 | `window_sec` 창 안에서 낙상 판정 비율이 이 값 이상이어야 alert로 확정 (순간적 오탐 방지) |
| `recent_check_sec` | 1.0 | "최근 이 시간(초) 안에 checking 상태가 있었나"를 판단하는 창. lost로 빠질 때 risk를 열지 말지 결정하는 데 씀 |

### 입력 프레임 형식

`process_frame(frame)`의 `frame`은 `np.ndarray`, BGR, `(H, W, 3)` 형식이어야 한다 (OpenCV/YOLO 관례).

## 데이터 종류

### STATE — 사람별 실시간 상태 (그리기용, 매 프레임)

| 필드 | 타입 | 설명 |
|---|---|---|
| type | string | 항상 `"state"` |
| camera_id | string | 카메라 식별자 |
| timestamp | string (ISO8601, UTC) | 처리 시각 |
| persons | array | 이 프레임에서 추적 중인 사람 목록 |
| persons[].track_id | integer | 카메라 내 추적 ID (세션 내에서만 유효) |
| persons[].bbox | [x1,y1,x2,y2] | 픽셀 좌표 (원본 프레임 기준) |
| persons[].state | string | `normal` \| `checking` \| `alert` \| `lost` |
| persons[].fall_prob | float (0~1) | LSTM 확률값 (디버깅/로그용) |

예시:
```json
{
  "type": "state",
  "camera_id": "CAM-03",
  "timestamp": "2026-09-22T10:15:32.512Z",
  "persons": [
    { "track_id": 7, "bbox": [120.0, 88.0, 240.0, 400.0], "state": "checking", "fall_prob": 0.81 }
  ]
}
```

### EVENT — 사건(risk/alert) 전이 신호 (발생 시 1회)

| 필드 | 타입 | 설명 |
|---|---|---|
| type | string | 항상 `"event"` |
| event | string | `ALERT_OPENED` \| `RISK_OPENED` \| `RISK_CLOSED_AUTO` \| `RISK_ABSORBED_INTO_ALERT` |
| camera_id | string | |
| track_id | integer | 사건을 유발한 트랙 ID |
| timestamp | string (ISO8601, UTC) | |

각 event 값의 의미:

- **ALERT_OPENED** — 낙상이 확정된 순간 (최근 `window_sec`초 중 `min_ratio` 이상 낙상 판정). GUI: 낙상 탭 강제 전환 + 빨간 알림 카드. 사람이 "케어완료" 누르기 전까진 재발생 안 함.
- **RISK_OPENED** — 낙상처럼 보이던(checking) 사람이 화면에서 사라졌을(lost) 때. GUI: 노란 알림 카드 (탭 강제 전환 없음).
- **RISK_CLOSED_AUTO** — risk였던 사람이 다시 나타났는데 낙상이 아니었을 때 자동 발생. GUI: 노란 카드 제거.
- **RISK_ABSORBED_INTO_ALERT** — risk였던 사람이 다시 나타났는데 이번엔 낙상으로 확정됐을 때. 항상 ALERT_OPENED와 같은 순간에 같이 발생. GUI: 노란 카드 제거 + 빨간 카드/탭 전환.

예시:
```json
{
  "type": "event",
  "event": "ALERT_OPENED",
  "camera_id": "CAM-03",
  "track_id": 7,
  "timestamp": "2026-09-22T10:15:34.020Z"
}
```

### ACK — 역방향 (Main Service → AI Service, 아직 미구현)

"케어완료" 클릭 시 Main Service가 AI Service에 보내는 신호. `FallDetectorSession.handle_ack()`가
이 역할을 하도록 코드에 이미 준비돼 있지만, 이걸 실제로 호출해줄 HTTP 수신 서버(엔드포인트)는
아직 만들지 않았다. Main Service가 AI Service를 호출하는 구조로 갈지, AI Service가 폴링하는
구조로 갈지부터 정해야 함.

| 필드 | 타입 | 설명 |
|---|---|---|
| type | string | 항상 `"ack"` |
| camera_id | string | |
| incident_type | string | `alert` (risk는 자동종료/흡수 처리라 ack 불필요) |
| timestamp | string (ISO8601, UTC) | |

### 상태 전이표 (EVENT 발생 근거)

`RoomIncidents`가 내부적으로 어떤 전이에서 어떤 이벤트를 내는지 정리한 표. 트랙(사람)이
이전 프레임에서 다음 프레임으로 넘어갈 때 상태가 어떻게 바뀌었는지 기준.

| 이전 상태 | 전이 | 판단 / 동작 |
|---|---|---|
| normal | → lost | 무시 (일상적인 화면 이탈로 간주, risk 열지 않음) |
| checking | → lost | risk(위험) 사건 오픈 → `RISK_OPENED` |
| alert | → lost | alert 유지, 추가 알림 없음 (alert가 열려 있는 동안 risk는 새로 열리지 않음) |
| lost | → normal (재탐지, 낙상 아님) | risk 자동 종료 → `RISK_CLOSED_AUTO` |
| lost | → checking (재탐지, 낙상처럼 보임) | risk 유지 (아직 안심 못함, 닫지 않음) |
| lost | → alert (재탐지, 확정 조건 충족) | risk를 alert로 흡수 후 제거 → `ALERT_OPENED` + `RISK_ABSORBED_INTO_ALERT` 동시 발생 |
| (추적 소실 지속) | `MAX_MISSED`(10프레임) 초과 → 트랙 객체 삭제 | risk/alert는 트랙과 무관하게 방(room) 단위로 별도 관리되므로 그대로 유지됨 |

## 아직 안 된 부분

- **프레임 수신**: `process_frame()`은 이미 디코딩된 프레임(`np.ndarray`)을 받는다고 가정함.
  Main Service가 UDP로 보내주는 스트림을 실제로 받아서 프레임으로 만드는 부분은 별도로 필요 (미구현).
- **메시지 전송(HTTP)**: `to_json()`까지만 이 모듈 책임. 실제 `requests.post()` 같은 전송 코드는
  호출하는 쪽에서 붙여야 함.
- **ACK 수신 서버 없음**: 위 ACK 섹션 참고.
- **엔드포인트 경로 미정**: Main Service의 실제 라우트가 아직 안 정해짐.

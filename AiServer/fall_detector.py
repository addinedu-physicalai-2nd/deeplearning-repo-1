"""
fall_detector.py — 낙상 감지 AI 프로덕션 모듈

fall-detection-service/ 에 들어가는 실제 배포용 코드.
"한 프레임을 넣으면 STATE/EVENT 메시지가 나온다"는 것에만 집중하고,
프레임을 어디서/어떻게 받아오는지(UDP 수신 등)는 다루지 않는다 — 그건 호출하는 쪽(main.py 등,
아직 미정)의 책임이다.

역할 경계:
  - 이 모듈이 하는 일: 프레임 → YOLO pose → LSTM 판단 → STATE/EVENT 메시지 생성
  - 이 모듈이 안 하는 일: 프레임 수신(UDP 등), HTTP 전송, Main Service와의 통신 프로토콜
    (전송은 호출하는 쪽에서 to_json()으로 직렬화해서 원하는 방식으로 보내면 됨)

예시:
    yolo, lstm_model, device = load_models()
    session = FallDetectorSession(camera_id="CAM-03", yolo=yolo, lstm_model=lstm_model,
                                   device=device, fps=25.0)

    state_msg, event_msgs = session.process_frame(frame)
    requests.post(f"{MAIN_URL}/api/fall-detection/state", data=to_json(state_msg))
    for e in event_msgs:
        requests.post(f"{MAIN_URL}/api/fall-detection/event", data=to_json(e))
"""

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# ── 모델/특징 관련 상수 ─────────────────────────────────────────────
SEQ_LEN = 16
HIDDEN_SIZE = 64
NUM_LAYERS = 1
INPUT_SIZE = 68
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12

# ── 추적(트래킹) 관련 상수 ──────────────────────────────────────────
MAX_MATCH_DIST = 150
MAX_MISSED = 10


# ─────────────────────────────────────────────────────────────────
# 메시지 스키마 (Main Service로 나가는/들어오는 데이터 형태)
# ─────────────────────────────────────────────────────────────────

@dataclass
class PersonState:
    track_id: int
    bbox: list                  # [x1, y1, x2, y2] 픽셀 좌표
    state: Literal["normal", "checking", "alert", "lost"]
    fall_prob: float


@dataclass
class StateMessage:
    camera_id: str
    timestamp: str
    persons: list = field(default_factory=list)   # list[PersonState]
    type: Literal["state"] = "state"


@dataclass
class EventMessage:
    camera_id: str
    track_id: int
    timestamp: str
    event: Literal["ALERT_OPENED", "RISK_OPENED", "RISK_CLOSED_AUTO", "RISK_ABSORBED_INTO_ALERT"]
    type: Literal["event"] = "event"


@dataclass
class AckMessage:
    camera_id: str
    timestamp: str
    incident_type: Literal["alert"] = "alert"
    type: Literal["ack"] = "ack"


def to_json(message) -> str:
    """dataclass 메시지(StateMessage/EventMessage/AckMessage)를 JSON 문자열로 직렬화."""
    return json.dumps(asdict(message), ensure_ascii=False)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ─────────────────────────────────────────────────────────────────
# 모델 정의 / 로딩
# ─────────────────────────────────────────────────────────────────

class FallLSTM(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=0.3 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def load_models(lstm_path=None, yolo_path=None, device=None):
    """
    fall_lstm.pt(직접 학습한 낙상 분류기) + yolov8n-pose.pt(사전학습 포즈 추정 모델)를
    한 번만 로드한다. 프레임마다 다시 로드하지 말 것 — 세션 시작 시 한 번만 호출.

    경로를 지정하지 않으면 이 파일과 같은 디렉토리(fall-detection-service/)에서
    fall_lstm.pt / yolov8n-pose.pt를 찾는다.
    """
    base = Path(__file__).resolve().parent
    lstm_path = Path(lstm_path) if lstm_path else base / "fall_lstm.pt"
    yolo_path = Path(yolo_path) if yolo_path else base / "yolov8n-pose.pt"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    lstm_model = FallLSTM().to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()

    yolo = YOLO(str(yolo_path))

    return yolo, lstm_model, device


# ─────────────────────────────────────────────────────────────────
# 특징 추출 / 추적
# ─────────────────────────────────────────────────────────────────

def normalize(kpts):
    """YOLO pose keypoint(17개)를 hip 중심 상대좌표로 정규화 (위치/크기 무관하게)."""
    l_sh, r_sh = kpts[L_SHOULDER], kpts[R_SHOULDER]
    l_hip, r_hip = kpts[L_HIP], kpts[R_HIP]
    hip_mid = (l_hip + r_hip) / 2.0
    sh_mid = (l_sh + r_sh) / 2.0
    scale = np.linalg.norm(sh_mid - hip_mid)
    if scale < 1e-6:
        scale = 1e-6
    return (kpts - hip_mid) / scale


class Track:
    """카메라 한 대 내에서 한 사람을 추적하는 단위. window_len/recent_len은 세션(fps)마다
    다를 수 있어서 전역 상수가 아니라 생성 시점에 주입받는다."""
    _next_id = 1

    def __init__(self, center, box, valid, norm_pos, window_len, recent_len):
        self.id = Track._next_id
        Track._next_id += 1
        self.center = center
        self.box = box
        self.missed = 0
        self.last_norm = norm_pos if valid else np.zeros(34, dtype=np.float32)
        feat = np.concatenate([self.last_norm, np.zeros(34, dtype=np.float32)]).astype(np.float32)
        self.buffer = deque([feat] * SEQ_LEN, maxlen=SEQ_LEN)
        self.fall_prob = 0.0
        self.state_window = deque(maxlen=window_len)      # confirm 디바운스용 (최근 N초 fall 비율)
        self.recent_checking = deque(maxlen=recent_len)   # "최근 N초 내 checking 있었나"용
        self.confirmed = False

    def update(self, center, box, valid, norm_pos):
        if valid:
            vel = norm_pos - self.last_norm
            feat = np.concatenate([norm_pos, vel]).astype(np.float32)
            self.last_norm = norm_pos
        else:
            feat = np.concatenate([self.last_norm, np.zeros(34, dtype=np.float32)]).astype(np.float32)
        self.buffer.append(feat)
        self.center = center
        self.box = box
        self.missed = 0


def match_tracks(tracks, detections, window_len, recent_len):
    """현재 프레임의 detection들을 기존 트랙에 매칭하고, 새 트랙 생성/오래된 트랙 삭제를 처리."""
    unmatched_tracks = set(tracks.keys())
    unmatched_dets = set(range(len(detections)))
    pairs = []
    for tid in tracks:
        for di, (center, _, _, _) in enumerate(detections):
            d = np.linalg.norm(np.array(tracks[tid].center) - np.array(center))
            if d <= MAX_MATCH_DIST:
                pairs.append((d, tid, di))
    pairs.sort(key=lambda x: x[0])

    for d, tid, di in pairs:
        if tid in unmatched_tracks and di in unmatched_dets:
            center, box, valid, norm_pos = detections[di]
            tracks[tid].update(center, box, valid, norm_pos)
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)

    for di in unmatched_dets:
        center, box, valid, norm_pos = detections[di]
        t = Track(center, box, valid, norm_pos, window_len, recent_len)
        tracks[t.id] = t

    for tid in unmatched_tracks:
        tracks[tid].missed += 1

    to_delete = [tid for tid, t in tracks.items() if t.missed > MAX_MISSED]
    for tid in to_delete:
        del tracks[tid]


# ─────────────────────────────────────────────────────────────────
# 방(카메라) 단위 사건(risk/alert) 상태 — side-effect-free
#   기존 fall_detect_incident.py의 RoomIncidents는 events 리스트에 직접 append하고
#   send_http()까지 호출하는 형태였는데, 여기서는 "무슨 이벤트가 발생했는지"를
#   문자열 리스트로 반환만 하도록 바꿔서 호출부(전송 방식)와 완전히 분리한다.
# ─────────────────────────────────────────────────────────────────

class RoomIncidents:
    def __init__(self):
        self.alert = None
        self.risk = None

    def open_alert(self, frame_idx, t_sec):
        fired = []
        if self.alert is None:
            self.alert = {"opened_frame": frame_idx, "opened_time": t_sec}
            fired.append("ALERT_OPENED")
        if self.risk is not None:
            fired.append("RISK_ABSORBED_INTO_ALERT")
            self.risk = None
        return fired

    def open_risk(self, frame_idx, t_sec):
        if self.alert is not None:
            return []
        if self.risk is None:
            self.risk = {"opened_frame": frame_idx, "opened_time": t_sec}
            return ["RISK_OPENED"]
        return []

    def close_risk_auto(self, frame_idx, t_sec):
        if self.risk is not None:
            self.risk = None
            return ["RISK_CLOSED_AUTO"]
        return []

    def acknowledge_alert(self, frame_idx, t_sec):
        """Main Service로부터 '케어완료'(ACK)를 받았을 때 호출. ACK 수신 서버는 아직
        구현되지 않았으므로 현재는 어디서도 호출되지 않음 (README의 '아직 안 된 부분' 참고)."""
        if self.alert is not None:
            self.alert = None
            return ["ALERT_ACKNOWLEDGED"]
        return []


# ─────────────────────────────────────────────────────────────────
# 메인 엔트리 포인트
# ─────────────────────────────────────────────────────────────────

class FallDetectorSession:
    """카메라 한 대에 대응하는 낙상 감지 세션. process_frame()에 프레임을 하나씩 넣으면
    STATE 메시지 1개 + (있다면) EVENT 메시지 여러 개를 돌려준다.

    프레임을 어디서 가져오는지(파일, UDP 스트림 등)는 이 클래스의 책임이 아니다 —
    호출하는 쪽이 프레임(np.ndarray, BGR)만 넘겨주면 된다.
    """

    def __init__(self, camera_id, yolo, lstm_model, device, fps,
                 threshold=0.5, window_sec=1.5, min_ratio=0.7, recent_check_sec=1.0):
        self.camera_id = camera_id
        self.yolo = yolo
        self.model = lstm_model
        self.device = device
        self.fps = fps
        self.threshold = threshold
        self.min_ratio = min_ratio
        self.window_len = max(1, round(window_sec * fps))
        self.recent_len = max(1, round(recent_check_sec * fps))

        self.tracks = {}
        self.room = RoomIncidents()
        self.frame_idx = 0

    def process_frame(self, frame, frame_idx: Optional[int] = None, t_sec: Optional[float] = None):
        """
        프레임 1장을 처리해서 (StateMessage, list[EventMessage])를 반환한다.

        frame_idx/t_sec을 안 주면 내부 카운터로 자동 증가(오프라인 순차 처리 가정).
        실시간 스트림이라 프레임 번호/타임스탬프를 호출부가 따로 관리한다면 넘겨주면 된다.
        """
        if frame_idx is None:
            self.frame_idx += 1
            frame_idx = self.frame_idx
        else:
            self.frame_idx = frame_idx
        if t_sec is None:
            t_sec = frame_idx / self.fps

        timestamp = _now_iso()
        prev_missed = {tid: t.missed for tid, t in self.tracks.items()}

        results = self.yolo(frame, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            if results.keypoints is not None:
                all_kpts = results.keypoints.xy.cpu().numpy()
            else:
                all_kpts = [None] * len(boxes)
            for box, kpts in zip(boxes, all_kpts):
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                valid = False
                norm_pos = None
                if kpts is not None and kpts.shape[0] >= 17:
                    core = kpts[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]
                    if not np.any(np.all(core == 0, axis=1)):
                        valid = True
                        norm_pos = normalize(kpts).flatten().astype(np.float32)
                detections.append((center, box, valid, norm_pos))

        match_tracks(self.tracks, detections, self.window_len, self.recent_len)

        persons = []
        event_msgs = []

        for tid, t in list(self.tracks.items()):
            was_lost = prev_missed.get(tid, 0) > 0
            now_lost = t.missed > 0

            if now_lost:
                if not was_lost:
                    recent_checking = any(t.recent_checking)
                    if recent_checking:
                        for etype in self.room.open_risk(frame_idx, t_sec):
                            event_msgs.append(self._make_event(etype, tid, timestamp))
                persons.append(PersonState(
                    track_id=tid,
                    bbox=[float(v) for v in t.box],
                    state="lost",
                    fall_prob=float(t.fall_prob),
                ))
                continue

            feat_seq = np.stack(list(t.buffer))
            x = torch.tensor(feat_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                prob = torch.softmax(logits, dim=1)[0, 1].item()
            t.fall_prob = prob
            is_fall = prob >= self.threshold

            t.state_window.append(is_fall)
            t.recent_checking.append(is_fall)

            if len(t.state_window) == self.window_len:
                ratio = sum(t.state_window) / self.window_len
                t.confirmed = ratio >= self.min_ratio
            else:
                t.confirmed = False

            if was_lost and not is_fall:
                for etype in self.room.close_risk_auto(frame_idx, t_sec):
                    event_msgs.append(self._make_event(etype, tid, timestamp))

            if t.confirmed:
                for etype in self.room.open_alert(frame_idx, t_sec):
                    event_msgs.append(self._make_event(etype, tid, timestamp))
                state = "alert"
            elif is_fall:
                state = "checking"
            else:
                state = "normal"

            persons.append(PersonState(
                track_id=tid,
                bbox=[float(v) for v in t.box],
                state=state,
                fall_prob=float(t.fall_prob),
            ))

        state_msg = StateMessage(camera_id=self.camera_id, timestamp=timestamp, persons=persons)
        return state_msg, event_msgs

    def handle_ack(self, ack: AckMessage):
        """Main Service로부터 ACK(케어완료)를 수신했을 때 호출. ACK 수신 서버/경로는
        아직 없어서(README 참고) 현재는 호출하는 곳이 없음 — 나중에 그 부분 구현되면 연결."""
        if ack.incident_type == "alert":
            self.room.acknowledge_alert(self.frame_idx, self.frame_idx / self.fps)

    def _make_event(self, event_type, track_id, timestamp) -> EventMessage:
        return EventMessage(camera_id=self.camera_id, track_id=track_id, timestamp=timestamp, event=event_type)

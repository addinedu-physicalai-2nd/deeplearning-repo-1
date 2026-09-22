"""
fall_detector.py — 낙상 감지 AI 모듈 (판단근거 제공자)

역할 경계 (v2 — Main Service를 정책 엔진으로 분리):
  - 이 모듈이 하는 일: keypoints(이미 추출됨) → LSTM 추론 → fall_prob(판단근거) 계산 → dict 반환
  - 이 모듈이 안 하는 일:
      * 프레임에서 keypoints 뽑기 (YOLO pose는 이 모듈 밖, 이미 끝난 상태로 입력됨)
      * threshold 적용, state(normal/checking/alert/lost) 판정
      * risk/alert 상태 전이, EVENT 생성
      * Main Service와의 통신 프로토콜(HTTP 등)
    이 전부는 Main Service의 정책 엔진이 담당한다 — 그래야 정책(threshold 등)을
    바꿀 때마다 이 AI 모듈을 재배포하지 않아도 되고, gait/stretch 등 다른 모듈과
    Main 쪽에서 통합 판단할 수 있다.

예시:
    yolo, lstm_model, device = load_models()
    session = FallDetectorSession(camera_id="CAM-03", yolo=yolo, lstm_model=lstm_model,
                                   device=device, fps=25.0)

    keypoints_data = [
        (7, keypoints_of_person_7),   # keypoints: 17 x 2 (x, y)
        (8, keypoints_of_person_8),
    ]
    output = session.process_keypoints(frame_idx=1234, keypoints_list=keypoints_data)

    json_str = to_json(output)
    requests.post(f"{MAIN_URL}/api/fall-detection/raw", data=json_str)
"""

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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

# ── mode 값 (다른 AI 모듈과의 통합 판단 시 어떤 모듈 출력인지 구분용) ──
MODE_FALL_ONLY = 0   # 이 모듈(fall_detector)의 고정값. gait=1, stretch=2 등은 다른 모듈이 씀.

# 입력 keypoints엔 point별 confidence가 없어서(현재 스펙 기준) 아직 실제 계산 방법이
# 없다. Main Service/팀장님과 "confidence를 어떻게 정의할지" 확정되기 전까지의 placeholder.
_PLACEHOLDER_CONFIDENCE = 1.0


# ─────────────────────────────────────────────────────────────────
# 출력 스키마 — 판단근거(raw data)만 담는다. state/event는 여기 없음(Main이 만듦).
# ─────────────────────────────────────────────────────────────────

@dataclass
class RawDetection:
    """AI가 반환하는 원본 탐지 데이터 (판단근거만, state 판정 없음)."""
    track_id: int
    bbox: list          # [x1, y1, x2, y2] — 현재 입력엔 bbox가 없어서 기본값 [0,0,0,0] (아래 참고)
    raw_data: dict       # {"fall_prob": float, "confidence": float}


@dataclass
class AIOutput:
    """AI 모듈의 최종 출력 (JSON 직렬화 가능)."""
    camera_id: str
    timestamp: str
    mode: int = MODE_FALL_ONLY
    detections: list = field(default_factory=list)   # list[RawDetection]


def to_json(message) -> str:
    """dataclass 메시지(AIOutput 등)를 JSON 문자열로 직렬화."""
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


def load_models(lstm_path=None, yolo_path=None, device=None, load_yolo=False):
    """
    fall_lstm.pt(직접 학습한 낙상 분류기)를 로드한다.

    load_yolo=True로 주면 yolov8n-pose.pt도 같이 로드해서 반환하지만, 현재
    FallDetectorSession.process_keypoints()는 keypoints를 직접 입력받으므로
    yolo를 쓰지 않는다 — 프레임을 직접 넣는 옛 방식(process_frame)으로 되돌아갈
    가능성을 대비해 옵션으로만 남겨둠. 기본은 False (불필요한 모델 로딩 방지).
    """
    base = Path(__file__).resolve().parent
    lstm_path = Path(lstm_path) if lstm_path else base / "fall_lstm.pt"

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    lstm_model = FallLSTM().to(device)
    lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
    lstm_model.eval()

    yolo = None
    if load_yolo:
        yolo_path = Path(yolo_path) if yolo_path else base / "yolov8n-pose.pt"
        yolo = YOLO(str(yolo_path))

    return yolo, lstm_model, device


# ─────────────────────────────────────────────────────────────────
# 특징 추출
# ─────────────────────────────────────────────────────────────────

def normalize(kpts):
    """YOLO pose keypoint(17개, (x,y))를 hip 중심 상대좌표로 정규화 (위치/크기 무관하게)."""
    kpts = np.asarray(kpts, dtype=np.float32)
    l_sh, r_sh = kpts[L_SHOULDER], kpts[R_SHOULDER]
    l_hip, r_hip = kpts[L_HIP], kpts[R_HIP]
    hip_mid = (l_hip + r_hip) / 2.0
    sh_mid = (l_sh + r_sh) / 2.0
    scale = np.linalg.norm(sh_mid - hip_mid)
    if scale < 1e-6:
        scale = 1e-6
    return (kpts - hip_mid) / scale


class Track:
    """track_id 하나에 대응하는 LSTM 입력 시퀀스 버퍼.

    track_id는 이 클래스가 만드는 게 아니라 호출하는 쪽(이미 포즈 추정 + 트래킹을
    끝낸 상위 시스템)이 매 프레임 넘겨준다 — 여기선 그 track_id별로 최근 SEQ_LEN
    프레임의 feature만 쌓아서 LSTM에 넣을 시퀀스를 유지한다. state 판정용 필드
    (state_window/recent_checking/confirmed)는 없음 — 그건 Main이 한다.
    """

    def __init__(self, track_id, norm_pos, bbox=None):
        self.id = track_id
        self.last_norm = norm_pos
        feat = np.concatenate([norm_pos, np.zeros(34, dtype=np.float32)]).astype(np.float32)
        self.buffer = deque([feat] * SEQ_LEN, maxlen=SEQ_LEN)
        # 현재 입력 포맷(track_id, keypoints)엔 bbox가 없어서 기본값. keypoints의
        # min/max로 계산해서 채우는 게 나을 수도 있는데, 우선 스펙 그대로 둠 — 필요하면 바꿔줘.
        self.box = bbox if bbox is not None else [0.0, 0.0, 0.0, 0.0]

    def update(self, norm_pos, bbox=None):
        vel = norm_pos - self.last_norm
        feat = np.concatenate([norm_pos, vel]).astype(np.float32)
        self.buffer.append(feat)
        self.last_norm = norm_pos
        if bbox is not None:
            self.box = bbox


# ─────────────────────────────────────────────────────────────────
# 메인 엔트리 포인트
# ─────────────────────────────────────────────────────────────────

class FallDetectorSession:
    """카메라 한 대에 대응하는 낙상 판단근거 세션.

    process_keypoints()에 이미 추출된 keypoints를 넣으면 fall_prob(판단근거)만
    계산해서 반환한다. state(normal/checking/alert/lost) 판정, risk/alert 전이,
    EVENT 생성은 여기서 안 함 — 전부 Main Service의 정책 엔진 몫.
    """

    def __init__(self, camera_id, yolo=None, lstm_model=None, device=None, fps=None):
        self.camera_id = camera_id
        self.yolo = yolo          # 현재 process_keypoints()는 이걸 쓰지 않음 (사용 안 함)
        self.model = lstm_model
        self.device = device
        self.fps = fps            # 지금 로직엔 안 쓰이지만, 추후 필요해질 수 있어 남겨둠

        self.tracks = {}          # track_id -> Track
        self.frame_idx = 0

    def process_keypoints(self, frame_idx: int, keypoints_list) -> AIOutput:
        """
        이미 추출된 keypoints에서 fall_prob(판단근거)만 계산한다.

        Args:
            frame_idx: 프레임 번호 (참고/로깅용. 시퀀스 순서는 track별 buffer가 보장)
            keypoints_list: [(track_id, keypoints), ...] 리스트.
                keypoints는 17 x 2 ([[x, y], ...]) 배열 또는 그와 동등한 리스트.
                keypoints가 None이거나 17개 미만이면 그 track은 이번 프레임에서 건너뜀.

        Returns:
            AIOutput — camera_id, timestamp, mode, detections(list[RawDetection]).
            state/event는 없음 — Main Service가 fall_prob 시계열을 보고 직접 판단.
        """
        self.frame_idx = frame_idx
        timestamp = _now_iso()
        detections = []

        for track_id, keypoints in keypoints_list:
            if keypoints is None or len(keypoints) < 17:
                continue

            norm_pos = normalize(keypoints).flatten().astype(np.float32)

            if track_id in self.tracks:
                track = self.tracks[track_id]
                track.update(norm_pos)
            else:
                track = Track(track_id=track_id, norm_pos=norm_pos)
                self.tracks[track_id] = track

            feat_seq = np.stack(list(track.buffer))
            x = torch.tensor(feat_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                fall_prob = torch.softmax(logits, dim=1)[0, 1].item()

            detections.append(RawDetection(
                track_id=track_id,
                bbox=[float(v) for v in track.box],
                raw_data={
                    "fall_prob": float(fall_prob),
                    "confidence": _PLACEHOLDER_CONFIDENCE,
                },
            ))

        return AIOutput(
            camera_id=self.camera_id,
            timestamp=timestamp,
            mode=MODE_FALL_ONLY,
            detections=detections,
        )

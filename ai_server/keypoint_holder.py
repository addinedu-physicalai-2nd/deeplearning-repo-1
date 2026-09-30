# ai_server/keypoint_holder.py
"""사라진 관절은 직전 좌표로 채우기 (카메라 1대당 1개, AIManager가 보유).

Ultralytics YOLO pose는 신뢰도가 낮은(0.5 미만) 관절의 좌표를 (0, 0)으로 준다.
신뢰도가 경계에서 오르내리면 같은 관절이 프레임마다 제자리 ↔ (0, 0)을 오가서
스켈레톤 선이 화면 모서리로 튀고, 보행/스트레칭 분석 입력도 흔들린다.

여기서는 트랙(사람)별로 관절마다 마지막으로 보인 좌표와 시각을 기억해 두고,
(0, 0)으로 온 관절을 그 좌표로 채운다. 단, 너무 오래된 좌표로 계속 채우면
화면 밖으로 나간 발목이 제자리에 멈춰 있는 것처럼 보이므로 HOLD_SEC까지만 유지하고,
그보다 오래 안 보이면 원래대로 (0, 0)을 그대로 둔다.
"""
import time

import numpy as np

HOLD_SEC = 1.0          # 사라진 관절을 직전 좌표로 채워주는 최대 시간
FORGET_SEC = 5.0        # 이 시간 동안 안 보인 트랙은 기억에서 지움 (메모리 정리)


class KeypointHolder:
    def __init__(self, hold_sec=HOLD_SEC, forget_sec=FORGET_SEC):
        self.hold_sec = hold_sec
        self.forget_sec = forget_sec
        self.last_xy = {}       # track_id → (17, 2) 관절별 마지막으로 보인 좌표
        self.last_time = {}     # track_id → (17,) 관절별 마지막으로 보인 시각
        self.track_seen = {}    # track_id → 트랙이 마지막으로 검출된 시각

    def apply(self, keypoints_dict, now=None):
        """{track_id: (17, 2)} → 같은 형식. (0, 0) 관절은 HOLD_SEC 이내면 직전 좌표로 채움.
        입력 dict는 건드리지 않고 새 dict를 돌려준다."""
        now = time.monotonic() if now is None else now
        filled = {}
        for track_id, kpts in keypoints_dict.items():
            kpts = np.asarray(kpts, dtype=np.float32)
            missing = (kpts[:, 0] == 0) & (kpts[:, 1] == 0)

            if track_id not in self.last_xy:
                # 처음 보는 트랙: 채울 기록이 없으니 그대로. 보이는 관절만 기억
                self.last_xy[track_id] = kpts.copy()
                self.last_time[track_id] = np.where(missing, -np.inf, now)
                self.track_seen[track_id] = now
                filled[track_id] = kpts
                continue

            last_xy = self.last_xy[track_id]
            last_time = self.last_time[track_id]
            fresh = (now - last_time) <= self.hold_sec     # 최근에 본 적 있는 관절
            fill = missing & fresh

            out = kpts.copy()
            out[fill] = last_xy[fill]

            # 이번에 실제로 보인 관절만 기억 갱신 (채운 값으로 갱신하면 시간이 계속 연장됨)
            seen = ~missing
            last_xy[seen] = kpts[seen]
            last_time[seen] = now
            self.track_seen[track_id] = now
            filled[track_id] = out

        self._forget_old(now)
        return filled

    def reset(self):
        self.last_xy.clear()
        self.last_time.clear()
        self.track_seen.clear()

    def _forget_old(self, now):
        for track_id in [t for t, seen in self.track_seen.items() if now - seen > self.forget_sec]:
            del self.last_xy[track_id]
            del self.last_time[track_id]
            del self.track_seen[track_id]

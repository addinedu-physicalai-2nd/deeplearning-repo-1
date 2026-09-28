"""
fall_judge.py — Main Service 낙상 판단 로직

Main Service가 HTTP POST body로 AIOutput을 받으면, 웹 프레임워크(Flask/FastAPI 등)가
이미 JSON을 dict로 파싱해서 넘겨준다. 그래서 이 모듈은 JSON 문자열이 아니라
이미 파싱된 dict(ai_output)를 받는다 — stretching_analyzer.py의
analyze(aioutput: dict)와 같은 컨벤션.
"""

from collections import deque
from datetime import datetime
from dataclasses import dataclass, field

FALL_THRESHOLD = 0.5
WINDOW_SEC = 1.5
MIN_RATIO = 0.7
LOST_TIMEOUT_SEC = 1.0
TRACK_EXPIRE_SEC = 10.0

STATE_NORMAL = "normal"
STATE_CHECKING = "checking"
STATE_ALERT = "alert"

STATE_COLOR = {
    STATE_NORMAL: "none",
    STATE_CHECKING: "orange",
    STATE_ALERT: "red",
}

_SEVERITY = {STATE_NORMAL: 0, STATE_CHECKING: 1, STATE_ALERT: 2}


@dataclass
class TrackState:
    track_id: int
    fall_window: deque = field(default_factory=deque)
    state: str = STATE_NORMAL
    last_bbox: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    last_seen_ts: float = 0.0


def _parse_timestamp(ts_str: str) -> float:
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


def is_fall_frame(fall_prob: float, threshold: float = FALL_THRESHOLD) -> bool:
    return fall_prob >= threshold


def update_window_and_check_confirmed(
    track: TrackState, is_fall: bool, now: float,
    window_sec: float = WINDOW_SEC, min_ratio: float = MIN_RATIO,
) -> bool:
    track.fall_window.append((now, is_fall))
    while track.fall_window and now - track.fall_window[0][0] > window_sec:
        track.fall_window.popleft()

    if not track.fall_window:
        return False
    fall_count = sum(1 for _, f in track.fall_window if f)
    return (fall_count / len(track.fall_window)) >= min_ratio


def transition_state(track: TrackState, is_fall: bool, confirmed: bool) -> str:
    if track.state == STATE_ALERT:
        return STATE_ALERT
    if confirmed:
        return STATE_ALERT
    if is_fall:
        return STATE_CHECKING
    if track.state == STATE_CHECKING and not any(f for _, f in track.fall_window):
        return STATE_NORMAL
    return track.state


def state_to_color(state: str) -> str:
    return STATE_COLOR.get(state, "none")


class FallJudgeService:
    def __init__(self):
        self.tracks: dict[tuple[str, int], TrackState] = {}

    def process(self, ai_output: dict) -> dict | None:
        if ai_output.get("mode") != 0:
            return None

        camera_id = ai_output["camera_id"]
        now = _parse_timestamp(ai_output["timestamp"])
        seen_ids = set()
        track_results = []
        events = []

        for det in ai_output["detections"]:
            track_id = det["track_id"]
            key = (camera_id, track_id)
            seen_ids.add(track_id)

            track = self.tracks.setdefault(key, TrackState(track_id=track_id))
            prev_state = track.state
            track.last_bbox = det["bbox"]
            track.last_seen_ts = now

            fall_prob = det["raw_data"]["fall_prob"]
            is_fall = is_fall_frame(fall_prob)
            confirmed = update_window_and_check_confirmed(track, is_fall, now)
            track.state = transition_state(track, is_fall, confirmed)

            if track.state != prev_state:
                events.append({
                    "track_id": track_id,
                    "event": f"{prev_state.upper()}_TO_{track.state.upper()}",
                    "timestamp": ai_output["timestamp"],
                })

            track_results.append({
                "track_id": track_id, "bbox": track.last_bbox,
                "state": track.state, "color": state_to_color(track.state),
            })

        for (cam_id, track_id), track in list(self.tracks.items()):
            if cam_id != camera_id or track_id in seen_ids:
                continue
            gap = now - track.last_seen_ts

            if track.state == STATE_NORMAL:
                if gap > LOST_TIMEOUT_SEC:
                    del self.tracks[(cam_id, track_id)]
                continue

            track_results.append({
                "track_id": track_id, "bbox": track.last_bbox,
                "state": track.state, "color": state_to_color(track.state),
            })
            if gap > TRACK_EXPIRE_SEC:
                del self.tracks[(cam_id, track_id)]

        return {
            "camera_id": camera_id,
            "camera_status": self.get_camera_status(camera_id),
            "tracks": track_results,
            "events": events,
        }

    def get_camera_status(self, camera_id: str) -> str:
        camera_tracks = [t for (cam, _), t in self.tracks.items() if cam == camera_id]
        if not camera_tracks:
            return STATE_NORMAL
        return max((t.state for t in camera_tracks), key=lambda s: _SEVERITY[s])

    def resolve(self, camera_id: str, track_id: int):
        key = (camera_id, track_id)
        if key in self.tracks:
            self.tracks[key].state = STATE_NORMAL
            self.tracks[key].fall_window.clear()

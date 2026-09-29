# admin_gui/drawing.py
"""영상 프레임 위에 그리는 함수들 + 화면 크기 변환 유틸. 기존 admin_gui.py 초안의
"mode별 그리기" 섹션을 그대로 옮겼다 — 로직은 안 건드림, import 경로만 정리."""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .qt_compat import QImage, QPixmap

VIEW_W, VIEW_H = 480, 360

FALL_COLOR = {'none': (0, 255, 0), 'orange': (0, 165, 255), 'red': (0, 0, 255)}   # BGR

# 스트레칭 판정 — 실제 main_service/stretching_analyzer.py가 주는 level은 숫자
# 1~5단계가 아니라 'good'/'adjust'/'check' 3단계 문자열이다. (화면에 "Lv.check"라고
# 그대로 찍히던 게 그 증거 — 예전에 여기 int 키 딕셔너리를 쓰고 있어서 'check' 같은
# 문자열 키가 하나도 안 걸리고 전부 UNKNOWN_COLOR로만 그려지는 바람에 스켈레톤이
# 흑백처럼 보였다.) overall뿐 아니라 limbs/joint_accuracy도 부위별로 각각 이 3단계
# 중 하나를 주므로, stretch_tab.py에서 부위별 색으로 draw_skeleton에 넘긴다.
# LEVEL_COLOR: OpenCV용 BGR, LEVEL_HEX: Qt 스타일시트용 hex — 같은 색을 두 형식으로.
LEVEL_COLOR = {
    'good': (0, 200, 0),       # 초록 — 잘 맞음
    'adjust': (0, 200, 230),   # 노랑 — 약간 보정 필요
    'check': (0, 0, 230),      # 빨강 — 확인 필요
}
LEVEL_TEXT = {'good': '양호', 'adjust': '보정 필요', 'check': '확인 필요'}
LEVEL_HEX = {
    'good': '#1f9d3c',
    'adjust': '#e0b400',
    'check': '#dc2f2f',
}
UNKNOWN_COLOR = (200, 200, 200)
UNKNOWN_HEX = '#9aa2ae'

# COCO-17 스켈레톤 (StretchingAnalyzer의 limbs / joint_accuracy 키와 대응)
LIMB_LINES = {
    'left_arm': [(5, 7), (7, 9)],
    'right_arm': [(6, 8), (8, 10)],
    'left_leg': [(11, 13), (13, 15)],
    'right_leg': [(12, 14), (14, 16)],
}
TORSO_LINES = [(5, 6), (11, 12), (5, 11), (6, 12)]
JOINT_POINTS = {
    'shoulder': (5, 6), 'elbow': (7, 8), 'wrist': (9, 10),
    'hip': (11, 12), 'knee': (13, 14), 'ankle': (15, 16),
}


def draw_box(frame, bbox, color, label):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def score_color(score):
    if score >= 70:
        return (0, 255, 0)
    if score >= 50:
        return (0, 255, 255)
    return (0, 0, 255)


def draw_camera_overlay(frame, camera_id, room_label):
    """영상 좌상단에 '카메라ID - 병실', 우상단에 시각 + REC 점 표시 (목업 스타일).
    fit_to_view()로 480x360 등 뷰 크기에 맞춘 뒤 호출할 것 (원본 해상도에서 하면
    텍스트가 리사이즈되면서 뭉개짐)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, f"{camera_id} - {room_label}", (10, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    (tw, _), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, timestamp, (w - tw - 20, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.circle(frame, (w - tw - 32, 17), 4, (0, 0, 255), -1)


def draw_stretch_badge(frame, level, score):
    """스트레칭 종합 판정을 옆 패널 대신 영상 우측 상단에 직접 찍는다.
    level: 'good'/'adjust'/'check' 문자열, 없으면 회색 '판정 대기'로 표시. score: 0~100.
    2026-09: "Lv.check" 같은 내부 라벨 문구는 화면에 노출하지 않고 점수만 보여준다
    (배경색으로는 계속 good/adjust/check 판정을 구분해서 보여준다)."""
    color = LEVEL_COLOR.get(level, UNKNOWN_COLOR)
    label = f"{score:.0f}점" if level else "판정 대기"
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pad_x, pad_y = 14, 10
    box_w, box_h = tw + pad_x * 2, th + pad_y * 2
    x2, y2 = w - 16, 16 + box_h
    x1, y1 = x2 - box_w, 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, label, (x1 + pad_x, y2 - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def draw_session_result(width, height, avg_score=None, course_label=None):
    """스트레칭 세션이 끝난 직후 '실시간 카메라' 자리에 그대로 박아 넣는 결과 화면.
    화면을 완전히 검정으로 덮고 가운데에 코스 이름(있으면) + 평균 점수를 크게 찍는다.
    avg_score가 없으면(완료 못 하고 종료된 경우) 점수 없이 "측정 결과 없음"만 표시.
    점수 색은 score_color()로 통일 — 스트레칭 good/adjust/check 3단계가 아니라
    0~100 평균 점수 자체의 높낮이를 보여주는 것이므로 기존 낙상/보행에서 쓰던
    70/50 기준의 score_color 팔레트를 그대로 쓴다."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)   # 완전 검정
    cx = width // 2
    cy = height // 2

    if course_label:
        (tw, _), _ = cv2.getTextSize(course_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(canvas, course_label, (cx - tw // 2, cy - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 190, 190), 2, cv2.LINE_AA)

    if avg_score is None:
        label = "측정 결과 없음"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(canvas, label, (cx - tw // 2, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (150, 150, 150), 2, cv2.LINE_AA)
        return canvas

    label = "평균 점수"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(canvas, label, (cx - tw // 2, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 170, 170), 2, cv2.LINE_AA)

    score_text = f"{avg_score:.0f}점"
    (tw, _), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
    cv2.putText(canvas, score_text, (cx - tw // 2, cy + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, score_color(avg_score), 5, cv2.LINE_AA)
    return canvas


def draw_fall(frame, data):
    """낙상 탭용 — bbox를 프레임에 직접 그림. 테두리 색/팝업 로직은 fall_tab.py에서 담당."""
    for track in data.get('tracks', []):
        color = FALL_COLOR.get(track.get('color'), (0, 255, 0))
        draw_box(frame, track['bbox'], color, f"ID {track['track_id']} {track['state']}")
    return data.get('camera_status', '-')


GAIT_SKELETON_COLOR = (255, 191, 0)   # BGR — 걷는 스켈레톤 표시용 (bbox+텍스트 아님)


def draw_gait(frame, data):
    """보행 탭 영상 — bbox에 "walking" 같은 라벨을 붙이는 게 아니라, 걷는 동안
    스켈레톤이 그대로 움직이는 모습을 보여준다. 질환별 확률(도넛차트)은
    gait_tab.py가 raw_data.cumulative_scores로 따로 그린다."""
    detections = data.get('detections', [])
    if not detections:
        return None
    for det in detections:
        keypoints = det.get('keypoints_px') or det.get('keypoints')
        if not keypoints:
            # cumulative_scores가 det['raw_data']에 들어있는 것처럼(render() 참고),
            # keypoints도 최상위가 아니라 raw_data 밑에 있을 수 있어서 한 번 더 찾아본다.
            raw = det.get('raw_data', {})
            keypoints = raw.get('keypoints_px') or raw.get('keypoints')
        if keypoints:
            draw_skeleton(frame, keypoints, default=GAIT_SKELETON_COLOR)
        else:
            # keypoints를 끝내 못 찾은 경우에만 bbox로 폴백 (실제 스키마 확인되면 제거 가능)
            draw_box(frame, det['bbox'], GAIT_SKELETON_COLOR, f"ID {det['track_id']}")
    return detections


def draw_stretch(frame, data):
    """참고용 — 실제 스트레칭 탭 그리기는 stretch_tab.py의 on_result()가 직접 한다.
    전체 종합 판정(good/adjust/check)만 예시로 텍스트로 찍어본다."""
    if 'tracking_data' not in data:
        for det in data.get('detections', []):
            draw_box(frame, det['bbox'], (255, 255, 0), f"ID {det['track_id']}")
        return "기준 자세 없음 (--stretch-ref 미지정)"

    tracking = data['tracking_data']
    if not tracking:
        return "기준 영상에 해당 프레임 없음"

    texts = []
    for track_id, info in tracking.items():
        overall = info['overall']
        level = overall.get('level')
        color = LEVEL_COLOR.get(level, UNKNOWN_COLOR)
        label = LEVEL_TEXT.get(level, level or '-')
        draw_box(frame, info['bbox'], color, f"ID {track_id} {label} {overall.get('score', 0):.0f}")
        texts.append(f"ID {track_id}: {label} {overall.get('score', 0):.0f}점")
    return " / ".join(texts)


def draw_skeleton(frame, keypoints, limb_colors=None, joint_colors=None, default=(255, 255, 255)):
    """COCO-17 스켈레톤. limb_colors/joint_colors: {이름: BGR} (없으면 default 색)"""
    limb_colors = limb_colors or {}
    joint_colors = joint_colors or {}
    pts = [(int(p[0]), int(p[1])) for p in keypoints[:17]]
    if len(pts) < 17:
        return

    def visible(i):
        return pts[i][0] > 0 or pts[i][1] > 0      # YOLO가 못 찾은 점은 (0, 0)

    for a, b in TORSO_LINES:
        if visible(a) and visible(b):
            cv2.line(frame, pts[a], pts[b], (160, 160, 160), 2)
    for limb, lines in LIMB_LINES.items():
        color = tuple(int(c) for c in limb_colors.get(limb, default))
        for a, b in lines:
            if visible(a) and visible(b):
                cv2.line(frame, pts[a], pts[b], color, 4)
    for joint, idxs in JOINT_POINTS.items():
        color = tuple(int(c) for c in joint_colors.get(joint, default))
        for i in idxs:
            if visible(i):
                cv2.circle(frame, pts[i], 6, color, -1)
    if visible(0):
        cv2.circle(frame, pts[0], 6, default, -1)


def fit_to_view(frame, width=VIEW_W, height=VIEW_H):
    """비율 유지하면서 뷰 크기에 맞추고 남는 부분은 검은색"""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def to_pixmap(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image)


def load_reference(path):
    """기준 자세 JSON 로드 + frame_index를 0부터 시작하도록 재정렬 (main_server와 같은 규칙)"""
    with open(path, 'r', encoding='utf-8') as f:
        reference = json.load(f)
    skeletons = reference.get('skeletons', [])
    offset = min((s['frame_index'] for s in skeletons), default=0)
    return {s['frame_index'] - offset: s for s in skeletons}

# admin_gui/admin_gui.py
"""Admin GUI: 영상(UDP 9998) + 분석 결과(TCP 9999)를 frame_idx로 매칭해서 표시"""
import argparse
import json
import os
import queue
import socket
import struct
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

# opencv-python이 설정한 Qt 플러그인 경로가 PyQt와 충돌하는 문제 방지
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QColor, QImage, QPixmap
    from PyQt6.QtWidgets import (QApplication, QComboBox, QGridLayout, QHBoxLayout, QLabel,
                                 QListWidget, QListWidgetItem, QMainWindow, QTabWidget,
                                 QVBoxLayout, QWidget)
except ImportError:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QColor, QImage, QPixmap
    from PyQt5.QtWidgets import (QApplication, QComboBox, QGridLayout, QHBoxLayout, QLabel,
                                 QListWidget, QListWidgetItem, QMainWindow, QTabWidget,
                                 QVBoxLayout, QWidget)

from config.settings import CAMERA_PORTS, GUI_VIDEO_PORT, GUI_INFO_PORT

FRAME_BUFFER_LEN = 150        # 카메라별 보관 프레임 수 (15FPS 기준 약 10초)
NO_RESULT_TIMEOUT_SEC = 1.0   # 이 시간 동안 AI 결과가 없으면 원본 영상만 표시
UPDATE_INTERVAL_MS = 30       # 화면 갱신 주기
VIEW_W, VIEW_H = 480, 360

FALL_COLOR = {'none': (0, 255, 0), 'orange': (0, 165, 255), 'red': (0, 0, 255)}   # BGR
MODE_NAME = {0: 'FALL', 1: 'GAIT', 2: 'STRETCH'}
MODE_STRETCH = 2

# 스트레칭 단계 → 화면 표시 (단계 기준값은 main_server의 StretchingAnalyzer에만 있음)
LEVEL_COLOR = {'good': (0, 255, 0), 'adjust': (0, 255, 255), 'check': (0, 0, 255)}   # BGR (OpenCV로 그림)
LEVEL_TEXT = {'good': 'GOOD', 'adjust': 'ADJUST', 'check': 'CHECK'}
UNKNOWN_COLOR = (200, 200, 200)

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


# ============ 영상 버퍼 ============
class FrameStore:
    """camera_id별 {frame_idx: jpg_bytes} 버퍼 (스레드 안전)"""

    def __init__(self, maxlen=FRAME_BUFFER_LEN):
        self.maxlen = maxlen
        self.buffers = {cam_id: OrderedDict() for cam_id in CAMERA_PORTS}
        self.latest_idx = {cam_id: None for cam_id in CAMERA_PORTS}
        self.lock = threading.Lock()

    def put(self, camera_id, frame_idx, jpg_bytes):
        with self.lock:
            buf = self.buffers.get(camera_id)
            if buf is None or frame_idx is None:
                return
            buf[frame_idx] = jpg_bytes
            while len(buf) > self.maxlen:
                buf.popitem(last=False)
            self.latest_idx[camera_id] = frame_idx

    def pop(self, camera_id, frame_idx):
        """결과와 같은 frame_idx의 프레임 꺼내기 (그보다 오래된 프레임은 삭제)"""
        with self.lock:
            buf = self.buffers.get(camera_id)
            if buf is None:
                return None
            jpg_bytes = buf.pop(frame_idx, None)
            if jpg_bytes is not None:
                for idx in [i for i in buf if i < frame_idx]:
                    del buf[idx]
            return jpg_bytes

    def latest(self, camera_id):
        """가장 최근에 받은 프레임 (AI 결과가 없을 때 원본 표시용)"""
        with self.lock:
            idx = self.latest_idx.get(camera_id)
            buf = self.buffers.get(camera_id)
            if idx is None or buf is None:
                return None, None
            return idx, buf.get(idx)

    def get_latest_idx(self, camera_id):
        with self.lock:
            return self.latest_idx.get(camera_id)


# ============ 수신 스레드 ============
def video_receiver(store, stop_event, port=GUI_VIDEO_PORT):
    """UDP 영상 수신: [2바이트 헤더 길이][JSON 헤더][JPEG]"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(1.0)
    print(f"[GUI] video listening on UDP {port}")

    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break

        if len(data) < 2:
            continue
        header_len = struct.unpack('!H', data[:2])[0]
        try:
            header = json.loads(data[2:2 + header_len].decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        store.put(header.get('camera_id'), header.get('frame_idx'), data[2 + header_len:])

    sock.close()


class MainLink:
    """main_server와의 TCP 연결 (결과 수신 + 명령 송신에 같은 연결 사용)"""

    def __init__(self):
        self.conn = None
        self.lock = threading.Lock()

    def set_conn(self, conn):
        with self.lock:
            self.conn = conn

    def send(self, cmd):
        line = (json.dumps(cmd, ensure_ascii=False) + '\n').encode('utf-8')
        with self.lock:
            if self.conn is None:
                return False
            try:
                self.conn.sendall(line)
                return True
            except OSError:
                return False


def result_receiver(result_queue, stop_event, link, port=GUI_INFO_PORT):
    """TCP 결과 수신 서버: main_server가 접속해서 한 줄에 JSON 하나씩 보냄"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    server.settimeout(1.0)
    print(f"[GUI] result listening on TCP {port}")

    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        print(f"[GUI] main_server connected: {addr}")
        conn.settimeout(1.0)
        link.set_conn(conn)
        buffer = b''
        while not stop_event.is_set():
            try:
                chunk = conn.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break

            buffer += chunk
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                if not line.strip():
                    continue
                try:
                    result_queue.put(json.loads(line.decode('utf-8')))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
        link.set_conn(None)
        conn.close()
        print("[GUI] main_server disconnected")

    server.close()


# ============ mode별 그리기 (frame은 BGR, 원본 해상도 기준 좌표) ============
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


def draw_fall(frame, data):
    for track in data.get('tracks', []):
        color = FALL_COLOR.get(track.get('color'), (0, 255, 0))
        draw_box(frame, track['bbox'], color, f"ID {track['track_id']} {track['state']}")
    return f"상태: {data.get('camera_status', '-')}"


def draw_gait(frame, data):
    detections = data.get('detections', [])
    if not detections:
        return "보행 데이터 수집 중 (30프레임)"

    texts = []
    for det in detections:
        raw = det.get('raw_data', {})
        score = raw.get('gait_score', 0.0)
        cumulative = raw.get('cumulative_scores', {})
        top = max(cumulative, key=cumulative.get) if cumulative else '-'
        draw_box(frame, det['bbox'], score_color(score), f"ID {det['track_id']} gait {score:.0f}")
        texts.append(f"ID {det['track_id']}: {top} {cumulative.get(top, 0.0):.2f}")
    return " / ".join(texts)


def draw_stretch(frame, data):
    if 'tracking_data' not in data:          # 기준 자세 없이 AI 결과 그대로 온 경우
        for det in data.get('detections', []):
            draw_box(frame, det['bbox'], (255, 255, 0), f"ID {det['track_id']}")
        return "기준 자세 없음 (--stretch-ref 미지정)"

    tracking = data['tracking_data']
    if not tracking:
        return "기준 영상에 해당 프레임 없음"

    texts = []
    for track_id, info in tracking.items():
        overall = info['overall']
        color = LEVEL_COLOR.get(overall['level'], UNKNOWN_COLOR)
        label = LEVEL_TEXT.get(overall['level'], overall['level'])
        draw_box(frame, info['bbox'], color, f"ID {track_id} {label} {overall['score']:.0f}")
        texts.append(f"ID {track_id}: {label} {overall['score']:.0f}점")
    return " / ".join(texts)


DRAWERS = {0: draw_fall, 1: draw_gait, 2: draw_stretch}


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


class StretchingTab(QWidget):
    """왼쪽: 기준 영상+기준 스켈레톤 / 가운데: 내 캠+분석 스켈레톤 / 오른쪽: 영상 목록"""

    def __init__(self, link, stretch_dir):
        super().__init__()
        self.link = link
        self.stretch_dir = Path(stretch_dir)
        self.session = None

        self.ref_view = self._make_view("기준 영상")
        self.my_view = self._make_view("내 영상")
        self.ref_status = QLabel("영상을 선택하세요")
        self.my_status = QLabel("-")
        self.my_status.setWordWrap(True)

        self.camera_combo = QComboBox()
        self.camera_combo.addItems(list(CAMERA_PORTS))
        self.video_list = QListWidget()
        self.video_list.setFixedWidth(260)
        self.video_list.itemClicked.connect(self.start_video)
        self.load_video_list()

        ref_box = QVBoxLayout()
        ref_box.addWidget(self.ref_view)
        ref_box.addWidget(self.ref_status)
        ref_box.addStretch()
        my_box = QVBoxLayout()
        my_box.addWidget(self.my_view)
        my_box.addWidget(self.my_status)
        my_box.addStretch()
        list_box = QVBoxLayout()
        list_box.addWidget(QLabel("카메라"))
        list_box.addWidget(self.camera_combo)
        list_box.addWidget(QLabel(f"스트레칭 영상 ({self.stretch_dir})"))
        list_box.addWidget(self.video_list)

        layout = QHBoxLayout(self)
        layout.addLayout(ref_box)
        layout.addLayout(my_box)
        layout.addLayout(list_box)

    def _make_view(self, text):
        view = QLabel(text)
        view.setFixedSize(VIEW_W, VIEW_H)
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view.setStyleSheet("background-color: black; color: gray;")
        return view

    def load_video_list(self):
        self.video_list.clear()
        for mp4 in sorted(self.stretch_dir.glob('*.mp4')):
            item = QListWidgetItem(mp4.stem)
            item.setData(Qt.ItemDataRole.UserRole, str(mp4.resolve()))
            self.video_list.addItem(item)
        if self.video_list.count() == 0:
            self.ref_status.setText(f"{self.stretch_dir}에 mp4 없음")

    def start_video(self, item):
        mp4 = Path(item.data(Qt.ItemDataRole.UserRole))
        json_path = mp4.with_name(f"{mp4.stem}_skeleton.json")
        if not json_path.exists():
            self.ref_status.setText(f"스켈레톤 JSON 없음: {json_path.name}")
            return

        camera_id = self.camera_combo.currentText()
        if not self.link.send({'cmd': 'start_stretching', 'camera_id': camera_id,
                               'reference': str(json_path)}):
            self.ref_status.setText("main_server 미연결 — main_server를 먼저 실행하세요")
            return

        if self.session is not None:
            self.session['cap'].release()
        cap = cv2.VideoCapture(str(mp4))
        self.session = {
            'camera_id': camera_id,
            'name': mp4.stem,
            'cap': cap,
            'total': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'ref_skeletons': load_reference(json_path),
            'ref_pos': 0,            # 다음에 read()하면 나올 프레임 번호
            'ref_frame': None,       # 마지막으로 읽은 기준 프레임
            'ended': False,
        }
        self._show_reference(0)      # 결과 오기 전에 첫 프레임 표시
        self.my_status.setText(f"{camera_id} 결과 대기 중...")

    def _read_reference(self, idx):
        """기준 영상에서 idx번 프레임 읽기 (앞으로만 이동, 같은 번호면 재사용)"""
        s = self.session
        if idx < s['ref_pos'] and s['ref_frame'] is not None:
            return s['ref_frame']
        while s['ref_pos'] < idx:
            if not s['cap'].grab():
                return None
            s['ref_pos'] += 1
        ok, frame = s['cap'].read()
        if not ok:
            return None
        s['ref_pos'] += 1
        s['ref_frame'] = frame
        return frame

    def _show_reference(self, idx):
        frame = self._read_reference(idx)
        if frame is None:
            self.session['ended'] = True
            self.ref_status.setText(f"{self.session['name']} 종료 ({self.session['total']} frames)")
            return
        frame = frame.copy()
        ref = self.session['ref_skeletons'].get(idx)
        if ref is not None:
            draw_skeleton(frame, ref['keypoints'], default=(255, 255, 0))
        self.ref_view.setPixmap(to_pixmap(fit_to_view(frame)))
        self.ref_status.setText(f"{self.session['name']} | frame {idx} / {self.session['total']}"
                                + ("" if ref is not None else " | 기준 스켈레톤 없음"))

    def on_result(self, camera_id, msg, frame):
        """main_server 결과 + 같은 frame_idx의 내 영상이 매칭됐을 때 호출"""
        s = self.session
        if s is None or s['ended'] or camera_id != s['camera_id'] or msg.get('mode') != MODE_STRETCH:
            return

        idx = msg.get('frame_idx')
        self._show_reference(idx)

        data = msg.get('data') or {}
        tracking = data.get('tracking_data')
        if tracking is None:
            summary = "기준 자세 비교 결과 없음"
        elif not tracking:
            summary = "사람 없음 또는 기준 프레임 없음"
        else:
            texts = []
            for track_id, info in tracking.items():
                limb_colors = {k: LEVEL_COLOR.get(v['level'], UNKNOWN_COLOR) for k, v in info.get('limbs', {}).items()}
                joint_colors = {k: LEVEL_COLOR.get(v['level'], UNKNOWN_COLOR) for k, v in info.get('joint_accuracy', {}).items()}
                if info.get('keypoints_px'):
                    draw_skeleton(frame, info['keypoints_px'], limb_colors, joint_colors)
                overall = info['overall']
                color = LEVEL_COLOR.get(overall['level'], UNKNOWN_COLOR)
                label = LEVEL_TEXT.get(overall['level'], overall['level'])
                x1, y1 = int(info['bbox'][0]), int(info['bbox'][1])
                cv2.putText(frame, f"{label} {overall['score']:.0f}",
                            (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                limbs = ", ".join(f"{k} {v['score']:.0f}" for k, v in info.get('limbs', {}).items())
                texts.append(f"ID {track_id}: {label} {overall['score']:.0f}점 ({limbs})")
            summary = " / ".join(texts)

        self.my_view.setPixmap(to_pixmap(fit_to_view(frame)))
        self.my_status.setText(f"{camera_id} | frame {idx} | {summary}")


# ============ 화면 ============
class CameraView(QWidget):
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id

        self.video = QLabel(f"{camera_id}\nNO SIGNAL")
        self.video.setFixedSize(VIEW_W, VIEW_H)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet("background-color: black; color: gray;")

        self.status = QLabel(camera_id)
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.video)
        layout.addWidget(self.status)

    def show_frame(self, frame_bgr):
        self.video.setPixmap(to_pixmap(fit_to_view(frame_bgr)))

    def set_status(self, text):
        self.status.setText(text)


class AdminWindow(QMainWindow):
    def __init__(self, stretch_dir='stretching'):
        super().__init__()
        self.setWindowTitle("SilverCare Admin GUI")

        self.store = FrameStore()
        self.link = MainLink()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.last_result_time = {cam_id: 0.0 for cam_id in CAMERA_PORTS}

        # 카메라 화면 (3열 그리드)
        self.views = {}
        grid = QGridLayout()
        for i, cam_id in enumerate(CAMERA_PORTS):
            view = CameraView(cam_id)
            self.views[cam_id] = view
            grid.addWidget(view, i // 3, i % 3)

        # 이벤트 로그
        self.event_log = QListWidget()
        self.event_log.setFixedWidth(320)
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("이벤트"))
        log_layout.addWidget(self.event_log)

        root = QHBoxLayout()
        root.addLayout(grid)
        root.addLayout(log_layout)
        monitor_tab = QWidget()
        monitor_tab.setLayout(root)

        self.stretch_tab = StretchingTab(self.link, stretch_dir)

        tabs = QTabWidget()
        tabs.addTab(monitor_tab, "모니터링")
        tabs.addTab(self.stretch_tab, "스트레칭")
        self.setCentralWidget(tabs)

        threading.Thread(target=video_receiver, args=(self.store, self.stop_event), daemon=True).start()
        threading.Thread(target=result_receiver, args=(self.result_queue, self.stop_event, self.link), daemon=True).start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_views)
        self.timer.start(UPDATE_INTERVAL_MS)

    def update_views(self):
        # 쌓인 결과 전부 꺼내기: 이벤트는 전부 처리, 그리기는 카메라별 최신 결과 하나만
        latest = {}
        while True:
            try:
                msg = self.result_queue.get_nowait()
            except queue.Empty:
                break
            camera_id = msg.get('camera_id')
            if camera_id not in self.views:
                continue
            latest[camera_id] = msg
            self.handle_events(camera_id, msg)

        now = time.monotonic()
        for camera_id, msg in latest.items():
            self.last_result_time[camera_id] = now
            self.render_result(camera_id, msg)

        for camera_id in self.views:
            if now - self.last_result_time[camera_id] > NO_RESULT_TIMEOUT_SEC:
                self.render_raw(camera_id)

    def render_result(self, camera_id, msg):
        view = self.views[camera_id]
        frame_idx = msg.get('frame_idx')
        mode = msg.get('mode')
        latest_idx = self.store.get_latest_idx(camera_id)
        lag = latest_idx - frame_idx if (latest_idx is not None and frame_idx is not None) else '-'

        jpg_bytes = self.store.pop(camera_id, frame_idx)
        if jpg_bytes is None:
            view.set_status(f"{camera_id} | frame {frame_idx} 영상 없음 (만료 또는 유실)")
            return
        frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        # 스트레칭 탭에도 같은 프레임 전달 (그리기 전 원본 복사본)
        try:
            self.stretch_tab.on_result(camera_id, msg, frame.copy())
        except (KeyError, TypeError, ValueError) as e:
            self.stretch_tab.my_status.setText(f"그리기 오류: {e}")

        drawer = DRAWERS.get(mode)
        try:
            summary = drawer(frame, msg.get('data') or {}) if drawer else ''
        except (KeyError, TypeError, ValueError) as e:
            summary = f"그리기 오류: {e}"

        view.show_frame(frame)
        view.set_status(f"{camera_id} | {MODE_NAME.get(mode, mode)} | frame {frame_idx} (지연 {lag}f) | {summary}")

    def render_raw(self, camera_id):
        frame_idx, jpg_bytes = self.store.latest(camera_id)
        if jpg_bytes is None:
            return
        frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        cv2.putText(frame, "NO AI RESULT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        self.views[camera_id].show_frame(frame)
        self.views[camera_id].set_status(f"{camera_id} | frame {frame_idx} | AI 결과 없음 (영상만 수신 중)")

    def handle_events(self, camera_id, msg):
        if msg.get('mode') != 0:
            return
        for event in (msg.get('data') or {}).get('events', []):
            text = f"{time.strftime('%H:%M:%S')}  {camera_id}  ID {event.get('track_id')}  {event.get('event')}"
            item = QListWidgetItem(text)
            if 'ALERT' in str(event.get('event')):
                item.setForeground(QColor('red'))
            self.event_log.insertItem(0, item)

    def closeEvent(self, event):
        self.stop_event.set()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stretch-dir', default='stretching', help='스트레칭 영상/스켈레톤 JSON 폴더')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = AdminWindow(stretch_dir=args.stretch_dir)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
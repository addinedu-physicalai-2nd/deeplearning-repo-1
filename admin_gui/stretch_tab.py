# admin_gui/stretch_tab.py
"""스트레칭 탭.

실제 `stretch_dir`(예: ~/stretching) 안에는 목/어깨/허리 같은 3개짜리 코스가
아니라, "01_준비운동.mp4" ~ "23_숨고르기.mp4" 식으로 번호가 매겨진 스트레칭
루틴 영상 23개가 각자 짝이 되는 "<같은 이름>_skeleton.json"과 함께 들어있다
(예: 05_목운동.mp4 ↔ 05_목운동_skeleton.json). 그래서 코스 목록을 코드에
하드코딩하지 않고 `list_courses()`가 그 폴더를 실제로 스캔해서 만든다 —
나중에 영상이 추가/삭제돼도 코드를 안 건드려도 된다.

화면 구성(보행 탭과 통일):
  - 좌측: 환자(=카메라) 검색 + 선택 목록 (보행 탭과 동일하게 좌측 사이드바)
  - 우측: 기준 동작 영상 + 실시간 카메라 영상을 크게 나란히 보여주고,
    그 아래에 스트레칭 코스 선택 목록 + 시작 버튼을 둔다.

종합 판정(1~5단계) 패널은 따로 두지 않는다 — 실시간으로 우측 영상 자체에
스켈레톤 색 + 작은 뱃지로 바로 찍어버리면 되므로 별도 패널은 불필요
(draw_stretch_badge, drawing.py).

동작:
  1) 환자(카메라)와 스트레칭 코스를 고른 뒤 "시작"을 누르면
  2) 좌측 영상에 그 코스의 기준 동작 영상(사람이 스트레칭하는 시범 영상)이 반복 재생된다
     — 환자는 이 화면을 보면서 따라한다.
  3) main_server에 start_stretching 명령(기준 자세 JSON 절대경로 포함)을 보낸다.
  4) 우측 영상에는 그 환자의 실시간 카메라 영상 + 스켈레톤 + 판정 뱃지를 보여준다.
"""
import os
import re

import cv2

from . import patients as patients_module
from .drawing import (LEVEL_COLOR, UNKNOWN_COLOR, draw_skeleton,
                       draw_stretch_badge, fit_to_view, to_pixmap)
from .qt_compat import (
    Qt, QFont, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QTimer, QVBoxLayout, QWidget,
)

SKELETON_SUFFIX = '_skeleton.json'

# 보행 탭(480x360)보다 크게 — 좌우 영상이 화면의 실질적인 주인공이므로.
VIEW_W, VIEW_H = 560, 420


def list_courses(stretch_dir):
    """stretch_dir 안의 "<이름>_skeleton.json" + "<이름>.mp4" 짝을 스캔해서
    코스 목록을 만든다. 파일명 앞의 번호(01, 02, ...)로 정렬한다."""
    courses = []
    if not os.path.isdir(stretch_dir):
        return courses
    for fname in os.listdir(stretch_dir):
        if not fname.endswith(SKELETON_SUFFIX):
            continue
        base = fname[:-len(SKELETON_SUFFIX)]
        video_fname = base + '.mp4'
        if not os.path.exists(os.path.join(stretch_dir, video_fname)):
            continue   # 짝이 되는 영상이 없으면 목록에서 제외
        m = re.match(r'^(\d+)_(.+)$', base)
        order, name = (int(m.group(1)), m.group(2)) if m else (999, base)
        courses.append({
            'course_id': base,
            'order': order,
            'label': name.replace('_', ' '),
            'video_filename': video_fname,
            'skeleton_filename': fname,
        })
    courses.sort(key=lambda c: c['order'])
    return courses


class StretchingTab(QWidget):
    def __init__(self, link, stretch_dir='stretching'):
        super().__init__()
        self.link = link
        self.stretch_dir = stretch_dir
        self.active_camera_id = None
        self.ref_cap = None
        self.ref_timer = QTimer()
        self.ref_timer.timeout.connect(self._next_ref_frame)

        self.courses = list_courses(stretch_dir)
        self.patients = patients_module.get_patients()

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # 좌측: 환자(카메라) 검색 + 선택 — 보행 탭과 동일한 사이드바 구성
        left = QVBoxLayout()
        left.setSpacing(8)
        left_title = QLabel("환자 선택")
        left_title.setFont(QFont('', -1, QFont.Weight.Bold))
        left.addWidget(left_title)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("환자 검색")
        self.search_input.textChanged.connect(self._filter_patients)
        left.addWidget(self.search_input)

        self.patient_list = QListWidget()
        left.addWidget(self.patient_list)
        self._populate_patient_list(self.patients)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(250)
        root.addWidget(left_widget)

        # 우측: 영상 두 개(크게) + 그 아래 스트레칭 코스 선택 + 시작 버튼
        right = QVBoxLayout()
        right.setSpacing(12)

        views = QHBoxLayout()
        views.setSpacing(16)

        ref_col = QVBoxLayout()
        self.ref_title = QLabel("기준 동작")
        self.ref_title.setFont(QFont('', -1, QFont.Weight.Bold))
        ref_col.addWidget(self.ref_title)
        self.ref_view = QLabel()
        self.ref_view.setFixedSize(VIEW_W, VIEW_H)
        self.ref_view.setStyleSheet("background:#0d0d10; border-radius:10px;")
        self.ref_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ref_col.addWidget(self.ref_view)
        views.addLayout(ref_col)

        my_col = QVBoxLayout()
        self.my_title = QLabel("실시간 카메라")
        self.my_title.setFont(QFont('', -1, QFont.Weight.Bold))
        my_col.addWidget(self.my_title)
        self.my_view = QLabel()
        self.my_view.setFixedSize(VIEW_W, VIEW_H)
        self.my_view.setStyleSheet("background:#0d0d10; border-radius:10px;")
        self.my_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        my_col.addWidget(self.my_view)
        views.addLayout(my_col)

        right.addLayout(views)

        course_title = QLabel("스트레칭 선택")
        course_title.setFont(QFont('', -1, QFont.Weight.Bold))
        right.addWidget(course_title)
        self.course_list = QListWidget()
        self.course_list.setMaximumHeight(150)
        if not self.courses:
            placeholder = QListWidgetItem(f"'{stretch_dir}' 폴더에서 영상을 찾지 못했습니다")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.course_list.addItem(placeholder)
        for course in self.courses:
            item = QListWidgetItem(course['label'])
            item.setData(Qt.ItemDataRole.UserRole, course['course_id'])
            self.course_list.addItem(item)
        right.addWidget(self.course_list)

        self.start_btn = QPushButton("시작")
        self.start_btn.clicked.connect(self.start_video)
        right.addWidget(self.start_btn)

        root.addLayout(right, stretch=1)

    def _populate_patient_list(self, patients):
        self.patient_list.clear()
        for p in patients:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.patient_list.addItem(item)

    def _filter_patients(self, text):
        text = text.strip()
        filtered = [p for p in self.patients if text in p.name] if text else self.patients
        self._populate_patient_list(filtered)

    def set_patients(self, patients):
        """main_server에서 환자 목록이 도착하면 app.py가 호출 (검색어는 유지)"""
        self.patients = patients
        self._filter_patients(self.search_input.text())

    def on_session_end(self, msg):
        """main_server가 스트레칭 세션을 끝내면 호출 (기준 영상 끝까지 → 평균 점수 DB 저장)"""
        if msg.get('camera_id') != self.active_camera_id:
            return
        avg = msg.get('avg_score')
        if msg.get('completed') and avg is not None:
            saved = "저장됨" if msg.get('saved') else "저장 실패"
            self.my_title.setText(f"완료 — {msg.get('course_name')} 평균 {avg:.0f}점 ({saved})")

    def _course_by_id(self, course_id):
        for course in self.courses:
            if course['course_id'] == course_id:
                return course
        return None

    def start_video(self):
        patient_item = self.patient_list.currentItem()
        course_item = self.course_list.currentItem()
        if patient_item is None or course_item is None:
            return
        patient = patient_item.data(Qt.ItemDataRole.UserRole)
        course = self._course_by_id(course_item.data(Qt.ItemDataRole.UserRole))
        if course is None:
            return

        self.active_camera_id = patient.camera_id
        self.ref_title.setText(f"기준 동작 — {course['label']}")
        self.my_title.setText(f"실시간 카메라 — {patient.camera_id} · {patient.name}님")

        video_path = os.path.abspath(os.path.join(self.stretch_dir, course['video_filename']))
        skeleton_path = os.path.abspath(os.path.join(self.stretch_dir, course['skeleton_filename']))
        self._start_reference_player(video_path)

        self.link.send({
            "cmd": "start_stretching",
            "camera_id": patient.camera_id,
            "patient_id": patient.patient_id,
            "reference": skeleton_path,
        })

    def _start_reference_player(self, video_path):
        """좌측 '기준 동작' 영상을 반복 재생 (환자가 보고 따라할 실제 시범 영상)"""
        self.ref_timer.stop()
        if self.ref_cap is not None:
            self.ref_cap.release()
        self.ref_cap = cv2.VideoCapture(video_path)
        if not self.ref_cap.isOpened():
            print(f"[StretchTab] 기준 동작 영상 열기 실패: {video_path}")
            return
        fps = self.ref_cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.ref_timer.start(max(20, int(1000 / fps)))

    def _next_ref_frame(self):
        if self.ref_cap is None:
            return
        ok, frame = self.ref_cap.read()
        if not ok:
            self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 끝나면 처음부터 반복 재생
            ok, frame = self.ref_cap.read()
        if ok:
            self.ref_view.setPixmap(to_pixmap(fit_to_view(frame, VIEW_W, VIEW_H)))

    def stop(self):
        """앱 종료 시 정리용 (app.py의 closeEvent에서 호출)"""
        self.ref_timer.stop()
        if self.ref_cap is not None:
            self.ref_cap.release()
            self.ref_cap = None

    def on_result(self, camera_id, msg, frame):
        """result_receiver가 mode 2 메시지를 줄 때마다 app.py의 dispatch()가 호출.
        frame은 app.py가 이미 FrameStore에서 꺼내 cv2.imdecode까지 해둔 것 —
        우측 '실시간 카메라' 화면이 바로 이 프레임이다."""
        if camera_id != self.active_camera_id:
            return
        data = msg.get('data', {})
        tracking = data.get('tracking_data', {})

        if not tracking:
            view = fit_to_view(frame, VIEW_W, VIEW_H)
            draw_stretch_badge(view, None, 0)
            self.my_view.setPixmap(to_pixmap(view))
            return

        # 화면에는 한 명만 표시 (첫 번째 track)
        track_id, info = next(iter(tracking.items()))
        overall = info.get('overall', {})
        level = overall.get('level')          # 1(안 맞음) ~ 5(잘 맞음) — 부위별 세부 없음
        score = overall.get('score', 0)

        # fit_to_view 이후의 keypoints는 원본 해상도 좌표라 좌표가 안 맞을 수 있으므로,
        # 스켈레톤은 원본 frame에 먼저 그리고 그 다음에 뷰 크기로 맞춘다.
        draw_skeleton(frame, info.get('keypoints_px', []), default=LEVEL_COLOR.get(level, UNKNOWN_COLOR))
        view = fit_to_view(frame, VIEW_W, VIEW_H)
        draw_stretch_badge(view, level, score)
        self.my_view.setPixmap(to_pixmap(view))
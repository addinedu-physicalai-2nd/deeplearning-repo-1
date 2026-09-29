# admin_gui/stretch_tab.py
"""스트레칭 탭.

실제 `stretch_dir`(예: ~/stretching) 안에는 목/어깨/허리 같은 3개짜리 코스가
아니라, "01_준비운동.mp4" ~ "23_숨고르기.mp4" 식으로 번호가 매겨진 스트레칭
루틴 영상 23개가 각자 짝이 되는 "<같은 이름>_skeleton.json"과 함께 들어있다
(예: 05_목운동.mp4 ↔ 05_목운동_skeleton.json). 그래서 코스 목록을 코드에
하드코딩하지 않고 `list_courses()`가 그 폴더를 실제로 스캔해서 만든다 —
나중에 영상이 추가/삭제돼도 코드를 안 건드려도 된다.

어떤 환자(=카메라)가 스트레칭을 하는지는 코스마다 고정된 게 아니라 매번
골라야 하므로, 보행 탭과 마찬가지로 `patients.py`의 더미 환자 목록에서
선택한다(진짜 DB가 붙으면 `patients.get_patients()` 내부만 바뀌고 여기는
그대로 쓰면 됨).

동작:
  1) 환자(카메라)와 스트레칭 코스를 고른 뒤 "시작"을 누르면
  2) 좌측에 그 코스의 기준 동작 영상(사람이 스트레칭하는 시범 영상)이 반복 재생된다
     — 환자는 이 화면을 보면서 따라한다.
  3) main_server에 start_stretching 명령(기준 자세 JSON 절대경로 포함)을 보낸다.
  4) 우측에는 그 환자의 실시간 카메라 영상 + 스켈레톤을 보여준다.

부위별로 "어디가 얼마나 틀어졌는지" 세세하게 나오는 게 아니라, 전체적으로
기준 자세와 얼마나 맞는지를 1~5단계로만 판정해서 색으로 보여준다
(1단계=가장 안 맞음/빨강 ~ 5단계=가장 잘 맞음/초록). 그래서 스켈레톤 전체가
그 단계 색 하나로 칠해지고, 우측에 단계 배지 + 점수만 표시한다. 반복 횟수는
세지 않는다(StretchingAnalyzer가 판단, 단계 기준값은 GUI가 아니라
main_server 쪽에만 있음).
"""
import os
import re

import cv2

from . import patients as patients_module
from .drawing import LEVEL_COLOR, LEVEL_HEX, UNKNOWN_COLOR, UNKNOWN_HEX, VIEW_W, VIEW_H, draw_skeleton, fit_to_view, to_pixmap
from .qt_compat import (
    Qt, QFont, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QTimer, QVBoxLayout, QWidget,
)

SKELETON_SUFFIX = '_skeleton.json'


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

        # 좌측: 환자(카메라) 선택 + 코스 목록 + 시작 버튼
        left = QVBoxLayout()
        left.addWidget(QLabel("환자 선택"))
        self.patient_list = QListWidget()
        for p in self.patients:
            item = QListWidgetItem(f"{p.name} · {p.room} ({p.camera_id})")
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.patient_list.addItem(item)
        self.patient_list.setMaximumHeight(120)
        left.addWidget(self.patient_list)

        left.addWidget(QLabel("스트레칭 선택"))
        self.course_list = QListWidget()
        if not self.courses:
            placeholder = QListWidgetItem(f"'{stretch_dir}' 폴더에서 영상을 찾지 못했습니다")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.course_list.addItem(placeholder)
        for course in self.courses:
            item = QListWidgetItem(course['label'])
            item.setData(Qt.ItemDataRole.UserRole, course['course_id'])
            self.course_list.addItem(item)
        left.addWidget(self.course_list)

        self.start_btn = QPushButton("시작")
        self.start_btn.clicked.connect(self.start_video)
        left.addWidget(self.start_btn)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(240)
        root.addWidget(left_widget)

        # 중앙: 기준 동작 영상(반복 재생) / 실시간 카메라
        views = QHBoxLayout()
        ref_col = QVBoxLayout()
        self.ref_title = QLabel("기준 동작")
        self.ref_title.setFont(QFont('', -1, QFont.Weight.Bold))
        ref_col.addWidget(self.ref_title)
        self.ref_view = QLabel()
        self.ref_view.setFixedSize(VIEW_W, VIEW_H)
        self.ref_view.setStyleSheet("background:#111; border-radius:6px;")
        self.ref_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ref_col.addWidget(self.ref_view)
        views.addLayout(ref_col)

        my_col = QVBoxLayout()
        self.my_title = QLabel("실시간 카메라")
        self.my_title.setFont(QFont('', -1, QFont.Weight.Bold))
        my_col.addWidget(self.my_title)
        self.my_view = QLabel()
        self.my_view.setFixedSize(VIEW_W, VIEW_H)
        self.my_view.setStyleSheet("background:#111; border-radius:6px;")
        self.my_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        my_col.addWidget(self.my_view)
        views.addLayout(my_col)
        root.addLayout(views)

        # 우측: 종합 판정(1~5단계, 부위별 세부 항목 없음)
        right = QVBoxLayout()
        right.addWidget(QLabel("종합 판정"))

        self.level_badge = QLabel("-")
        self.level_badge.setFixedSize(90, 90)
        self.level_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.level_badge.setFont(QFont('', 26, QFont.Weight.Bold))
        self._set_level_style(None)
        right.addWidget(self.level_badge)

        self.score_label = QLabel("대기 중")
        right.addWidget(self.score_label)
        right.addStretch()
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setFixedWidth(160)
        root.addWidget(right_widget)

    def _set_level_style(self, level):
        color = LEVEL_HEX.get(level, UNKNOWN_HEX)
        self.level_badge.setStyleSheet(
            f"background:{color}; color:white; border-radius:45px;")
        self.level_badge.setText(f"{level}" if level else "-")

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
            self.ref_view.setPixmap(to_pixmap(fit_to_view(frame)))

    def stop(self):
        """앱 종료 시 정리용 (admin_gui.py의 closeEvent에서 호출)"""
        self.ref_timer.stop()
        if self.ref_cap is not None:
            self.ref_cap.release()
            self.ref_cap = None

    def on_result(self, camera_id, msg, frame):
        """result_receiver가 mode 2 메시지를 줄 때마다 admin_gui.py의 dispatch()가 호출.
        frame은 admin_gui.py가 이미 FrameStore에서 꺼내 cv2.imdecode까지 해둔 것 —
        우측 '실시간 카메라' 화면이 바로 이 프레임이다."""
        if camera_id != self.active_camera_id:
            return
        data = msg.get('data', {})
        tracking = data.get('tracking_data', {})
        if not tracking:
            self.my_view.setPixmap(to_pixmap(fit_to_view(frame)))
            self.score_label.setText("추적된 사람 없음")
            self._set_level_style(None)
            return

        # 화면에는 한 명만 표시 (첫 번째 track)
        track_id, info = next(iter(tracking.items()))
        overall = info.get('overall', {})
        level = overall.get('level')          # 1(안 맞음) ~ 5(잘 맞음) — 부위별 세부 없음
        score = overall.get('score', 0)

        # 스켈레톤 전체를 그 단계 색 하나로 그린다 (부위별로 다른 색을 주지 않음)
        draw_skeleton(frame, info.get('keypoints_px', []), default=LEVEL_COLOR.get(level, UNKNOWN_COLOR))
        self.my_view.setPixmap(to_pixmap(fit_to_view(frame)))

        self._set_level_style(level)
        self.score_label.setText(f"{score:.0f}점")

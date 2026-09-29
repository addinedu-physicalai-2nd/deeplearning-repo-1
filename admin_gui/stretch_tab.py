# admin_gui/stretch_tab.py
"""스트레칭 탭 — 코스(목/어깨/허리) 선택 → 시작 시:
  1) 좌측에 "기준 동작" 영상(사람이 스트레칭하는 시범 영상)을 반복 재생한다.
     환자는 이 화면을 보면서 따라한다.
  2) main_server에 start_stretching 명령(기준 자세 JSON 절대경로 포함)을 보낸다.
  3) 우측에는 그 환자의 실시간 카메라 영상 + 스켈레톤을 보여준다.

부위별로 "어디가 얼마나 틀어졌는지" 세세하게 나오는 게 아니라, 전체적으로
기준 자세와 얼마나 맞는지를 1~5단계로만 판정해서 색으로 보여준다
(1단계=가장 안 맞음/빨강 ~ 5단계=가장 잘 맞음/초록). 그래서 목/왼쪽 어깨/
오른쪽 어깨 같은 부위별 항목은 없고, 스켈레톤 전체가 그 단계 색 하나로
칠해지고 우측에 단계 배지 + 점수만 표시한다. 반복 횟수(rep count)도 세지
않는다(StretchingAnalyzer가 판단, 단계 기준값은 GUI가 아니라 main_server
쪽에만 있음).

TODO(DB 연동 담당자): COURSES 아래 각 코스의 patient/camera_id는 더미값입니다.
실제로는 "이 스트레칭을 할 환자 ↔ 그 환자를 찍는 카메라"가 낙상 탭과 마찬가지로
1:1로 고정되어 있어야 하므로, DB 스키마가 나오면 이 매핑을 실제 조회로 교체해야
합니다. 기준 동작 영상(.mp4)과 기준 자세 JSON의 파일명 규칙도 팀 컨벤션에 맞춰
조정해주세요(현재는 <course>_stretch.mp4 / <course>_skeleton.json으로 가정).
"""
import os

from .qt_compat import (
    Qt, QComboBox, QFont, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QTimer, QVBoxLayout, QWidget,
)
from .drawing import LEVEL_COLOR, LEVEL_HEX, UNKNOWN_COLOR, UNKNOWN_HEX, VIEW_W, VIEW_H, draw_skeleton, fit_to_view, to_pixmap

import cv2

# TODO(DB 연동 담당자): 더미 매핑 — 실제 병상-환자-카메라 배정 테이블로 교체 필요
COURSES = [
    ('neck', '목 스트레칭', '김철수', 'CAM-01', 'neck_stretch.mp4', 'neck_skeleton.json'),
    ('shoulder', '어깨 스트레칭', '홍길동', 'CAM-02', 'shoulder_stretch.mp4', 'shoulder_skeleton.json'),
    ('waist', '허리 스트레칭', '박영자', 'CAM-03', 'waist_stretch.mp4', 'waist_skeleton.json'),
]


class StretchingTab(QWidget):
    def __init__(self, link, stretch_dir='stretching'):
        super().__init__()
        self.link = link
        self.stretch_dir = stretch_dir
        self.active_camera_id = None
        self.ref_cap = None
        self.ref_timer = QTimer()
        self.ref_timer.timeout.connect(self._next_ref_frame)

        root = QHBoxLayout(self)

        # 좌측: 코스 목록 + 시작 버튼
        left = QVBoxLayout()
        left.addWidget(QLabel("스트레칭 선택"))
        self.course_list = QListWidget()
        for course_id, label, patient, camera_id, _, _ in COURSES:
            item = QListWidgetItem(f"{label} — {patient}님")
            item.setData(Qt.ItemDataRole.UserRole, course_id)
            self.course_list.addItem(item)
        left.addWidget(self.course_list)

        self.start_btn = QPushButton("시작")
        self.start_btn.clicked.connect(self.start_video)
        left.addWidget(self.start_btn)
        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(220)
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
        for row in COURSES:
            if row[0] == course_id:
                return row
        return None

    def start_video(self):
        course_item = self.course_list.currentItem()
        if course_item is None:
            return
        course_id = course_item.data(Qt.ItemDataRole.UserRole)
        _, label, patient, camera_id, video_filename, skeleton_filename = self._course_by_id(course_id)

        self.active_camera_id = camera_id
        self.ref_title.setText(f"기준 동작 — {label}")
        self.my_title.setText(f"실시간 카메라 — {camera_id} · {patient}님")

        video_path = os.path.abspath(os.path.join(self.stretch_dir, video_filename))
        skeleton_path = os.path.abspath(os.path.join(self.stretch_dir, skeleton_filename))
        self._start_reference_player(video_path)

        self.link.send({
            "cmd": "start_stretching",
            "camera_id": camera_id,
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

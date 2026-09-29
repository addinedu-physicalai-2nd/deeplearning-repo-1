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

종합 판정(good/adjust/check 3단계) 패널은 따로 두지 않는다 — 실시간으로 우측
영상 자체에 스켈레톤 색 + 작은 뱃지로 바로 찍어버리면 되므로 별도 패널은 불필요
(draw_stretch_badge, drawing.py). 부위별(limbs)/관절별(joint_accuracy) 판정도
같은 3단계 값을 주므로, 스켈레톤은 부위마다 그 색으로 따로 그린다.

동작:
  1) 환자(카메라)와 스트레칭 코스를 고른 뒤 "시작"을 누르면
  2) 좌측 영상에 그 코스의 기준 동작 영상(사람이 스트레칭하는 시범 영상)이 재생된다
     — 환자는 이 화면을 보면서 따라한다. 영상이 끝나면 반복 재생하지 않고 마지막
     프레임에서 멈춘다.
  3) main_server에 start_stretching 명령(기준 자세 JSON 절대경로 포함)을 보낸다.
  4) 우측 영상에는 그 환자의 실시간 카메라 영상(거울처럼 좌우 반전) + 부위별 색이
     입혀진 스켈레톤 + 판정 뱃지를 보여준다.
  5) 좌측 기준 영상이 끝나는 순간, 서버 응답을 기다리지 않고 우측 화면을 즉시
     멈추고 검정 결과 화면(코스 이름 + 평균 점수)으로 바꾼다 (_finish_session 참고).
"""
import os
import re

import cv2

from .drawing import (LEVEL_COLOR, UNKNOWN_COLOR, draw_session_result,
                       draw_skeleton, draw_stretch_badge, fit_to_view, to_pixmap)
from config.settings import CAMERA_PORTS
from .qt_compat import (
    Qt, QComboBox, QFont, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
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


def _mirror_keypoints(keypoints, frame_width):
    """실시간 카메라 프레임을 cv2.flip(frame, 1)로 좌우 반전해서 보여줄 것이므로,
    그 위에 그릴 keypoints의 x좌표도 같은 기준(프레임 폭)으로 반전시켜야 스켈레톤이
    반전된 영상 위치와 어긋나지 않는다. YOLO가 못 찾은 점은 (0, 0)으로 오는데, 그걸
    그대로 반전하면 (width, 0)이 되어 "화면 오른쪽 끝에 보이는 점"으로 잘못
    인식되므로 (0, 0)은 반전하지 않고 그대로 둔다."""
    mirrored = []
    for p in keypoints:
        x, y = p[0], p[1]
        if x == 0 and y == 0:
            mirrored.append((0, 0))
        else:
            mirrored.append((frame_width - x, y))
    return mirrored


class StretchingTab(QWidget):
    def __init__(self, link, stretch_dir='stretching'):
        super().__init__()
        self.link = link
        self.stretch_dir = stretch_dir
        self.active_camera_id = None
        # 세션 종료 직후부터는 우측 화면을 검정 결과 화면으로 고정한다 — 그 뒤에
        # main_server가 뒤늦게 흘려보내는 mode=2 프레임이 있어도 on_result()가
        # 다시 덮어쓰지 않도록 이 플래그로 막는다. start_video()가 새로 누르면 해제.
        self.session_ended = False
        # 이번 세션 동안 받은 프레임별 점수 — 좌측 기준 영상이 끝나는 순간 서버 응답을
        # 기다리지 않고 이걸로 즉시 평균을 내서 우측 화면에 보여준다(동기화 목적).
        self.score_history = []
        self.active_course_label = None
        self.ref_cap = None
        self.ref_timer = QTimer()
        self.ref_timer.timeout.connect(self._next_ref_frame)

        self.courses = list_courses(stretch_dir)
        self.patients = []          # main_server 연결 후 app.py가 set_patients()로 채움

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        # 좌측: 환자(카메라) 검색 + 선택 — 보행 탭과 동일한 사이드바 구성
        left = QVBoxLayout()
        left.setSpacing(8)
        # 스트레칭은 공용 카메라에서 측정 → 환자와 카메라를 따로 고른다 (환자는 전체 목록)
        camera_title = QLabel("측정 카메라")
        camera_title.setFont(QFont('', -1, QFont.Weight.Bold))
        left.addWidget(camera_title)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(list(CAMERA_PORTS))
        left.addWidget(self.camera_combo)

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
        # 영상 블록과 "코스 선택" 블록 사이에는 이 정도 여유를 두고, 블록 내부
        # (라벨↔목록↔버튼) 간격은 course_section에서 훨씬 좁게 따로 준다 — 예전엔
        # 이 spacing(12)이 라벨-목록 사이에도 그대로 적용돼서 "스트레칭 선택" 글자와
        # 목록 사이가 필요 이상으로 넓어 보였다.
        right.setSpacing(16)

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
        # 결과(코스 이름/평균 점수)는 세션 종료 후 우측 화면 자체(검정 결과 화면)에
        # 이미 크게 찍히므로, 제목 줄에 "완료 — OO 평균 N점" 같은 걸 또 띄우지 않고
        # 라벨은 항상 "실시간 카메라"로 고정한다.
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

        # 라벨↔목록↔버튼을 촘촘하게 묶는 별도 레이아웃 — 위 right.setSpacing(16)과
        # 분리해서 이 블록만 spacing(6)으로 좁혀야 "스트레칭 선택" 글자와 바로
        # 아래 목록 사이 간격이 붙어 보인다.
        course_section = QVBoxLayout()
        course_section.setSpacing(6)

        course_title = QLabel("스트레칭 선택")
        course_title.setFont(QFont('', -1, QFont.Weight.Bold))
        course_section.addWidget(course_title)
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
        course_section.addWidget(self.course_list)

        self.start_btn = QPushButton("시작")
        self.start_btn.clicked.connect(self.start_video)
        course_section.addWidget(self.start_btn)

        right.addLayout(course_section)

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
        """main_server가 스트레칭 세션을 끝냈다고(DB 저장 포함) 알려줄 때 호출.

        검정 결과 화면으로 바꾸는 '트리거'는 이제 이 메서드가 아니라 좌측 기준
        영상이 끝나는 순간(_next_ref_frame)이다 — 서버가 이 메시지를 보내기까지
        약간의 왕복 지연이 있으면 "좌측 영상 끝남"과 "우측 화면이 검정으로 바뀜"
        사이에 눈에 보이는 시차(요청하신 싱크 안 맞는 문제)가 생기기 때문. 그래서
        이 메서드는 이미 멈춰있는 화면의 점수를, 서버가 계산한 더 정확한 평균으로
        다시 한 번 갱신만 해준다(로컬 프레임 평균은 근사값이라 서버 값이 오면
        그걸로 교체). 아직 영상이 안 끝났는데 서버가 먼저 종료를 알리는 경우에도
        대비해 _finish_session을 그대로 호출한다."""
        if msg.get('camera_id') != self.active_camera_id:
            return
        self._finish_session(msg.get('avg_score'))

    def _finish_session(self, avg_score=None):
        """우측 '실시간 카메라' 화면을 멈추고 검정 결과 화면으로 바꾼다.
        avg_score가 없으면(서버 응답 전이거나 없는 경우) 이번 세션 동안 로컬에서
        받은 프레임별 점수(self.score_history)의 평균으로 대신한다 — 그래야 서버
        응답을 기다리지 않고 좌측 영상이 끝나는 즉시 화면을 바꿀 수 있다."""
        self.session_ended = True
        if avg_score is None and self.score_history:
            avg_score = sum(self.score_history) / len(self.score_history)
        result_view = draw_session_result(VIEW_W, VIEW_H, avg_score, self.active_course_label)
        self.my_view.setPixmap(to_pixmap(result_view))

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

        camera_id = self.camera_combo.currentText()
        self.active_camera_id = camera_id
        self.session_ended = False   # 새로 시작 → 이전 세션의 검정 결과 화면 고정 해제
        self.score_history = []
        self.active_course_label = course['label']
        self.ref_title.setText(f"기준 동작 — {course['label']}")

        video_path = os.path.abspath(os.path.join(self.stretch_dir, course['video_filename']))
        skeleton_path = os.path.abspath(os.path.join(self.stretch_dir, course['skeleton_filename']))
        self._start_reference_player(video_path)

        self.link.send({
            "cmd": "start_stretching",
            "camera_id": camera_id,
            "patient_id": patient.patient_id,
            "reference": skeleton_path,
        })

    def _start_reference_player(self, video_path):
        """좌측 '기준 동작' 영상 재생 (환자가 보고 따라할 실제 시범 영상).
        끝까지 재생되면 반복하지 않고 마지막 프레임에서 멈춘다(_next_ref_frame 참고)."""
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
            # 기준 동작 영상은 반복 재생하지 않는다 — 끝나면 타이머만 멈추고
            # 마지막으로 표시했던 프레임을 화면에 그대로 남겨둔다. 그리고 바로 이
            # 순간이 "끝났다"는 기준이므로, 서버의 세션 종료 메시지를 기다리지 않고
            # 우측 화면도 지금 바로 멈춰서 검정 결과 화면으로 바꾼다 — 좌/우가
            # 항상 같은 타이밍에 끝나도록.
            self.ref_timer.stop()
            if not self.session_ended:
                self._finish_session()
            return
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
        우측 '실시간 카메라' 화면이 바로 이 프레임이다.

        실시간 카메라는 환자가 거울 보듯 자연스럽게 보도록 좌우 반전해서 보여준다.
        반전은 스켈레톤/뱃지를 그리기 전에 원본 프레임에 먼저 적용한다 — 그래야
        뱃지 점수 텍스트가 거꾸로 뒤집혀 나오지 않고, keypoints도 같은 기준(프레임
        폭)으로 같이 반전시켜서 스켈레톤이 반전된 영상과 어긋나지 않게 맞춘다."""
        if camera_id != self.active_camera_id:
            return
        if self.session_ended:
            # 세션이 이미 끝나서 검정 결과 화면을 띄워둔 상태 — 뒤늦게 도착하는
            # 프레임으로 다시 덮어쓰지 않는다. (새 시작을 누르면 start_video()에서 해제)
            return
        data = msg.get('data', {})
        tracking = data.get('tracking_data', {})

        frame = cv2.flip(frame, 1)   # 거울 모드 — 좌우 반전

        if not tracking:
            view = fit_to_view(frame, VIEW_W, VIEW_H)
            draw_stretch_badge(view, None, 0)
            self.my_view.setPixmap(to_pixmap(view))
            return

        # 화면에는 한 명만 표시 (첫 번째 track)
        track_id, info = next(iter(tracking.items()))
        overall = info.get('overall', {})
        level = overall.get('level')          # 'good'/'adjust'/'check' — overall 종합 판정
        score = overall.get('score', 0)
        self.score_history.append(score)   # 세션 종료 시 로컬 평균 계산용 (_finish_session)

        # limbs(부위별)/joint_accuracy(관절별)도 각각 같은 3단계 값을 주므로, 스켈레톤을
        # 한 가지 색이 아니라 부위마다 다른 색으로 그린다(안 맞는 부위만 빨갛게 보이게).
        frame_w = frame.shape[1]
        keypoints = _mirror_keypoints(info.get('keypoints_px', []), frame_w)
        limb_colors = {name: LEVEL_COLOR.get(v.get('level'), UNKNOWN_COLOR)
                       for name, v in info.get('limbs', {}).items()}
        joint_colors = {name: LEVEL_COLOR.get(v.get('level'), UNKNOWN_COLOR)
                        for name, v in info.get('joint_accuracy', {}).items()}

        # fit_to_view 이후의 keypoints는 원본 해상도 좌표라 좌표가 안 맞을 수 있으므로,
        # 스켈레톤은 원본 frame(반전 이미 적용됨)에 먼저 그리고 그 다음에 뷰 크기로 맞춘다.
        draw_skeleton(frame, keypoints, limb_colors, joint_colors,
                      default=LEVEL_COLOR.get(level, UNKNOWN_COLOR))
        view = fit_to_view(frame, VIEW_W, VIEW_H)
        draw_stretch_badge(view, level, score)
        self.my_view.setPixmap(to_pixmap(view))

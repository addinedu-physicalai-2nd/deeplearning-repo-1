# admin_gui/fall_tab.py
"""낙상 탭 — 목업(gui_mockup_v5.html)의 낙상 탭 동작을 그대로 구현.

카메라 2대(CAM-01=101호, CAM-02=102호) 고정. 상태에 따라 카메라 박스 테두리
색이 바뀌고(정상=투명, 주의=주황, 낙상=빨강), 낙상/주의 이벤트가 발생하면
화면 우측 상단에 알림 카드가 쌓인다. 낙상(빨강) 카드를 클릭하면 팝업이 뜬다. 카메라(=침상)는 환자와 1:1로 고정
매칭되어 있으므로 환자명은 이미 알고 있는 값을 그대로 보여주고, 요양보호사명만
입력하면 "처리 완료" 버튼이 활성화된다. 주의(주황) 카드는 클릭하면 바로 닫힌다
(목업과 동일).

resolve_fall 명령은 main_server 쪽에 아직 없음 (작업 지시서 기준) — 여기서는
전송 코드까지만 만들어두고, 화면은 로컬에서 낙관적으로 정상 상태로 되돌린다.
서버가 명령을 지원하게 되면 MainLink.send() 호출 자체는 그대로 쓰면 된다.
"""
from . import db_stub
from .drawing import fit_to_view, to_pixmap
from .qt_compat import (
    Qt, QColor, QFont, QDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget, pyqtSignal,
)

# 카메라(=침상) ↔ 환자 1:1 고정 매칭.
# TODO(DB 연동 담당자): 더미값 — 실제 병상-환자 배정 테이블 조회로 교체 필요.
FALL_CAMERAS = [('CAM-01', '101호', '김철수'), ('CAM-02', '102호', '홍길동')]

STATE_BORDER = {'normal': 'transparent', 'checking': '#c9820a', 'alert': '#e5484d'}
STATE_BADGE_TEXT = {'normal': '정상', 'checking': '확인 필요', 'alert': '낙상 감지'}
STATE_BADGE_BG = {'normal': '#2f9e44', 'checking': '#c9820a', 'alert': '#e5484d'}


class FallCameraBox(QWidget):
    """카메라 1대 표시 박스 — 영상 + 방/환자 이름 + 상태 뱃지 + 상태별 테두리 색"""

    def __init__(self, camera_id, room_label, patient_name):
        super().__init__()
        self.camera_id = camera_id
        self.room_label = room_label
        self.patient_name = patient_name
        self.state = 'normal'

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QHBoxLayout()
        name_label = QLabel(f"{room_label} · {patient_name}님")
        name_label.setFont(QFont('', -1, QFont.Weight.Bold))
        self.badge = QLabel(STATE_BADGE_TEXT['normal'])
        self.badge.setStyleSheet(
            f"background:{STATE_BADGE_BG['normal']}; color:white; border-radius:8px; padding:2px 8px;")
        header.addWidget(name_label)
        header.addStretch()
        header.addWidget(self.badge)
        layout.addLayout(header)

        self.video_label = QLabel()
        self.video_label.setFixedSize(480, 360)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background:#111; border-radius:6px;")
        layout.addWidget(self.video_label)

        self.status_label = QLabel('-')
        self.status_label.setStyleSheet("color:#888;")
        layout.addWidget(self.status_label)

        self._apply_border()

    def _apply_border(self):
        color = STATE_BORDER[self.state]
        self.setStyleSheet(f"FallCameraBox {{ border: 3px solid {color}; border-radius: 10px; }}")

    def set_state(self, state, status_text=None):
        if state not in STATE_BORDER:
            return
        self.state = state
        self._apply_border()
        self.badge.setText(STATE_BADGE_TEXT[state])
        self.badge.setStyleSheet(
            f"background:{STATE_BADGE_BG[state]}; color:white; border-radius:8px; padding:2px 8px;")
        if status_text is not None:
            self.status_label.setText(status_text)

    def show_frame(self, frame_bgr):
        pixmap = to_pixmap(fit_to_view(frame_bgr, 480, 360))
        self.video_label.setPixmap(pixmap)


class AlertCard(QFrame):
    """우측 상단에 쌓이는 알림 카드 (주황=주의/즉시 닫힘, 빨강=낙상/팝업 오픈)"""
    clicked = pyqtSignal()

    def __init__(self, room_label, urgent, message):
        super().__init__()
        self.urgent = urgent
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        border_color = '#e5484d' if urgent else '#c9820a'
        self.setStyleSheet(
            f"AlertCard {{ background: palette(base); border-left: 4px solid {border_color}; "
            f"border-radius: 6px; }}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        title = QLabel(f"{'낙상 감지' if urgent else '주의'} · {room_label}")
        title.setFont(QFont('', -1, QFont.Weight.Bold))
        body = QLabel(message)
        body.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(body)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def remove(self):
        self.setParent(None)
        self.deleteLater()


class ResolveFallDialog(QDialog):
    """낙상 카드 클릭 시 뜨는 팝업. 카메라(=침상)가 환자와 1:1로 고정 매칭되어
    있으므로 환자명은 이미 알고 있는 값을 그대로 보여주고(읽기 전용),
    요양보호사명만 입력받아 "처리 완료" 버튼을 활성화한다."""

    def __init__(self, room_label, patient_name, parent=None):
        super().__init__(parent)
        self.patient_name = patient_name
        self.setWindowTitle(f"낙상 처리 — {room_label}")
        layout = QVBoxLayout(self)

        form = QFormLayout()
        patient_label = QLabel(f"{patient_name}님")
        patient_label.setFont(QFont('', -1, QFont.Weight.Bold))
        form.addRow("환자", patient_label)
        self.caregiver_input = QLineEdit()
        form.addRow("요양보호사명", self.caregiver_input)
        layout.addLayout(form)

        self.confirm_btn = QPushButton("처리 완료")
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.accept)
        layout.addWidget(self.confirm_btn)

        self.caregiver_input.textChanged.connect(self._update_confirm_state)

    def _update_confirm_state(self):
        ready = bool(self.caregiver_input.text().strip())
        self.confirm_btn.setEnabled(ready)

    def values(self):
        return self.patient_name, self.caregiver_input.text().strip()


class FallTab(QWidget):
    def __init__(self, link):
        super().__init__()
        self.link = link
        self.boxes = {}
        self.cards = {}          # track_id -> AlertCard (urgent 카드만 추적, 처리 완료 시 제거)
        self._camera_lookup = {cam_id: (room, patient) for cam_id, room, patient in FALL_CAMERAS}

        root = QHBoxLayout(self)

        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        for i, (camera_id, room_label, patient_name) in enumerate(FALL_CAMERAS):
            box = FallCameraBox(camera_id, room_label, patient_name)
            self.boxes[camera_id] = box
            grid.addWidget(box, i // 2, i % 2)
        root.addWidget(grid_widget, stretch=3)

        self.notif_stack = QVBoxLayout()
        self.notif_stack.addStretch()
        notif_widget = QWidget()
        notif_widget.setLayout(self.notif_stack)
        notif_widget.setFixedWidth(280)
        root.addWidget(notif_widget, stretch=1)

    def render(self, camera_id, msg, frame):
        """result_receiver가 mode 0 메시지를 줄 때마다 admin_gui.py의 dispatch()가 호출"""
        box = self.boxes.get(camera_id)
        if box is None:
            return
        data = msg.get('data', {})
        box.show_frame(frame)
        camera_status = data.get('camera_status', '-')

        state = 'normal'
        for track in data.get('tracks', []):
            if track.get('color') == 'red':
                state = 'alert'
                break
            if track.get('color') == 'orange' and state == 'normal':
                state = 'checking'
        box.set_state(state, camera_status)

        room_label, patient_name = self._camera_lookup.get(camera_id, (camera_id, '-'))
        for event in data.get('events', []):
            track_id = event.get('track_id')
            name = event.get('event', '')
            db_stub.log_event(camera_id, track_id, name)
            if name.endswith('TO_ALERT'):
                self._add_alert_card(camera_id, room_label, patient_name, track_id, urgent=True,
                                      message="낙상이 감지되었습니다. 확인 후 처리해주세요.")
            elif name.endswith('TO_CHECKING'):
                self._add_alert_card(camera_id, room_label, patient_name, track_id, urgent=False,
                                      message="자세 확인이 필요합니다.")

    def _add_alert_card(self, camera_id, room_label, patient_name, track_id, urgent, message):
        card = AlertCard(f"{room_label} · {patient_name}님", urgent, message)
        if urgent:
            card.clicked.connect(
                lambda: self._handle_urgent_click(camera_id, room_label, patient_name, track_id, card))
            self.cards[track_id] = card
        else:
            card.clicked.connect(card.remove)
        self.notif_stack.insertWidget(0, card)

    def _handle_urgent_click(self, camera_id, room_label, patient_name, track_id, card):
        dialog = ResolveFallDialog(room_label, patient_name, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            patient_name, caregiver_name = dialog.values()
            db_stub.resolve_fall_event(camera_id, track_id, patient_name, caregiver_name)
            # main_server가 아직 resolve_fall 명령을 처리하지 않음 (지시서 기준) —
            # 전송 코드만 미리 만들어둠. 서버 지원되면 이 send() 호출은 그대로 유효.
            self.link.send({
                "cmd": "resolve_fall",
                "camera_id": camera_id,
                "track_id": track_id,
                "patient_name": patient_name,
                "caregiver_name": caregiver_name,
            })
            box = self.boxes.get(camera_id)
            if box is not None:
                box.set_state('normal')   # 서버 응답 전까지 화면상 낙관적 처리
            card.remove()
            self.cards.pop(track_id, None)

# admin_gui/fall_tab.py
"""낙상 탭 — 목업(gui_mockup_v5.html)의 낙상 탭 동작 + 스타일을 그대로 구현.

settings.CAMERA_ROLES가 'fall'인 카메라(침대 카메라)만 표시. 상태에 따라 카드 테두리 색이
바뀌고(정상=연회색, 주의=주황, 낙상=빨강), 낙상/주의 이벤트가 발생하면 화면
우측에 알림 카드가 쌓인다. 낙상(빨강) 카드를 클릭하면 팝업이 뜬다. 카메라(=
침상)는 환자와 1:1로 고정 매칭되어 있으므로(settings.CAMERA_PATIENTS) 환자명은
main_server 환자 목록(get_patients)의 camera_id로 채워서 보여주고, 요양보호사명만 입력하면 "처리 완료" 버튼이 활성화된다.
주의(주황) 카드는 클릭하면 바로 닫힌다(목업과 동일).

처리 완료는 db_client.resolve_fall()로 main_server에 요청한다 — 서버가 그
요청을 받으면 FallAnalyzer 상태를 초기화하고 fall_logs에 처리완료를 기록한다.
화면은 응답을 기다리지 않고 로컬에서 낙관적으로 먼저 정상 상태로 되돌린다.
낙상/주의 이벤트 자체는 main_server가 FallAnalyzer 결과를 받는 즉시 직접
기록하므로(db_client.py 주석 참고), GUI가 따로 로그 기록 요청을 보내지 않는다
— 하단 "낙상 로그" 패널은 순수하게 화면에 보여주기 위한 로컬 UI 이력이다.

낙상 로그는 카메라 박스마다 따로 두지 않고, 탭 전체에서 하나만 공유한다
(스타일 개선 작업 — 101호/102호 로그가 따로 있으면 어느 쪽이 최근인지
한눈에 안 들어온다는 피드백에 따라 통합). 어느 병실 이벤트인지는 각 줄
맨 앞의 "[101호]" 같은 표시로 구분한다.

재알림(알림 중복) 방지: 같은 카메라에 이미 처리되지 않은 낙상 알림이 떠
있는 동안에는 같은 카메라에서 또 낙상이 감지돼도 새 알림 카드를 띄우지
않는다(self.active_camera_alerts). 요양보호사가 "처리 완료"를 눌러야만
그 카메라의 알림이 다시 허용된다 — 한 명의 환자를 계속 추적 감지하면서
매번 알림이 쌓여 피로해지는 문제를 막기 위함(사용자 확인 완료, 로직 변경
승인됨).

스타일 노트: 카드 헤더(방/이름 + 상태 뱃지) 줄에 고정 높이를 주지 않으면,
그리드가 창 크기에 맞춰 카드를 세로로 늘릴 때 남는 공간이 헤더 레이아웃으로
새어 들어가서 뱃지가 세로로 길게 늘어나는 버그가 생긴다 — header 컨테이너와
badge에 setFixedHeight를 줘서 막는다.
"""
import time

from . import db_client
from .drawing import draw_camera_overlay, fit_to_view, to_pixmap
from config.settings import CAMERA_PORTS, CAMERA_ROLES
from .qt_compat import (
    Qt, QDialog, QFormLayout, QFont, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
    pyqtSignal,
)
from .ui_kit import apply_card_shadow, make_section_panel, style_pill_badge

# 침대 카메라 → 병실 표시명 (화면 표시용). 여기 없으면 camera_id를 그대로 표시
FALL_ROOM_LABELS = {'CAM-01': '101호', 'CAM-02': '102호'}

# 낙상 탭에 표시할 카메라: settings.CAMERA_ROLES가 'fall'인 것만.
# 환자명은 처음엔 '-' → main_server 환자 목록이 도착하면 set_patients()가 채운다
FALL_CAMERAS = [(cam_id, FALL_ROOM_LABELS.get(cam_id, cam_id), '-')
                for cam_id in CAMERA_PORTS if CAMERA_ROLES.get(cam_id) == 'fall']

STATE_BORDER = {'normal': '#e5e7eb', 'checking': '#f59e0b', 'alert': '#ef4444'}
STATE_BADGE_TEXT = {'normal': '정상', 'checking': '확인 필요', 'alert': '낙상 감지'}
STATE_BADGE_BG = {'normal': '#2f9e44', 'checking': '#f59e0b', 'alert': '#ef4444'}

# 하단 공용 낙상 로그에 남기는 최근 항목 수 — 카메라 2대 몫을 한 목록에 같이
# 쌓으므로 예전(카메라별 20개)보다 넉넉하게 잡는다.
LOG_MAX_ITEMS = 30


class FallCameraBox(QWidget):
    """카메라 1대 표시 카드 — 영상 + 방/환자 이름 + 상태 뱃지(알약) + 상태별 카드 테두리"""

    def __init__(self, camera_id, room_label, patient_name):
        super().__init__()
        self.camera_id = camera_id
        self.room_label = room_label
        self.patient_name = patient_name
        self.state = 'normal'

        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # 헤더 줄: 고정 높이로 감싸서 카드가 늘어나도 뱃지가 같이 늘어나지 않게 한다.
        header_container = QWidget()
        header_container.setFixedHeight(30)
        header = QHBoxLayout(header_container)
        header.setContentsMargins(0, 0, 0, 0)

        # 카메라 = 침대 단위 모니터링(한 병실에 침대 여러 개 · 침대마다 카메라 1개)이라
        # 헤더에 환자명까지 붙이면 오해의 소지가 있다 — 깔끔하게 병실 번호만 표시.
        name_label = QLabel(room_label)
        name_label.setStyleSheet("font-size:15px; font-weight:600; color:#111827; border:none;")

        self.badge = QLabel(STATE_BADGE_TEXT['normal'])

        header.addWidget(name_label)
        header.addStretch()
        header.addWidget(self.badge)
        layout.addWidget(header_container)

        self.video_label = QLabel()
        self.video_label.setFixedSize(480, 360)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background:#0d0d10; border-radius:10px;")
        # 카드 너비가 그리드 열 너비만큼 넓어지는데 video_label은 고정 크기라,
        # 정렬을 안 주면 QVBoxLayout 기본값(왼쪽 정렬)대로 카드 왼쪽에 치우쳐 보인다 —
        # 가로 중앙 정렬로 카드 한가운데 오게 한다.
        layout.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.status_label = QLabel('-')
        self.status_label.setStyleSheet("color:#9ca3af; font-size:12px; border:none;")
        self.status_label.setFixedHeight(18)
        layout.addWidget(self.status_label)

        # 낙상 로그는 이제 카드마다 따로 두지 않고 탭 하단의 공용 패널 하나로 합쳤다
        # (FallTab._add_log_entry 참고) — 카드는 영상 + 상태만 담당한다.
        layout.addStretch()

        self._apply_badge_style('normal')
        self._apply_border()
        apply_card_shadow(self)

    def _apply_border(self):
        color = STATE_BORDER[self.state]
        self.setStyleSheet(
            f"FallCameraBox {{ background: #ffffff; border: 2px solid {color}; "
            f"border-radius: 14px; }}"
        )

    def _apply_badge_style(self, state):
        style_pill_badge(self.badge, STATE_BADGE_BG[state])

    def set_state(self, state, status_text=None):
        if state not in STATE_BORDER:
            return
        self.state = state
        self._apply_border()
        self.badge.setText(STATE_BADGE_TEXT[state])
        self._apply_badge_style(state)
        if status_text is not None:
            self.status_label.setText(status_text)

    def show_frame(self, frame_bgr):
        view = fit_to_view(frame_bgr, 480, 360)
        draw_camera_overlay(view, self.camera_id, self.room_label)
        self.video_label.setPixmap(to_pixmap(view))


class AlertCard(QFrame):
    """우측에 쌓이는 알림 카드 (주황=주의/즉시 닫힘, 빨강=낙상/팝업 오픈)"""
    clicked = pyqtSignal()

    def __init__(self, room_label, urgent, message):
        super().__init__()
        self.urgent = urgent
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        border_color = '#ef4444' if urgent else '#f59e0b'
        self.setStyleSheet(
            f"AlertCard {{ background: #ffffff; border: 1px solid #e5e7eb; "
            f"border-left: 4px solid {border_color}; border-radius: 10px; }}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        title = QLabel(f"{'낙상 감지' if urgent else '주의'} · {room_label}")
        title.setStyleSheet("font-weight:700; font-size:13px; color:#111827; border:none;")
        body = QLabel(message)
        body.setWordWrap(True)
        body.setStyleSheet("color:#6b7280; font-size:12px; border:none;")
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
        # 이미 처리되지 않은 낙상 알림이 떠 있는 카메라 id 집합 — 여기 들어있는
        # 카메라는 "처리 완료"가 눌리기 전까지 새 낙상 알림을 다시 띄우지 않는다.
        self.active_camera_alerts = set()
        self._camera_lookup = {cam_id: (room, patient) for cam_id, room, patient in FALL_CAMERAS}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)

        content = QHBoxLayout()
        content.setSpacing(16)

        grid_widget = QWidget()
        self._build_grid(grid_widget)

        # 우측 알림 목록 — 낙상 팝업과는 별개의 "패널"임을 한눈에 알 수 있게
        # 회색 배경 + "메시지 알림" 제목을 얹은 하나의 섹션으로 감싼다.
        notif_panel, notif_content, _ = make_section_panel("메시지 알림")
        notif_panel.setFixedWidth(280)

        self.notif_stack = QVBoxLayout()
        self.notif_stack.setSpacing(10)
        self.notif_stack.addStretch()
        notif_content.addLayout(self.notif_stack)

        # 메시지 알림 패널이 101호/102호 카메라 박스와 같은 줄 위치·높이를
        # 갖도록, 정렬을 따로 지정하지 않는다 — QHBoxLayout 기본 동작이 두
        # 위젯을 행 높이(=더 큰 쪽인 카메라 박스 높이)에 맞춰 위아래로
        # 늘려서 채우기 때문에, 이대로 두면 시작 위치와 높이가 저절로
        # 카메라 박스와 맞는다. (이전에 AlignTop을 줬던 건 별도의 다른
        # 정렬 문제 때문이었는데, 그 결과로 패널이 늘어나지 않게 돼서
        # 지금의 "위치/높이가 안 맞는" 문제가 생겼다 — 그래서 제거한다.)
        content.addWidget(grid_widget, stretch=3)
        content.addWidget(notif_panel, stretch=1)

        outer.addLayout(content)

        # 낙상 로그 — 카메라별로 따로 있던 것을 탭 전체에서 공유하는 패널 하나로
        # 통합했다(각 줄 앞에 "[101호]" 식으로 병실을 표시해서 구분).
        log_panel, log_content, _ = make_section_panel("낙상 로그")
        self.log_list = QListWidget()
        # 고정 높이를 주면 목록이 회색 패널보다 작아 보여서(패널만 늘어남)
        # 부자연스럽다 — 고정 높이를 없애고, 아래에서 log_panel에 stretch를
        # 줘서 목록(QListWidget 기본이 세로로 Expanding)이 패널을 꽉 채우게 한다.
        self.log_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.log_list.setStyleSheet(
            "QListWidget { border:1px solid #e5e7eb; border-radius:8px; "
            "background:#ffffff; font-size:12px; color:#374151; } "
            "QListWidget::item { padding:4px 8px; border:none; }"
        )
        log_content.addWidget(self.log_list)
        # 카메라 박스 줄(content)은 고정 크기 영상으로 정해지는 자연 높이를
        # 그대로 쓰고, 남는 세로 공간은 낙상 로그 패널이 가져가게 한다 —
        # 창 높이에 따라 로그 패널이 늘거나 줄어서, 이전처럼 고정 높이 때문에
        # 하단이 잘리는 문제도 같이 줄어든다.
        outer.addWidget(log_panel, stretch=1)

    def set_patients(self, patients):
        """main_server에서 환자 목록이 도착하면 app.py가 호출 — Patient.camera_id(낙상 카메라 배정)로 환자명 갱신"""
        for p in patients:
            if p.camera_id in self._camera_lookup:
                room_label, _ = self._camera_lookup[p.camera_id]
                self._camera_lookup[p.camera_id] = (room_label, p.name)
                self.boxes[p.camera_id].patient_name = p.name

    def _build_grid(self, grid_widget):
        grid = QGridLayout(grid_widget)
        grid.setSpacing(16)
        for i, (camera_id, room_label, patient_name) in enumerate(FALL_CAMERAS):
            box = FallCameraBox(camera_id, room_label, patient_name)
            self.boxes[camera_id] = box
            grid.addWidget(box, i // 2, i % 2)

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
            # 낙상/주의 이벤트 자체는 main_server가 이미 직접 기록하므로 여기서
            # DB에 따로 요청을 보내지 않는다 — add_log_entry는 화면용 로컬 이력.
            if name.endswith('TO_ALERT'):
                self._add_log_entry(room_label, "낙상 감지")
                # 이 카메라에 이미 처리되지 않은 낙상 알림이 떠 있으면 또 띄우지
                # 않는다 — 같은 사람을 계속 추적 감지해도 "처리 완료" 전까지는
                # 알림이 한 번만 뜬다(사용자 확인된 동작).
                if camera_id not in self.active_camera_alerts:
                    self._add_alert_card(camera_id, room_label, patient_name, track_id, urgent=True,
                                          message="낙상이 감지되었습니다. 확인 후 처리해주세요.")
                    self.active_camera_alerts.add(camera_id)
            elif name.endswith('TO_CHECKING'):
                self._add_log_entry(room_label, "자세 확인 필요")
                self._add_alert_card(camera_id, room_label, patient_name, track_id, urgent=False,
                                      message="자세 확인이 필요합니다.")

    def _add_log_entry(self, room_label, text):
        """낙상 로그 패널에 최신 이벤트를 맨 위에 쌓는다. 너무 길어지지 않게
        최근 LOG_MAX_ITEMS개까지만 유지한다."""
        timestamp = time.strftime('%H:%M:%S')
        self.log_list.insertItem(0, QListWidgetItem(f"{timestamp}  [{room_label}] {text}"))
        while self.log_list.count() > LOG_MAX_ITEMS:
            self.log_list.takeItem(self.log_list.count() - 1)

    def _add_alert_card(self, camera_id, room_label, patient_name, track_id, urgent, message):
        # 알림 카드 제목도 카드 헤더와 통일 — 환자명 없이 병실 번호만 (침대별
        # 카메라라 환자명을 붙이면 오해의 소지가 있음). 처리 팝업에서는 환자명을 보여준다.
        card = AlertCard(room_label, urgent, message)
        if urgent:
            card.clicked.connect(
                lambda: self._handle_urgent_click(camera_id, room_label, patient_name, track_id, card))
            self.cards[track_id] = card
        else:
            card.clicked.connect(card.remove)
        self.notif_stack.insertWidget(0, card)

    def _handle_urgent_click(self, camera_id, room_label, patient_name, track_id, card):
        # 알림이 환자 목록 도착 전에 떴을 수 있으니, 클릭 시점의 최신 환자명으로 보여준다
        patient_name = self._camera_lookup.get(camera_id, (room_label, patient_name))[1]
        dialog = ResolveFallDialog(room_label, patient_name, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            patient_name, caregiver_name = dialog.values()
            # main_server에 처리 완료 요청 — 서버가 FallAnalyzer 상태 초기화 +
            # fall_logs에 처리완료를 기록한다 (db_client.py 참고).
            db_client.resolve_fall(camera_id, track_id, caregiver_name)
            box = self.boxes.get(camera_id)
            if box is not None:
                box.set_state('normal')   # 서버 응답 전까지 화면상 낙관적 처리
            self._add_log_entry(room_label, f"{caregiver_name}님이 처리 완료")
            card.remove()
            self.cards.pop(track_id, None)
            # 처리 완료가 끝났으니 이 카메라는 다시 낙상 알림을 받을 수 있다.
            self.active_camera_alerts.discard(camera_id)

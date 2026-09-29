# admin_gui/gait_tab.py
"""보행 탭 — 좌측 측정 카메라 선택 + 환자 목록(main_server/DB에서 받음) + 우측 분석 화면.

"보행 분석 시작" 누르기 전에는 결과 없음(빈 상태) → 시작을 누르면
main_server에 mode=1로 전환 명령을 보내고, 이후 들어오는 결과의
cumulative_scores를 도넛 차트로 그린다. "중지"를 누르면 분석만 멈추고
마지막 결과는 화면에 남겨둔다(목업 v2와 동일). "저장"은 db_client에 위임.

환자 목록은 이제 로컬 더미(patients.py, 삭제됨)가 아니라 main_server가
db_client.request_patients() 응답으로 주는 실제 목록이다 — app.py의
handle_server_message()가 그 응답을 받으면 set_patients()를 호출해준다.

주의: db_client.Patient.camera_id는 "낙상(병실) 카메라 배정"이라 보행 측정에는
쓰지 않는다(db_client.py 주석 참고) — 그래서 스트레칭 탭과 동일하게 "측정
카메라"를 환자와 별도로 고르게 한다.

주의: cumulative_scores의 키 이름(normal/parkinsons/stroke/myopathic/
antalgic/abnormal)은 ai_server/models/의 모델 파일명(model_*.pth)에서
추론한 것으로, 실제 gait_analyzer.py 출력과 다를 수 있다. 보행 데이터
연동 담당자가 실제 키 이름으로 DISEASE_ORDER를 맞춰줘야 한다.
"""
from . import db_client
from .drawing import fit_to_view, to_pixmap
from config.settings import CAMERA_PORTS
from .qt_compat import (
    Qt, QColor, QComboBox, QFont, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPainter, QPen, QPushButton, QRectF, QVBoxLayout, QWidget,
)

# TODO(보행 데이터 연동 담당자): 실제 gait_analyzer.py의 cumulative_scores 키와
# 맞는지 확인/수정 필요. 현재는 ai_server/models/의 model_*.pth 파일명에서 추론.
DISEASE_ORDER = ['normal', 'parkinsons', 'stroke', 'myopathic', 'antalgic', 'abnormal']
DISEASE_LABEL = {
    'normal': '정상', 'parkinsons': '파킨슨', 'stroke': '뇌졸중',
    'myopathic': '근병증', 'antalgic': '통증성 보행', 'abnormal': '기타 이상',
}
# 스트레칭 탭 영상만큼 카메라 화면이 작아 보인다는 피드백 — 기존 480x360에서 키움.
VIEW_W, VIEW_H = 640, 480
# dataviz 스킬 validate_palette.js로 light/dark 모두 통과 확인된 8색 팔레트 중 앞 6개
SERIES_COLORS_LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
SERIES_COLORS_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300']


class DonutChartView(QWidget):
    """cumulative_scores(dict)를 도넛 차트로 그림. QPainter.drawArc 기반."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(220, 220)
        self.scores = {}

    def set_scores(self, scores):
        self.scores = scores or {}
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height()) - 20
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        ring_width = max(14, int(side * 0.16))
        inset = ring_width / 2 + 4
        arc_rect = rect.adjusted(inset, inset, -inset, -inset)

        total = sum(max(0.0, v) for v in self.scores.values()) or 1.0
        start_angle = 90 * 16   # 12시 방향에서 시작
        top_key, top_val = None, -1

        for i, key in enumerate(DISEASE_ORDER):
            value = max(0.0, self.scores.get(key, 0.0))
            if value > top_val:
                top_key, top_val = key, value
            span = int(round(-(value / total) * 360 * 16))
            if span == 0:
                continue
            color = QColor(SERIES_COLORS_LIGHT[i % len(SERIES_COLORS_LIGHT)])
            pen = QPen(color, ring_width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(arc_rect, start_angle, span)
            start_angle += span

        painter.setPen(QColor('#0b0b0b'))
        painter.setFont(QFont('', 13, QFont.Weight.Bold))
        pct = (self.scores.get(top_key, 0.0) / total * 100) if top_key else 0
        label = f"{DISEASE_LABEL.get(top_key, '-')}\n{pct:.0f}%" if top_key else "결과 없음"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)


class GaitTab(QWidget):
    def __init__(self, link):
        super().__init__()
        self.link = link
        self.patients = []          # main_server 연결 후 app.py가 set_patients()로 채움
        self.selected = None
        self.analyzing = False
        # set_mode/save_gait_session에 실제로 쓰는 카메라 — Patient.camera_id(낙상
        # 카메라)와는 별개로, 여기서 직접 고른 값. 분석을 시작할 때 확정된다.
        self.active_camera_id = None

        root = QHBoxLayout(self)

        # 좌측: 측정 카메라 선택 + 환자 검색 + 목록 (스트레칭 탭과 동일한 사이드바 구성)
        left = QVBoxLayout()
        camera_title = QLabel("측정 카메라")
        camera_title.setFont(QFont('', -1, QFont.Weight.Bold))
        left.addWidget(camera_title)
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(list(CAMERA_PORTS))
        left.addWidget(self.camera_combo)

        patient_title = QLabel("환자 선택")
        patient_title.setFont(QFont('', -1, QFont.Weight.Bold))
        left.addWidget(patient_title)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("환자 검색")
        self.search_input.textChanged.connect(self._filter_patients)
        left.addWidget(self.search_input)

        self.patient_list = QListWidget()
        self.patient_list.currentRowChanged.connect(self._on_select_patient)
        left.addWidget(self.patient_list)
        self._populate_patient_list(self.patients)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(250)
        root.addWidget(left_widget)

        # 우측: 영상 + 상태 + 도넛차트/빈 상태 + 버튼
        right = QVBoxLayout()
        header = QHBoxLayout()
        # 환자를 아직 안 골랐을 때 "환자를 선택하세요" 안내문을 따로 띄우지 않고
        # 그냥 비워둔다 — 왼쪽 목록 자체가 이미 선택하라는 UI이므로 중복.
        self.name_label = QLabel("")
        self.name_label.setFont(QFont('', -1, QFont.Weight.Bold))
        self.status_badge = QLabel("")
        header.addWidget(self.name_label)
        header.addStretch()
        header.addWidget(self.status_badge)
        right.addLayout(header)

        self.video_label = QLabel()
        self.video_label.setFixedSize(VIEW_W, VIEW_H)
        self.video_label.setStyleSheet("background:#111; border-radius:6px;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # video_label은 고정 크기라, 정렬을 안 주면 QVBoxLayout 기본값(왼쪽 정렬)대로
        # 우측 칼럼 왼쪽에 붙어 보인다 — 가로 중앙 정렬로 칼럼 한가운데 오게 한다.
        right.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.empty_label = QLabel("보행 분석을 시작하면 결과가 표시됩니다")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color:#888; padding:24px;")
        right.addWidget(self.empty_label)

        result_row = QHBoxLayout()
        self.donut = DonutChartView()
        self.donut.hide()
        result_row.addWidget(self.donut)
        self.legend_label = QLabel("")
        self.legend_label.setWordWrap(True)
        self.legend_label.hide()
        result_row.addWidget(self.legend_label)
        right.addLayout(result_row)

        actions = QHBoxLayout()
        self.toggle_btn = QPushButton("보행 분석 시작")
        self.toggle_btn.clicked.connect(self._toggle_analysis)
        self.save_btn = QPushButton("저장")
        self.save_btn.clicked.connect(self._save_session)
        actions.addWidget(self.toggle_btn)
        actions.addWidget(self.save_btn)
        right.addLayout(actions)

        root.addLayout(right)
        self.toggle_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    def _populate_patient_list(self, patients):
        self.patient_list.clear()
        for p in patients:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.patient_list.addItem(item)

    def _filter_patients(self, text):
        text = text.strip()
        filtered = [p for p in self.patients if text in p.name or text in p.room] if text else self.patients
        self._populate_patient_list(filtered)

    def set_patients(self, patients):
        """main_server에서 환자 목록이 도착하면 app.py가 호출 (검색어는 유지)"""
        self.patients = patients
        self._filter_patients(self.search_input.text())

    def _on_select_patient(self, row):
        if row < 0:
            self.selected = None
            self.toggle_btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            return
        item = self.patient_list.item(row)
        self.selected = item.data(Qt.ItemDataRole.UserRole)
        self.name_label.setText(f"{self.selected.name} · {self.selected.room}")
        self.analyzing = False
        # 환자를 새로 고르면 이전 측정의 화면 갱신을 멈춘다 — "측정 카메라"를 다시
        # 골라서 "보행 분석 시작"을 눌러야 새 active_camera_id가 잡힌다.
        self.active_camera_id = None
        self.toggle_btn.setText("보행 분석 시작")
        self.toggle_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self._show_empty_state()

    def _show_empty_state(self):
        self.status_badge.setText("")
        self.donut.hide()
        self.legend_label.hide()
        self.empty_label.show()

    def _toggle_analysis(self):
        if self.selected is None:
            return
        self.analyzing = not self.analyzing
        if self.analyzing:
            # 환자.camera_id는 낙상 카메라라 여기 쓰면 안 된다 — 좌측에서 고른
            # "측정 카메라"로 실제 분석을 시작한다.
            self.active_camera_id = self.camera_combo.currentText()
            self.toggle_btn.setText("분석 중지")
            self.status_badge.setText("분석 중")
            self.link.send({"cmd": "set_mode", "camera_id": self.active_camera_id, "mode": 1})
        else:
            self.toggle_btn.setText("보행 분석 시작")
            self.status_badge.setText("중지됨")
            # 마지막 결과는 화면에 남겨둠 (목업과 동일) — active_camera_id를
            # 그대로 둬서 show_frame이 계속 그 카메라의 프레임을 그린다.

    def _save_session(self):
        if self.selected is None or not self.donut.scores:
            return
        db_client.save_gait_session(self.selected.patient_id, self.donut.scores, self.active_camera_id)
        self.save_btn.setText("✓ 저장됨")
        from .qt_compat import QTimer
        QTimer.singleShot(1500, lambda: self.save_btn.setText("저장"))

    def show_frame(self, camera_id, frame_bgr):
        if self.active_camera_id is None or camera_id != self.active_camera_id:
            return
        self.video_label.setPixmap(to_pixmap(fit_to_view(frame_bgr, VIEW_W, VIEW_H)))

    def render(self, camera_id, msg):
        """result_receiver가 mode 1 메시지를 줄 때마다 admin_gui.py의 dispatch()가 호출"""
        if self.selected is None or camera_id != self.active_camera_id or not self.analyzing:
            return
        data = msg.get('data', {})
        detections = data.get('detections', [])
        if not detections:
            return
        raw = detections[0].get('raw_data', {})
        scores = raw.get('cumulative_scores', {})
        if not scores:
            return
        self.empty_label.hide()
        self.donut.set_scores(scores)
        self.donut.show()
        self._render_legend(scores)
        self.legend_label.show()
        self.save_btn.setEnabled(True)

    def _render_legend(self, scores):
        total = sum(max(0.0, v) for v in scores.values()) or 1.0
        lines = []
        for key in DISEASE_ORDER:
            value = max(0.0, scores.get(key, 0.0))
            pct = value / total * 100
            lines.append(f"{DISEASE_LABEL.get(key, key)}: {pct:.0f}%")
        self.legend_label.setText("\n".join(lines))

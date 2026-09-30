# admin_gui/patient_tab.py
"""환자 관리 탭 — 좌측 환자 목록 + 우측 선택한 환자의 DB 정보.

좌측 사이드바는 보행/스트레칭 탭과 같은 구성(회색 섹션 패널 "환자 선택" + 검색 +
카드형 목록, 폭 250 — ui_kit.style_selectable_list / populate_patient_list 공용).
우측은 회색 섹션 패널(ui_kit.make_section_panel)로 나눠서 보여준다:
  기본 정보 → 요약 카드 3개 → 낙상 기록 → 보행 기록 / 스트레칭 기록

데이터 흐름 (GUI는 DB에 직접 접속하지 않는다):
  환자 목록: 다른 탭과 같이 app.py가 get_patients 응답을 set_patients()로 넘겨줌
  상세 정보: 환자를 고르면 db_client.request_patient_detail() → main_server가
            DBManager.get_patient_detail()로 조회 → 응답을 app.py가 on_detail()로 넘겨줌
응답은 비동기로 오므로, 그 사이 다른 환자를 골랐으면 늦게 온 응답은 버린다.

낙상 알림(우상단 토스트, 낙상 탭 빨간 점)은 AdminWindow가 탭과 상관없이 띄우므로
이 탭에서도 그대로 보인다.
"""
from . import db_client
from . import patient_drawing as pd
from .qt_compat import Qt, QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QVBoxLayout, QWidget
from .ui_kit import make_section_panel, populate_patient_list, style_selectable_list


class PatientTab(QWidget):
    def __init__(self, link):
        super().__init__()
        self.link = link
        self.patients = []          # main_server 연결 후 app.py가 set_patients()로 채움 (db_client.Patient)
        self.selected = None        # 지금 고른 Patient

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)     # 보행/스트레칭 탭과 같은 여백
        root.setSpacing(16)

        # ---- 좌측: 환자 검색 + 목록 (보행/스트레칭 탭과 같은 사이드바) ----
        left = QVBoxLayout()
        left.setSpacing(16)

        patient_panel, patient_content, self.list_title = make_section_panel("환자 선택")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("이름 또는 병실 검색")
        self.search_input.textChanged.connect(self._filter_patients)
        patient_content.addWidget(self.search_input)

        self.patient_list = QListWidget()
        style_selectable_list(self.patient_list)
        self.patient_list.currentItemChanged.connect(self._on_select_patient)
        patient_content.addWidget(self.patient_list)

        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.refresh)
        patient_content.addWidget(self.refresh_btn)
        left.addWidget(patient_panel, stretch=1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setFixedWidth(250)
        root.addWidget(left_widget)

        # ---- 우측: 상세 (스크롤) ----
        detail = QWidget()
        detail.setObjectName("patientDetail")
        detail.setStyleSheet("#patientDetail { background:transparent; }")
        right = QVBoxLayout(detail)
        right.setContentsMargins(0, 0, 8, 0)
        right.setSpacing(16)

        self.empty_label = QLabel("왼쪽에서 환자를 선택하면 정보가 표시됩니다")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet("color:#888; padding:48px;")
        right.addWidget(self.empty_label)

        # 기본 정보
        self.info_panel, info_content, _ = make_section_panel("기본 정보")
        self.name_label, self.room_badge, self.grade_badge = pd.build_info_header(info_content)
        self.info_values = pd.build_info_grid(info_content)
        right.addWidget(self.info_panel)

        # 요약 카드 3개
        self.stats_row = QWidget()
        stats_layout = QHBoxLayout(self.stats_row)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(16)
        self.fall_card = pd.StatCard("낙상 기록")
        self.gait_card = pd.StatCard("최근 보행 측정")
        self.stretch_card = pd.StatCard("스트레칭 정확도")
        for card in (self.fall_card, self.gait_card, self.stretch_card):
            stats_layout.addWidget(card)
        right.addWidget(self.stats_row)

        # 낙상 기록
        self.fall_panel, fall_content, _ = make_section_panel("낙상 기록")
        self.fall_table = pd.make_table(["주의 감지", "낙상 확정", "상태", "처리 완료"], height=170)
        fall_content.addWidget(self.fall_table)
        right.addWidget(self.fall_panel)

        # 보행 / 스트레칭 기록 (좌우 배치)
        self.logs_row = QWidget()
        logs_layout = QHBoxLayout(self.logs_row)
        logs_layout.setContentsMargins(0, 0, 0, 0)
        logs_layout.setSpacing(16)

        gait_panel, gait_content, _ = make_section_panel("보행 기록")
        latest_title = QLabel("최근 측정 분포")
        latest_title.setStyleSheet("color:#6b7280; font-size:12px; border:none;")
        gait_content.addWidget(latest_title)
        self.gait_bar = pd.GaitDistributionBar()
        gait_content.addWidget(self.gait_bar)
        self.gait_table = pd.make_table(["측정 시각", "정상", "최다 이상 유형"], height=180)
        gait_content.addWidget(self.gait_table)
        logs_layout.addWidget(gait_panel)

        stretch_panel, stretch_content, _ = make_section_panel("스트레칭 기록")
        self.stretch_table = pd.make_table(["측정 시각", "운동", "정확도"], height=270)
        stretch_content.addWidget(self.stretch_table)
        logs_layout.addWidget(stretch_panel)
        right.addWidget(self.logs_row)

        right.addStretch()
        root.addWidget(pd.make_scroll_area(detail), stretch=1)

        self._detail_widgets = [self.info_panel, self.stats_row, self.fall_panel, self.logs_row]
        self._show_detail(False)

    # ============ 목록 ============
    def refresh(self):
        """새로고침: 환자 목록(모든 탭 공통 응답)과 지금 보고 있는 환자 상세를 다시 요청"""
        db_client.request_patients()
        self.reload_detail()

    def set_patients(self, patients):
        """main_server에서 환자 목록이 도착하면 app.py가 호출 (검색어/선택 유지)"""
        self.patients = patients
        self._filter_patients(self.search_input.text())

    def _filter_patients(self, text):
        text = text.strip()
        filtered = [p for p in self.patients if text in p.name or text in p.room] if text else self.patients
        self.list_title.setText(f"환자 선택")

        # 목록을 다시 채우는 동안 선택 변경 신호가 연달아 오지 않게 막고, 보고 있던 환자는 다시 선택
        self.patient_list.blockSignals(True)
        populate_patient_list(self.patient_list, filtered)
        if self.selected is not None:
            for row in range(self.patient_list.count()):
                p = self.patient_list.item(row).data(Qt.ItemDataRole.UserRole)
                if p.patient_id == self.selected.patient_id:
                    self.patient_list.setCurrentRow(row)
                    break
        self.patient_list.blockSignals(False)

    # ============ 상세 ============
    def _on_select_patient(self, current, previous=None):
        if current is None:
            return
        self.selected = current.data(Qt.ItemDataRole.UserRole)
        self.reload_detail()

    def reload_detail(self):
        """지금 고른 환자의 상세를 main_server에 요청 (탭 전환/새로고침 때도 호출)"""
        if self.selected is None:
            return
        if db_client.request_patient_detail(self.selected.patient_id) is None:
            self.empty_label.setText("서버에 연결되지 않아 정보를 불러올 수 없습니다")
            self._show_detail(False)
            return
        if not self.info_panel.isVisible():
            self.empty_label.setText("불러오는 중…")

    def on_detail(self, detail):
        """get_patient_detail 응답 도착 시 app.py가 호출"""
        patient = (detail or {}).get('patient')
        if not patient or self.selected is None:
            return
        if str(patient.get('id')) != str(self.selected.patient_id):
            return      # 응답이 오는 사이 다른 환자를 골랐음 → 늦게 온 응답은 버림
        self.render_detail(detail)

    def on_detail_failed(self):
        """get_patient_detail 요청 실패 응답 (ok=False) 시 app.py가 호출"""
        self.empty_label.setText("환자 정보를 불러오지 못했습니다")
        self._show_detail(False)

    def render_detail(self, detail):
        patient = detail['patient']
        pd.draw_info_header(self.name_label, self.room_badge, self.grade_badge, patient)
        pd.draw_info_grid(self.info_values, patient)
        pd.draw_stat_cards(self.fall_card, self.gait_card, self.stretch_card, detail)
        pd.draw_fall_table(self.fall_table, detail.get('fall_logs') or [])
        gait_logs = detail.get('gait_logs') or []
        self.gait_bar.set_scores(gait_logs[0] if gait_logs else {})
        pd.draw_gait_table(self.gait_table, gait_logs)
        pd.draw_stretch_table(self.stretch_table, detail.get('stretch_logs') or [])
        self._show_detail(True)

    def _show_detail(self, visible):
        self.empty_label.setVisible(not visible)
        for widget in self._detail_widgets:
            widget.setVisible(visible)

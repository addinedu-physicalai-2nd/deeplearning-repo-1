# admin_gui/patient_tab.py
"""환자 관리 탭 — 좌측 환자 목록 + 우측 선택한 환자의 DB 정보.

좌측 사이드바는 보행/스트레칭 탭과 같은 구성(제목 + 검색 + 목록, 폭 250).
우측은 회색 섹션 패널(ui_kit.make_section_panel)로 나눠서 보여준다:
  기본 정보 → 요약 카드 3개 → 낙상 기록 → 보행 기록 / 스트레칭 기록

데이터는 source 객체에서 받는다 (탭은 DB/네트워크를 직접 모름):
  source.list_patients()        → [{'patient_id', 'name', 'room', ...}, ...]
  source.get_patient_detail(id) → {'patient': {...}, 'fall_logs': [...],
                                   'gait_logs': [...], 'stretch_logs': [...]}
지금은 patient_tab_test.py의 DirectDBSource(DB 직접 연결)를 넣어서 테스트하고,
합칠 때는 main_server 요청(db_client)으로 같은 형식을 돌려주는 source로 바꾸면 된다.

그리기 함수는 drawing.py 작업이 끝날 때까지 patient_drawing.py에 따로 둔다.
"""
from . import patient_drawing as pd
from .qt_compat import Qt, QFont, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget
from .ui_kit import make_section_panel


class PatientTab(QWidget):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.patients = []          # list_patients() 결과 (dict 리스트)
        self.selected_id = None

        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)     # 보행/스트레칭 탭과 같은 여백
        root.setSpacing(16)

        # ---- 좌측: 환자 검색 + 목록 ----
        left = QVBoxLayout()
        left.setSpacing(8)
        title_row = QHBoxLayout()
        list_title = QLabel("환자 목록")
        list_title.setFont(QFont('', -1, QFont.Weight.Bold))
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("color:#9ca3af; font-size:12px;")
        title_row.addWidget(list_title)
        title_row.addStretch()
        title_row.addWidget(self.count_label)
        left.addLayout(title_row)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("이름 또는 병실 검색")
        self.search_input.textChanged.connect(self._filter_patients)
        left.addWidget(self.search_input)

        self.patient_list = QListWidget()
        self.patient_list.currentItemChanged.connect(self._on_select_patient)
        left.addWidget(self.patient_list)

        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self.refresh)
        left.addWidget(self.refresh_btn)

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
        self.fall_table = pd.make_table(["주의 감지", "낙상 확정", "상태", "처리 완료", "카메라"], height=170)
        fall_content.addWidget(self.fall_table)
        right.addWidget(self.fall_panel)

        # 보행 / 스트레칭 기록 (좌우 배치)
        logs_row = QWidget()
        logs_layout = QHBoxLayout(logs_row)
        logs_layout.setContentsMargins(0, 0, 0, 0)
        logs_layout.setSpacing(16)

        gait_panel, gait_content, _ = make_section_panel("보행 기록")
        latest_title = QLabel("최근 측정 분포")
        latest_title.setStyleSheet("color:#6b7280; font-size:12px; border:none;")
        gait_content.addWidget(latest_title)
        self.gait_bar = pd.GaitDistributionBar()
        gait_content.addWidget(self.gait_bar)
        self.gait_table = pd.make_table(["측정 시각", "정상", "최다 이상 유형", "카메라"], height=180)
        gait_content.addWidget(self.gait_table)
        logs_layout.addWidget(gait_panel)

        stretch_panel, stretch_content, _ = make_section_panel("스트레칭 기록")
        self.stretch_table = pd.make_table(["측정 시각", "운동", "정확도", "카메라"], height=270)
        stretch_content.addWidget(self.stretch_table)
        logs_layout.addWidget(stretch_panel)
        self.logs_row = logs_row
        right.addWidget(logs_row)

        right.addStretch()
        root.addWidget(pd.make_scroll_area(detail), stretch=1)

        self._detail_widgets = [self.info_panel, self.stats_row, self.fall_panel, self.logs_row]
        self._show_detail(False)

    # ============ 목록 ============
    def refresh(self):
        """source에서 환자 목록을 다시 받아온다 (선택 중인 환자는 유지)"""
        patients = self.source.list_patients()
        if patients is None:
            self.count_label.setText("불러오기 실패")
            return
        self.set_patients(patients)

    def set_patients(self, patients):
        self.patients = patients
        self._filter_patients(self.search_input.text())

    def _filter_patients(self, text):
        text = text.strip()
        filtered = [p for p in self.patients
                    if text in p['name'] or text in (p.get('room') or '')] if text else self.patients
        self.count_label.setText(f"{len(filtered)}명")

        # 목록을 다시 채우는 동안 선택 변경 신호가 연달아 오지 않게 막는다
        self.patient_list.blockSignals(True)
        self.patient_list.clear()
        reselect = None
        for p in filtered:
            item = QListWidgetItem(f"{p['name']}  ·  {p.get('room') or '-'}")
            item.setData(Qt.ItemDataRole.UserRole, p['patient_id'])
            self.patient_list.addItem(item)
            if p['patient_id'] == self.selected_id:
                reselect = item
        if reselect is not None:
            self.patient_list.setCurrentItem(reselect)
        self.patient_list.blockSignals(False)

    # ============ 상세 ============
    def _on_select_patient(self, current, previous=None):
        if current is None:
            return
        patient_id = current.data(Qt.ItemDataRole.UserRole)
        self.selected_id = patient_id
        detail = self.source.get_patient_detail(patient_id)
        if not detail or not detail.get('patient'):
            self.empty_label.setText("환자 정보를 불러오지 못했습니다")
            self._show_detail(False)
            return
        self.render_detail(detail)

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
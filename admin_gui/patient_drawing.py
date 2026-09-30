# admin_gui/patient_drawing.py
"""환자 관리 탭 전용 그리기 헬퍼 (테스트용 — drawing.py 작업이 끝나면 합칠 예정).

drawing.py는 카메라 프레임(OpenCV) 위에 그리는 함수들이라, 여기 있는 Qt 위젯용
헬퍼(정보 그리드, 요약 카드, 기록 표, 보행 분포 막대)와 성격이 조금 다르다.
합칠 때 drawing.py에 그대로 옮기거나 ui_kit.py 쪽으로 옮기면 된다.
로직은 없고, 받은 dict를 화면에 그리기만 한다.
"""
from .gait_tab import DISEASE_LABEL, DISEASE_ORDER, SERIES_COLORS_LIGHT
from .qt_compat import (
    PYQT_VERSION, Qt, QAbstractItemView, QColor, QFont, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QPainter, QRectF, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from .ui_kit import apply_card_shadow, style_pill_badge

# qt_compat에 아직 없는 위젯 — 합칠 때 qt_compat.py로 옮기면 이 블록은 지워도 됨
if PYQT_VERSION == 6:
    from PyQt6.QtWidgets import QScrollArea
else:
    from PyQt5.QtWidgets import QScrollArea

# stretch_logs.straching_id → 운동 이름 (db.sql 주석 기준: 1:목, 2:어깨, 3:허리, 4:무릎)
STRETCH_NAMES = {1: '목', 2: '어깨', 3: '허리', 4: '무릎'}

# fall_logs.status → 뱃지 색 (낙상 탭 뱃지 색과 톤 통일)
FALL_STATUS_BG = {'대기': '#f59e0b', '확인': '#3b82f6', '처리완료': '#2f9e44'}

GENDER_LABEL = {'M': '남', 'F': '여'}

# 기본 정보 그리드에 표시할 항목 (patient dict 키, 라벨) — 2열로 배치
INFO_FIELDS = [
    ('gender', '성별'), ('birth_date', '생년월일'),
    ('age', '나이'), ('admit_date', '입소일'),
    ('room_number', '병실'), ('care_grade', '장기요양등급'),
    ('caregiver_name', '담당 요양보호사'), ('camera_id', '낙상 카메라'),
    ('guardian_name', '보호자'), ('guardian_phone', '보호자 연락처'),
]


def fmt_dt(value):
    """'2026-09-02 23:40:10' → '09-02 23:40'. 없으면 '-'"""
    if not value:
        return '-'
    text = str(value)
    return text[5:16] if len(text) >= 16 else text


def fmt_info(key, value):
    """기본 정보 값 표시용 변환"""
    if value is None or value == '':
        return '-'
    if key == 'gender':
        return GENDER_LABEL.get(value, str(value))
    if key == 'age':
        return f"{value}세"
    if key == 'camera_id':
        return str(value)
    return str(value)


# ============ 스크롤 영역 ============
def make_scroll_area(content_widget):
    """오른쪽 상세 화면이 창보다 길어질 때를 위한 세로 스크롤 (테두리/배경 없음)"""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
    scroll.viewport().setStyleSheet("background:transparent;")
    scroll.setWidget(content_widget)
    return scroll


# ============ 기본 정보 ============
def build_info_header(layout):
    """이름(크게) + 병실/등급 알약 뱃지 줄. (name_label, room_badge, grade_badge) 반환"""
    row = QHBoxLayout()
    row.setSpacing(8)
    name_label = QLabel("")
    # 전역 스타일시트의 font-size가 setFont보다 우선이라 크기는 스타일시트로 지정
    name_label.setStyleSheet("font-size:20px; font-weight:700; color:#111827; border:none;")
    room_badge = QLabel("")
    grade_badge = QLabel("")
    room_badge.hide()
    grade_badge.hide()
    row.addWidget(name_label)
    row.addSpacing(6)
    row.addWidget(room_badge)
    row.addWidget(grade_badge)
    row.addStretch()
    layout.addLayout(row)
    return name_label, room_badge, grade_badge


def draw_info_header(name_label, room_badge, grade_badge, patient):
    name_label.setText(patient.get('name') or '-')
    for badge, text, color in ((room_badge, patient.get('room_number'), '#4f46e5'),
                               (grade_badge, patient.get('care_grade'), '#6b7280')):
        if text:
            badge.setText(str(text))
            style_pill_badge(badge, color)
            badge.show()
        else:
            badge.hide()


def build_info_grid(layout):
    """라벨/값 2열 그리드 + 특이사항 박스. 값 라벨 dict를 반환 (키: INFO_FIELDS 키 + 'notes')"""
    grid = QGridLayout()
    grid.setHorizontalSpacing(24)
    grid.setVerticalSpacing(10)
    values = {}
    for i, (key, title) in enumerate(INFO_FIELDS):
        row, col = i // 2, (i % 2) * 2
        title_label = QLabel(title)
        title_label.setStyleSheet("color:#6b7280; font-size:12px; border:none;")
        value_label = QLabel('-')
        value_label.setStyleSheet("color:#111827; font-size:13px; font-weight:600; border:none;")
        value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(title_label, row, col)
        grid.addWidget(value_label, row, col + 1)
        values[key] = value_label
    grid.setColumnMinimumWidth(0, 110)
    grid.setColumnMinimumWidth(2, 110)
    grid.setColumnStretch(1, 1)
    grid.setColumnStretch(3, 1)
    layout.addLayout(grid)

    notes_title = QLabel("특이 사항")
    notes_title.setStyleSheet("color:#6b7280; font-size:12px; border:none;")
    layout.addWidget(notes_title)
    notes = QLabel('-')
    notes.setWordWrap(True)
    notes.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    notes.setStyleSheet(
        "background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; "
        "padding:10px 12px; color:#111827; font-size:13px;")
    layout.addWidget(notes)
    values['notes'] = notes
    return values


def draw_info_grid(values, patient):
    for key, _ in INFO_FIELDS:
        values[key].setText(fmt_info(key, patient.get(key)))
    values['notes'].setText(patient.get('notes') or '-')


# ============ 요약 카드 ============
class StatCard(QWidget):
    """흰 카드에 제목(작게) + 큰 숫자 + 보조 설명. 요약 줄에 3개 나란히 놓는다"""

    def __init__(self, title):
        super().__init__()
        self.setObjectName(f"statCard{id(self)}")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"#{self.objectName()} {{ background:#ffffff; border:1px solid #e5e7eb; "
                           f"border-radius:12px; }}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        title_label = QLabel(title)
        title_label.setStyleSheet("color:#6b7280; font-size:12px; font-weight:600; border:none;")
        self.value_label = QLabel('-')
        self.value_label.setStyleSheet("color:#111827; font-size:22px; font-weight:700; border:none;")
        self.sub_label = QLabel('')
        self.sub_label.setStyleSheet("color:#9ca3af; font-size:11px; border:none;")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.sub_label)
        apply_card_shadow(self)

    def set_values(self, value, sub='', color='#111827'):
        self.value_label.setText(value)
        self.value_label.setStyleSheet(f"color:{color}; font-size:22px; font-weight:700; border:none;")
        self.sub_label.setText(sub)


def draw_stat_cards(fall_card, gait_card, stretch_card, detail):
    """요약 카드 3개: 낙상 건수(미처리 강조) / 최근 보행 정상 비율 / 스트레칭 평균 정확도"""
    falls = detail.get('fall_logs') or []
    pending = sum(1 for f in falls if f.get('status') != '처리완료')
    fall_card.set_values(f"{len(falls)}건",
                         f"미처리 {pending}건" if pending else "모두 처리 완료",
                         '#ef4444' if pending else '#111827')

    gaits = detail.get('gait_logs') or []
    if gaits:
        latest = gaits[0]
        abnormal_keys = [k for k in DISEASE_ORDER if k != 'normal']
        top_key = max(abnormal_keys, key=lambda k: float(latest.get(k) or 0))
        gait_card.set_values(f"정상 {float(latest.get('normal') or 0):.0f}%",
                             f"{fmt_dt(latest.get('checking_at'))} · 이상 최다 {DISEASE_LABEL.get(top_key, top_key)}")
    else:
        gait_card.set_values('-', '측정 기록 없음')

    stretches = detail.get('stretch_logs') or []
    if stretches:
        scores = [float(s.get('accuracy_id') or 0) for s in stretches]
        avg = sum(scores) / len(scores)
        stretch_card.set_values(f"{avg:.1f}점", f"최근 {len(scores)}회 평균",
                                '#2f9e44' if avg >= 70 else '#f59e0b' if avg >= 50 else '#ef4444')
    else:
        stretch_card.set_values('-', '측정 기록 없음')


# ============ 기록 표 ============
def make_table(headers, height=200):
    """읽기 전용 기록 표 — 헤더 회색, 행 구분선만, 선택/편집 없음"""
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.verticalHeader().hide()
    table.setShowGrid(False)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setHighlightSections(False)
    table.setFixedHeight(height)
    table.setStyleSheet(
        "QTableWidget { background:#ffffff; border:1px solid #e5e7eb; border-radius:8px; "
        "font-size:12px; color:#111827; }"
        "QTableWidget::item { border-bottom:1px solid #f3f4f6; padding:0 6px; }"
        "QHeaderView::section { background:#f9fafb; color:#6b7280; font-size:12px; font-weight:600; "
        "border:none; border-bottom:1px solid #e5e7eb; padding:6px; }")
    return table


def _cell(text, align_right=False):
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
    if align_right:
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def _pill_cell(text, color):
    """표 안에 넣는 알약 뱃지 (세로 가운데 정렬용으로 한 번 감싼다)"""
    wrapper = QWidget()
    layout = QHBoxLayout(wrapper)
    layout.setContentsMargins(6, 0, 6, 0)
    badge = QLabel(text)
    badge.setFixedHeight(20)
    badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
    # style_pill_badge는 높이 24 기준(radius 12)이라, 표 행에 맞춘 높이 20용으로 radius 10
    badge.setStyleSheet(f"background:{color}; color:#ffffff; border:none; border-radius:10px; "
                        f"padding:0 10px; font-size:11px; font-weight:600;")
    layout.addWidget(badge)
    layout.addStretch()
    return wrapper


def draw_fall_table(table, fall_logs):
    """낙상 기록: 주의 시각 | 낙상 확정 | 상태(뱃지) | 처리 시각 | 카메라"""
    table.setRowCount(len(fall_logs))
    for row, log in enumerate(fall_logs):
        table.setRowHeight(row, 32)
        table.setItem(row, 0, _cell(fmt_dt(log.get('checking_at'))))
        table.setItem(row, 1, _cell(fmt_dt(log.get('alert_at'))))
        status = log.get('status') or '-'
        table.setCellWidget(row, 2, _pill_cell(status, FALL_STATUS_BG.get(status, '#9ca3af')))
        table.setItem(row, 3, _cell(fmt_dt(log.get('resolved_at'))))
        table.setItem(row, 4, _cell(log.get('camera_id') or '-'))


def draw_gait_table(table, gait_logs):
    """보행 기록: 측정 시각 | 정상 | 최다 이상 유형 | 카메라"""
    table.setRowCount(len(gait_logs))
    for row, log in enumerate(gait_logs):
        table.setRowHeight(row, 30)
        abnormal_keys = [k for k in DISEASE_ORDER if k != 'normal']
        top_key = max(abnormal_keys, key=lambda k: float(log.get(k) or 0))
        table.setItem(row, 0, _cell(fmt_dt(log.get('checking_at'))))
        table.setItem(row, 1, _cell(f"{float(log.get('normal') or 0):.0f}%", align_right=True))
        table.setItem(row, 2, _cell(f"{DISEASE_LABEL.get(top_key, top_key)} "
                                    f"{float(log.get(top_key) or 0):.0f}%"))
        table.setItem(row, 3, _cell(log.get('camera_id') or '-'))


def draw_stretch_table(table, stretch_logs):
    """스트레칭 기록: 측정 시각 | 운동 | 정확도(색) | 카메라"""
    table.setRowCount(len(stretch_logs))
    for row, log in enumerate(stretch_logs):
        table.setRowHeight(row, 30)
        sid = log.get('straching_id')
        score = float(log.get('accuracy_id') or 0)
        table.setItem(row, 0, _cell(fmt_dt(log.get('checking_at'))))
        table.setItem(row, 1, _cell(STRETCH_NAMES.get(sid, f"{sid}번" if sid is not None else '-')))
        score_item = _cell(f"{score:.1f}점", align_right=True)
        score_item.setForeground(QColor('#2f9e44' if score >= 70 else '#f59e0b' if score >= 50 else '#ef4444'))
        table.setItem(row, 2, score_item)
        table.setItem(row, 3, _cell(log.get('camera_id') or '-'))


# ============ 보행 분포 막대 ============
class GaitDistributionBar(QWidget):
    """최근 보행 측정 1건의 질환별 비율을 가로 누적 막대 + 범례로 그림.
    색/순서는 보행 탭 도넛 차트(SERIES_COLORS_LIGHT, DISEASE_ORDER)와 같다."""

    BAR_H = 18

    def __init__(self):
        super().__init__()
        self.scores = {}
        self.setMinimumHeight(self.BAR_H + 52)

    def set_scores(self, scores):
        self.scores = {k: max(0.0, float(scores.get(k) or 0)) for k in DISEASE_ORDER}
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self.width()
        total = sum(self.scores.values())

        # 막대 배경
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor('#e5e7eb'))
        painter.drawRoundedRect(QRectF(0, 0, width, self.BAR_H), 9, 9)
        if total <= 0:
            painter.setPen(QColor('#9ca3af'))
            painter.drawText(QRectF(0, self.BAR_H + 6, width, 20),
                             Qt.AlignmentFlag.AlignLeft, "측정 기록 없음")
            return

        # 누적 막대 (양 끝만 둥글게 보이도록 전체를 클립)
        x = 0.0
        painter.save()
        clip = QRectF(0, 0, width, self.BAR_H)
        painter.setClipRect(clip)
        for i, key in enumerate(DISEASE_ORDER):
            w = width * self.scores[key] / total
            if w <= 0:
                continue
            painter.setBrush(QColor(SERIES_COLORS_LIGHT[i % len(SERIES_COLORS_LIGHT)]))
            painter.drawRect(QRectF(x, 0, w + 0.5, self.BAR_H))
            x += w
        painter.restore()

        # 범례: 색 점 + 이름 + % (3개씩 2줄)
        font = QFont()
        font.setPixelSize(12)
        painter.setFont(font)
        col_w = width / 3
        for i, key in enumerate(DISEASE_ORDER):
            cx = (i % 3) * col_w
            cy = self.BAR_H + 10 + (i // 3) * 20
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(SERIES_COLORS_LIGHT[i % len(SERIES_COLORS_LIGHT)]))
            painter.drawEllipse(QRectF(cx, cy + 3, 9, 9))
            painter.setPen(QColor('#374151'))
            pct = self.scores[key] / total * 100
            painter.drawText(QRectF(cx + 14, cy, col_w - 14, 16),
                             Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                             f"{DISEASE_LABEL.get(key, key)} {pct:.0f}%")
# admin_gui/ui_kit.py
"""탭 3개(낙상/보행/스트레칭)가 공통으로 쓰는 작은 스타일 헬퍼 모음.

원래 "회색 섹션 패널"(낙상 탭의 메시지 알림 패널)이나 "알약(pill) 모양 상태
뱃지"(낙상 탭의 정상/확인 필요/낙상 감지 뱃지) 스타일이 fall_tab.py 안에만
있었는데, 보행/스트레칭 탭에도 같은 스타일을 적용해달라는 요청(스타일 통일
작업)에 따라 여기로 빼서 세 탭이 공유한다. 로직은 없고 순수 스타일 헬퍼만
모아뒀다 — 기존 동작/데이터 흐름은 전혀 건드리지 않는다.
"""
from .qt_compat import (
    Qt, QColor, QFrame, QGraphicsDropShadowEffect, QLabel, QVBoxLayout, QWidget,
)

_SECTION_COUNTER = 0


def make_section_panel(title=None, shadow=True):
    """회색(#f3f4f6) 둥근 모서리 패널 하나를 만들어 (panel, content_layout, title_label)을
    반환한다. 낙상 탭의 "메시지 알림" 패널과 동일한 스타일.

    title을 주면 굵은 제목 라벨을 맨 위에 넣어준다(값은 title_label로 돌려주므로
    나중에 setText()로 내용을 갱신할 수 있다 — 예: 스트레칭 탭의 "기준 동작 — OO").
    title이 없으면 title_label은 None이다. content_layout에 그 아래로 위젯/레이아웃을
    쌓으면 된다. shadow=True(기본)면 카드형 위젯과 같은 은은한 그림자를 같이 준다."""
    global _SECTION_COUNTER
    _SECTION_COUNTER += 1
    object_name = f"sectionPanel{_SECTION_COUNTER}"

    panel = QWidget()
    panel.setObjectName(object_name)
    panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    panel.setStyleSheet(f"#{object_name} {{ background:#f3f4f6; border-radius:14px; }}")

    outer = QVBoxLayout(panel)
    outer.setContentsMargins(16, 14, 16, 14)
    outer.setSpacing(10)

    title_label = None
    if title:
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size:14px; font-weight:700; color:#111827; border:none;")
        outer.addWidget(title_label)

    content = QVBoxLayout()
    content.setSpacing(10)
    outer.addLayout(content)

    if shadow:
        apply_card_shadow(panel)

    return panel, content, title_label


def apply_card_shadow(widget):
    """카드형 위젯(낙상 카메라 박스 등)에 쓰는 것과 동일한 은은한 그림자."""
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(20)
    shadow.setColor(QColor(0, 0, 0, 40))
    shadow.setOffset(0, 4)
    widget.setGraphicsEffect(shadow)


def style_pill_badge(label, bg_color, text_color='#ffffff'):
    """상태를 알약 모양 뱃지로 보여줄 때 쓰는 공통 스타일 — 낙상 탭의 정상/확인
    필요/낙상 감지 뱃지와 동일한 모양(둥근 모서리 12px, 좌우 패딩, 높이 24px)."""
    label.setFixedHeight(24)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet(
        f"background:{bg_color}; color:{text_color}; border:none; "
        f"border-radius:12px; padding:0 12px; font-size:12px; font-weight:600;"
    )


def style_selectable_list(list_widget):
    """검색/조회 결과가 카드처럼 하나씩 쌓이는 느낌의 QListWidget 스타일
    (보행/스트레칭 탭의 환자 선택·스트레칭 선택 목록에 사용). 항목 하나하나가
    흰색 카드로 보이고, 선택된 항목은 파란 테두리+옅은 파란 배경으로 강조된다
    (낙상 탭의 메시지 알림 카드와 같은 시각 언어)."""
    list_widget.setSpacing(6)
    list_widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    list_widget.setFrameShape(QFrame.Shape.NoFrame)
    list_widget.setStyleSheet(
        "QListWidget { background:transparent; border:none; }"
        "QListWidget::item { background:#ffffff; border:1px solid #e5e7eb; "
        "border-radius:10px; padding:8px 10px; }"
        "QListWidget::item:selected { background:#eff6ff; border:1.5px solid #2563eb; }"
        "QListWidget::item:hover:!selected { border-color:#93c5fd; }"
    )


def build_patient_card(patient):
    """환자 선택 목록의 카드형 항목 위젯 — 이름(+나이가 있으면 같이) 굵게 위,
    병실·환자ID 옅은 회색으로 아래. QListWidgetItem.setSizeHint()와
    QListWidget.setItemWidget()으로 항목에 끼워 넣어서 쓴다.

    age 필드는 db_client.Patient에 아직 없을 수 있어(확인 전) getattr로 안전하게
    조회한다 — 있으면 "이름 · 나이세"로, 없으면 이름만 보여준다."""
    widget = QWidget()
    widget.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    widget.setStyleSheet("background:transparent;")
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(2, 0, 2, 0)
    layout.setSpacing(2)

    age = getattr(patient, 'age', None)
    top_text = f"{patient.name} · {age}세" if age is not None else patient.name
    top = QLabel(top_text)
    top.setStyleSheet("font-size:13px; font-weight:600; color:#111827; border:none; background:transparent;")
    layout.addWidget(top)

    room = getattr(patient, 'room', '') or ''
    pid = getattr(patient, 'patient_id', '') or ''
    sub_text = ' · '.join(str(v) for v in (room, pid) if v)
    if sub_text:
        bottom = QLabel(sub_text)
        bottom.setStyleSheet("font-size:11px; color:#9ca3af; border:none; background:transparent;")
        layout.addWidget(bottom)

    return widget

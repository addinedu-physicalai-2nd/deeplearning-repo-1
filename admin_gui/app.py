# admin_gui/app.py
"""GUI 엔트리포인트 — 낙상/보행/스트레칭 3탭 구조 (목업 gui_mockup_v5.html 기준).

기존 admin_gui.py 초안(형의 테스트용 코드, 모니터링+스트레칭 2탭 구조)과
달리, 실제로 쓸 구조인 3탭으로 재구성했다. 네트워크/그리기 로직은
network.py / drawing.py로 그대로 옮겼고, 이 파일은 그것들을 조립해서
화면을 구성하는 역할만 한다.

실행: python -m admin_gui.app [--stretch-dir stretching]

작업 지시서 규칙 준수:
  - 수신 스레드(video_receiver/result_receiver)는 Qt 위젯을 절대 직접 건드리지
    않는다. 화면 갱신은 전부 QTimer 기반 update_views()에서만 일어난다.
  - FrameStore.pop / 결과-프레임 매칭 로직은 수정하지 않았다 (network.py 그대로).
"""
import argparse
import json
import queue
import threading

import cv2
import numpy as np

from . import db_stub
from . import patients as patients_module
from .drawing import draw_fall, draw_gait
from .fall_tab import FallTab
from .gait_tab import GaitTab
from .network import FrameStore, MainLink, result_receiver, video_receiver
from .qt_compat import (Qt, QApplication, QFont, QHBoxLayout, QLabel, QMainWindow,
                        QTabWidget, QTimer, QVBoxLayout, QWidget)
from .stretch_tab import StretchingTab

UPDATE_INTERVAL_MS = 30

# 목업(gui_mockup_v5.html)과 맞춘 전역 스타일 — 탭 밑줄 강조, 카드 톤 배경 등.
# 참고: OS가 그리는 실제 창 타이틀바(맨 위 제목줄)는 여기서 손댈 수 없다 —
# 목업의 상단 바는 브라우저가 그린 가짜 macOS 창틀이라 실제 앱 창틀과는 다르다.
#
# 주의: 예전엔 "QMainWindow, QWidget { background: #f7f8fa; ... }"처럼 QWidget에
# 배경색을 통째로 줬었는데, QLabel도 QWidget의 하위 클래스라 스타일시트가 상속되면서
# 라벨/컨테이너마다 의도치 않은 회색 사각형 배경이 찍히는 버그가 있었다(낙상 탭 카드
# 하단의 빈 회색 박스 등). 배경은 QMainWindow/전용 컨테이너에만 주고, QWidget에는
# 폰트만 주고, QLabel은 명시적으로 투명 처리해서 막는다.
STYLE_SHEET = """
QMainWindow, #rootContainer {
    background: #f7f8fa;
}
QWidget {
    font-family: "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
    font-size: 13px;
    color: #111827;
}
QLabel {
    background: transparent;
}
#brandHeader {
    background: #ffffff;
    border-bottom: 1px solid #e5e7eb;
}
#brandIcon {
    background: #eef1ff;
    border-radius: 20px;
}
QTabWidget::pane {
    border: none;
    border-top: 1px solid #e5e7eb;
    background: #ffffff;
}
QTabBar::tab {
    background: transparent;
    color: #6b7280;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: 600;
    border: none;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: #4f46e5;
    border-bottom: 2px solid #4f46e5;
}
QTabBar::tab:hover:!selected {
    color: #374151;
}
QListWidget {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}
QLineEdit {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 5px 8px;
}
QPushButton {
    background: #4f46e5;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #4338ca;
}
QPushButton:disabled {
    background: #c7c9d9;
}
"""


class AdminWindow(QMainWindow):
    def __init__(self, stretch_dir='stretching'):
        super().__init__()
        self.setWindowTitle("AI 돌봄 모니터링 — Admin GUI")

        self.store = FrameStore()
        self.link = MainLink()
        db_stub.set_link(self.link)
        self.was_connected = False          # main_server 연결 시점을 감지해서 환자 목록 요청
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.fall_tab = FallTab(self.link)
        self.gait_tab = GaitTab(self.link)
        self.stretch_tab = StretchingTab(self.link, stretch_dir)

        # keypoints_px, overall.level, cumulative_scores 같은 필드명은 아직 실제
        # 서버 출력으로 확인 못 한 추측값이라, mode별로 딱 한 번씩만 원본 메시지를
        # 콘솔에 그대로 찍어서 눈으로 확인할 수 있게 해둔다. 확인 끝나면 지워도 됨.
        self._debug_seen_modes = set()

        tabs = QTabWidget()
        tabs.addTab(self.fall_tab, "낙상")
        tabs.addTab(self.gait_tab, "보행")
        tabs.addTab(self.stretch_tab, "스트레칭")

        # 탭 위에 "돌봄" 브랜드 헤더 — 이 창이 돌봄 GUI라는 걸 한눈에 알 수 있게.
        header = QWidget()
        header.setObjectName("brandHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)
        header_layout.setSpacing(10)

        icon_label = QLabel("\U0001F9D3")   # 🧓 — 귀여운 노인 아이콘, 로고처럼 사용
        icon_label.setFixedSize(40, 40)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setObjectName("brandIcon")
        icon_label.setStyleSheet("font-size:20px;")

        title_label = QLabel("돌봄")
        title_label.setFont(QFont('', 20, QFont.Weight.Bold))
        title_label.setStyleSheet("color:#111827;")

        header_layout.addWidget(icon_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        central = QWidget()
        central.setObjectName("rootContainer")
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(header)
        central_layout.addWidget(tabs)
        self.setCentralWidget(central)

        self.video_thread = threading.Thread(
            target=video_receiver, args=(self.store, self.stop_event), daemon=True)
        self.result_thread = threading.Thread(
            target=result_receiver, args=(self.result_queue, self.stop_event, self.link), daemon=True)
        self.video_thread.start()
        self.result_thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_views)
        self.timer.start(UPDATE_INTERVAL_MS)

    def update_views(self):
        """QTimer(30ms)에서만 호출 — 여기서만 Qt 위젯을 갱신한다."""
        connected = self.link.conn is not None
        if connected and not self.was_connected:
            db_stub.request_patients()      # main_server에 붙자마자 DB 환자 목록 요청
        self.was_connected = connected

        while True:
            try:
                msg = self.result_queue.get_nowait()
            except queue.Empty:
                break
            self.dispatch(msg)

    def handle_server_message(self, msg):
        """분석 결과가 아닌 main_server 메시지 (DB 요청 응답, 스트레칭 세션 종료 등)"""
        kind = msg.get('type')
        if kind == 'response':
            cmd = msg.get('cmd')
            if not msg.get('ok'):
                print(f"[GUI] 요청 실패: {cmd} (req_id={msg.get('req_id')})")
                return
            if cmd == 'get_patients':
                patients = patients_module.set_patients(msg.get('data') or [])
                self.gait_tab.set_patients(patients)
                self.stretch_tab.set_patients(patients)
                print(f"[GUI] 환자 {len(patients)}명 로드")
        elif kind == 'stretch_session_end':
            self.stretch_tab.on_session_end(msg)

    def dispatch(self, msg):
        if msg.get('type'):                 # 분석 결과 메시지에는 'type'이 없음
            self.handle_server_message(msg)
            return
        camera_id = msg.get('camera_id')
        mode = msg.get('mode')
        frame_idx = msg.get('frame_idx')
        data = msg.get('data', {})

        if mode not in self._debug_seen_modes:
            self._debug_seen_modes.add(mode)
            print(f"\n[DEBUG] mode={mode} 메시지 원본 최초 1회 출력 (실제 필드명 확인용):")
            print(json.dumps(msg, ensure_ascii=False, indent=2)[:3000])
            print("[DEBUG] (이 mode는 이후로 다시 안 찍힘)\n")

        jpg_bytes = self.store.pop(camera_id, frame_idx)
        if jpg_bytes is None:
            return
        frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        if mode == 0:
            draw_fall(frame, data)
            self.fall_tab.render(camera_id, msg, frame)
        elif mode == 1:
            draw_gait(frame, data)
            self.gait_tab.show_frame(camera_id, frame)
            self.gait_tab.render(camera_id, msg)
        elif mode == 2:
            self.stretch_tab.on_result(camera_id, msg, frame)

    def closeEvent(self, event):
        self.stop_event.set()
        self.stretch_tab.stop()
        super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stretch-dir', default='stretching',
                         help="스트레칭 기준 자세 JSON이 있는 디렉터리 (기본: stretching)")
    args = parser.parse_args()

    app = QApplication([])
    app.setStyleSheet(STYLE_SHEET)
    window = AdminWindow(stretch_dir=args.stretch_dir)
    window.resize(1600, 900)   # 스트레칭 탭 영상이 커져서 창도 키운다
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
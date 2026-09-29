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

from .drawing import draw_fall, draw_gait
from .fall_tab import FallTab
from .gait_tab import GaitTab
from .network import FrameStore, MainLink, result_receiver, video_receiver
from .qt_compat import QApplication, QMainWindow, QTabWidget, QTimer
from .stretch_tab import StretchingTab

UPDATE_INTERVAL_MS = 30

# 목업(gui_mockup_v5.html)과 맞춘 전역 스타일 — 탭 밑줄 강조, 카드 톤 배경 등.
# 참고: OS가 그리는 실제 창 타이틀바(맨 위 제목줄)는 여기서 손댈 수 없다 —
# 목업의 상단 바는 브라우저가 그린 가짜 macOS 창틀이라 실제 앱 창틀과는 다르다.
STYLE_SHEET = """
QMainWindow, QWidget {
    background: #f7f8fa;
    font-family: "Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
    font-size: 13px;
    color: #111827;
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
        self.setCentralWidget(tabs)

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
        while True:
            try:
                msg = self.result_queue.get_nowait()
            except queue.Empty:
                break
            self.dispatch(msg)

    def dispatch(self, msg):
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

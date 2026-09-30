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
import time

import cv2
import numpy as np

from . import db_client
from .drawing import draw_fall, draw_gait
from .fall_tab import FallTab
from .gait_tab import GaitTab
from .network import FrameStore, MainLink, result_receiver, video_receiver
from .qt_compat import (Qt, QApplication, QFont, QHBoxLayout, QLabel, QMainWindow,
                        QTabWidget, QTimer, QVBoxLayout, QWidget)
from .stretch_tab import StretchingTab

UPDATE_INTERVAL_MS = 30
FPS_LOG_SEC = 5.0   # 스트레칭 수신/표시 프레임 수 로그 주기

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
        db_client.set_link(self.link)
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
        # 스트레칭 FPS 계측용 (update_views에서 FPS_LOG_SEC마다 출력 후 초기화)
        self._fps_recv = 0          # 받은 mode=2 결과 수
        self._fps_drawn = 0         # 프레임 매칭 성공해서 on_result까지 간 수
        self._fps_ticks = 0         # mode=2가 1개 이상 들어온 tick 수 (= 체감 FPS 상한)
        self._fps_last = time.monotonic()

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
        """QTimer(30ms)에서만 호출 — 여기서만 Qt 위젯을 갱신한다.

        예전엔 result_queue에 쌓인 메시지를 전부 도착한 순서대로 dispatch()
        했는데, 스트레칭(mode=2) 실시간 화면에 부위별 스켈레톤을 매 프레임
        그리는 처리가 30ms tick 예산보다 오래 걸리면 큐가 계속 쌓이기
        시작한다. 그러면 세션을 막 시작했을 때(큐가 비어있을 때)는 거의
        실시간으로 보이다가, 시간이 지날수록 "밀린 과거 프레임"을 순서대로
        따라잡느라 화면이 갈수록 느려지고 실제 카메라와 점점 벌어진다 —
        "최초(세션 시작 직후)랑 카메라 프레임이 너무 차이난다"는 증상의
        원인이 바로 이 무제한 백로그다.

        낙상(mode 0)/보행(mode 1)과 type이 있는 서버 메시지(DB 응답, 스트레칭
        세션 종료 등, handle_server_message로 감)는 메시지 하나하나가
        로그·알림·상태에 영향을 줄 수 있으므로 지금처럼 전부 순서대로
        처리한다. 스트레칭(mode 2)만 화면엔 카메라별로 "가장 최신" 프레임
        한 장만 보여주면 충분하므로, 같은 배치 안에 같은 카메라의 더 최신
        mode=2 메시지가 있으면 오래된 쪽은 화면에 그리지 않고 건너뛴다.
        다만 FrameStore에는 그대로 두면 메모리가 계속 쌓이므로, 그리지 않는
        프레임도 store.pop()으로 꺼내서 비워만 준다(디코딩/그리기 같은 무거운
        작업만 건너뛴다). type이 있는 메시지는 'mode' 키 자체가 없어서 이
        건너뛰기 대상에 걸리지 않고 항상 그대로 dispatch된다."""
        connected = self.link.conn is not None
        if connected and not self.was_connected:
            db_client.request_patients()    # main_server에 붙자마자 DB 환자 목록 요청
        self.was_connected = connected

        pending = []
        while True:
            try:
                pending.append(self.result_queue.get_nowait())
            except queue.Empty:
                break

        # camera_id별로 이 배치 안에서 가장 마지막(=가장 최신) mode=2 메시지의
        # 인덱스만 기록 — 그 인덱스가 아닌 mode=2 메시지는 오래된 것이므로 버린다.
        latest_stretch_idx = {}
        for i, msg in enumerate(pending):
            if msg.get('mode') == 2:
                latest_stretch_idx[msg.get('camera_id')] = i

        # 스트레칭 FPS 계측 — sender / main_server 로그와 비교해서 병목 위치 확인용
        stretch_count = sum(1 for m in pending if m.get('mode') == 2)
        self._fps_recv += stretch_count
        if stretch_count:
            self._fps_ticks += 1
        now = time.monotonic()
        if now - self._fps_last >= FPS_LOG_SEC:
            sec = now - self._fps_last
            print(f"[GUI] stretch recv {self._fps_recv / sec:.1f}/s, "
                  f"drawn {self._fps_drawn / sec:.1f}/s, update ticks {self._fps_ticks / sec:.1f}/s")
            self._fps_recv = self._fps_drawn = self._fps_ticks = 0
            self._fps_last = now

        for i, msg in enumerate(pending):
            if msg.get('mode') == 2 and latest_stretch_idx.get(msg.get('camera_id')) != i:
                # 이 카메라의 더 최신 mode=2 프레임이 같은 배치 안에 있음 →
                # 이건 이미 낡은 프레임이니 화면엔 안 그리고 FrameStore만 비운다.
                self.store.pop(msg.get('camera_id'), msg.get('frame_idx'))
                continue
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
                patients = db_client.to_patients(msg.get('data') or [])
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
            self._fps_drawn += 1
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
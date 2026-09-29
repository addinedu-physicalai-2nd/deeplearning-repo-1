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

from .branding import build_brand_header
from .drawing import draw_fall, draw_gait
from .fall_tab import FallTab
from .gait_tab import GaitTab
from .network import FrameStore, MainLink, result_receiver, video_receiver
from .qt_compat import (QApplication, QMainWindow, QTabWidget, QTimer,
                        QVBoxLayout, QWidget)
from .stretch_tab import StretchingTab
from .styles import STYLE_SHEET

UPDATE_INTERVAL_MS = 30


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

        # 탭 위에 "돌봄" 브랜드 헤더 — 이 창이 돌봄 GUI라는 걸 한눈에 알 수 있게.
        # (branding.py의 build_brand_header()로 뺐다 — dev_preview.py 미리보기 도구도
        # 똑같은 헤더를 그려야 해서 공용 함수로 통일)
        header = build_brand_header()

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

        낙상(mode 0)/보행(mode 1)은 메시지 하나하나가 로그·알림(낙상 로그,
        환자 상태 등)에 영향을 줄 수 있으므로 지금처럼 전부 순서대로
        처리한다. 스트레칭(mode 2)은 화면엔 카메라별로 "가장 최신" 프레임 한
        장만 보여주면 충분하므로, 같은 배치 안에 같은 카메라의 더 최신
        mode=2 메시지가 있으면 오래된 쪽은 화면에 그리지 않고 건너뛴다.
        다만 FrameStore에는 그대로 두면 메모리가 계속 쌓이므로, 그리지 않는
        프레임도 store.pop()으로 꺼내서 비워만 준다(디코딩/그리기 같은 무거운
        작업만 건너뛴다)."""
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

        for i, msg in enumerate(pending):
            if msg.get('mode') == 2 and latest_stretch_idx.get(msg.get('camera_id')) != i:
                # 이 카메라의 더 최신 mode=2 프레임이 같은 배치 안에 있음 →
                # 이건 이미 낡은 프레임이니 화면엔 안 그리고 FrameStore만 비운다.
                self.store.pop(msg.get('camera_id'), msg.get('frame_idx'))
                continue
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

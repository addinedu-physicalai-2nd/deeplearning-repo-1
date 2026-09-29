# main_service/main_server.py
import argparse
import json
import logging
import threading
import time

from config.settings import CAMERA_PORTS
from .main_networkmanager import MainNetworkManager
from .fall_analyzer import FallAnalyzer
from .stretching_analyzer import StretchingAnalyzer

MODE_FALL = 0
MODE_GAIT = 1
MODE_STRETCH = 2

UPDATE_INTERVAL_SEC = 0.05   # 결과 처리 주기 (20Hz)
STATUS_LOG_SEC = 5.0         # 수신 현황 로그 주기


def load_reference(path):
    """기준 자세 JSON 로드 + frame_index를 0부터 시작하도록 재정렬

    구간 영상(mp4)은 0번 프레임부터 시작하지만, JSON의 frame_index는 원본 영상 기준 번호일 수 있어서
    가장 작은 frame_index를 빼서 영상 프레임 번호와 맞춘다.
    """
    with open(path, 'r', encoding='utf-8') as f:
        reference = json.load(f)
    skeletons = reference.get('skeletons', [])
    offset = min((s['frame_index'] for s in skeletons), default=0)
    reference['skeletons'] = [{**s, 'frame_index': s['frame_index'] - offset} for s in skeletons]
    return reference


class MainService:
    """카메라 프레임 → AI Server, AI 결과 → mode별 분석 → GUI"""

    def __init__(self, default_mode=MODE_FALL, stretch_ref_path=None,
                 ai_host='localhost', gui_host='localhost'):
        self.net = MainNetworkManager(ai_host=ai_host, gui_host=gui_host)

        self.fall_analyzer = FallAnalyzer()
        self.stretching_analyzer = None
        if stretch_ref_path:
            self.stretching_analyzer = StretchingAnalyzer(load_reference(stretch_ref_path))

        # mode별 분석 모듈 (None이면 AI 결과를 그대로 GUI로 전달)
        self.analyzers = {
            MODE_FALL: self.fall_analyzer,
            MODE_GAIT: None,                       # Main 쪽 보행 분석 모듈 미구현
            MODE_STRETCH: self.stretching_analyzer,
        }

        self.modes = {cam_id: default_mode for cam_id in CAMERA_PORTS}
        self.frame_idx = {cam_id: 0 for cam_id in CAMERA_PORTS}        # AI/GUI 매칭용 번호
        self.frame_counter = {cam_id: 0 for cam_id in CAMERA_PORTS}
        self.result_counter = {cam_id: 0 for cam_id in CAMERA_PORTS}
        self.state_lock = threading.Lock()                              # frame_idx 리셋/증가 보호
        self.is_running = False

        self.logger = logging.getLogger('MainService')
        logging.basicConfig(level=logging.INFO)

    def set_mode(self, camera_id, mode):
        """카메라별 분석 모드 변경 (0 낙상 / 1 보행 / 2 스트레칭)"""
        if camera_id in self.modes and mode in (MODE_FALL, MODE_GAIT, MODE_STRETCH):
            self.modes[camera_id] = mode
            self.logger.info(f"[{camera_id}] mode -> {mode}")

    def handle_command(self, cmd):
        """GUI에서 온 명령 처리"""
        name = cmd.get('cmd')
        camera_id = cmd.get('camera_id')
        if camera_id not in self.modes:
            self.logger.warning(f"Command with unknown camera_id: {cmd}")
            return

        if name == 'start_stretching':
            # 기준 영상 재생 시작 → 기준 자세 교체 + frame_idx 0부터 다시 매김
            reference = load_reference(cmd['reference'])
            with self.state_lock:
                self.stretching_analyzer = StretchingAnalyzer(reference)
                self.analyzers[MODE_STRETCH] = self.stretching_analyzer
                self.frame_idx[camera_id] = 0
                self.modes[camera_id] = MODE_STRETCH
            self.logger.info(f"[{camera_id}] stretching start: {reference.get('movement_name', cmd['reference'])} "
                             f"({len(reference['skeletons'])} ref frames), frame_idx reset")

        elif name == 'set_mode':
            self.set_mode(camera_id, cmd.get('mode'))

        else:
            self.logger.warning(f"Unknown command: {name}")

    def process_frame(self, camera_id, jpg_bytes):
        """카메라 프레임 수신 시 호출 → 같은 frame_idx로 AI Server와 GUI에 전송"""
        with self.state_lock:
            frame_idx = self.frame_idx[camera_id]
            self.frame_idx[camera_id] += 1
            mode = self.modes[camera_id]
        self.frame_counter[camera_id] += 1

        self.net.send_frame(camera_id, mode, frame_idx, jpg_bytes)
        self.net.send_video_to_gui(camera_id, frame_idx, jpg_bytes)

    def update_gui(self):
        """AI 결과를 주기적으로 꺼내서 분석 후 GUI로 전송"""
        while self.is_running:
            results = self.net.get_latest_results()   # {camera_id: [aioutput, ...]}

            for camera_id, outputs in results.items():
                for aioutput in outputs:
                    self.result_counter[camera_id] += 1
                    # frame_idx 리셋 전에 보낸 프레임의 결과는 아직 매기지 않은 번호로 돌아옴 → 버림
                    frame_idx = aioutput.get('frame_idx')
                    with self.state_lock:
                        issued = self.frame_idx[camera_id]
                    if frame_idx is None or frame_idx >= issued:
                        continue
                    try:
                        gui_data = self._analyze(aioutput)
                    except Exception as e:
                        self.logger.error(f"[{camera_id}] Analyze error (mode={aioutput.get('mode')}): {e}")
                        continue
                    if gui_data is None:
                        continue
                    self.send_to_gui(camera_id, aioutput, gui_data)

            time.sleep(UPDATE_INTERVAL_SEC)

    def _analyze(self, aioutput):
        mode = aioutput.get('mode')
        analyzer = self.analyzers.get(mode)
        if analyzer is None:
            return {
                'frame_idx': aioutput.get('frame_idx'),
                'detections': aioutput.get('detections', []),
            }
        gui_data = analyzer.analyze(aioutput)

        if mode == MODE_STRETCH and gui_data:
            # GUI가 영상 위에 스켈레톤을 그릴 수 있게 원본 픽셀 좌표 추가
            pixel_kps = {det['track_id']: det['raw_data'].get('keypoints_px')
                         for det in aioutput.get('detections', [])}
            for track_id, info in gui_data.get('tracking_data', {}).items():
                info['keypoints_px'] = pixel_kps.get(track_id)
        return gui_data

    def send_to_gui(self, camera_id, aioutput, gui_data):
        message = {
            'camera_id': camera_id,
            'mode': aioutput.get('mode'),
            'frame_idx': aioutput.get('frame_idx'),
            'timestamp': aioutput.get('timestamp'),
            'data': gui_data,
        }
        self.net.send_to_gui(message)

    def _log_status(self):
        active = []
        for cam_id in CAMERA_PORTS:
            frames = self.frame_counter[cam_id]
            results = self.result_counter[cam_id]
            if frames or results:
                active.append(f"{cam_id} frames={frames} results={results}")
            self.frame_counter[cam_id] = 0
            self.result_counter[cam_id] = 0
        self.logger.info(" | ".join(active) if active else "no camera frames / AI results")

    def run(self):
        self.is_running = True
        self.net.start(on_frame=self.process_frame, on_command=self.handle_command)
        threading.Thread(target=self.update_gui, daemon=True).start()
        self.logger.info("MainService started")

        try:
            while True:
                time.sleep(STATUS_LOG_SEC)
                self._log_status()
        except KeyboardInterrupt:
            self.logger.info("Stopping MainService")
        finally:
            self.is_running = False
            self.net.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=MODE_FALL, choices=[0, 1, 2],
                        help='모든 카메라의 시작 모드 (0 낙상 / 1 보행 / 2 스트레칭)')
    parser.add_argument('--stretch-ref', default=None,
                        help='스트레칭 기준 자세 JSON 경로 (없으면 mode 2는 AI 결과 그대로 전달)')
    parser.add_argument('--ai-host', default='localhost')
    parser.add_argument('--gui-host', default='localhost')
    args = parser.parse_args()

    service = MainService(
        default_mode=args.mode,
        stretch_ref_path=args.stretch_ref,
        ai_host=args.ai_host,
        gui_host=args.gui_host,
    )
    service.run()
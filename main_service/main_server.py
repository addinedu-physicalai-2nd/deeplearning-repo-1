# main_service/main_server.py
import argparse
import json
import logging
import queue
import re
import threading
import time
from pathlib import Path

from config.settings import CAMERA_PORTS
from .main_networkmanager import MainNetworkManager
from .db_manager import DBManager
from .fall_analyzer import FallAnalyzer
from .stretching_analyzer import StretchingAnalyzer

MODE_FALL = 0
MODE_GAIT = 1
MODE_STRETCH = 2

UPDATE_INTERVAL_SEC = 0.01   # 결과 처리 주기 (100Hz) — 50ms 배치 대기로 생기는 지연/몰림 감소
STATUS_LOG_SEC = 5.0         # 수신 현황 로그 주기
DEFAULT_REF_FPS = 15.0       # GUI가 ref_fps를 안 보냈을 때 쓰는 기준영상 FPS


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


def parse_course(reference_path):
    """'.../05_목운동_skeleton.json' → (5, '목운동'). 형식이 다르면 (None, 파일 이름)"""
    stem = Path(reference_path).stem
    if stem.endswith('_skeleton'):
        stem = stem[:-len('_skeleton')]
    match = re.match(r'^(\d+)_(.+)$', stem)
    if match:
        return int(match.group(1)), match.group(2)
    return None, stem


class MainService:
    """카메라 프레임 → AI Server, AI 결과 → mode별 분석 → GUI"""

    def __init__(self, default_mode=MODE_FALL, stretch_ref_path=None,
                 ai_host='localhost', gui_host='localhost', use_db=True):
        self.net = MainNetworkManager(ai_host=ai_host, gui_host=gui_host)
        self.db = DBManager(enabled=use_db)

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

        self.default_mode = default_mode
        self.modes = {cam_id: default_mode for cam_id in CAMERA_PORTS}
        self.frame_idx = {cam_id: 0 for cam_id in CAMERA_PORTS}        # AI/GUI 매칭용 번호
        self.frame_counter = {cam_id: 0 for cam_id in CAMERA_PORTS}
        self.result_counter = {cam_id: 0 for cam_id in CAMERA_PORTS}
        self.state_lock = threading.Lock()                              # frame_idx 리셋/증가 보호
        self.stretch_sessions = {}          # camera_id → 진행 중인 스트레칭 세션 (점수 누적, DB 저장용)
        self.pending_resolves = queue.Queue()   # 낙상 처리완료 요청 → 결과 처리 스레드에서 FallAnalyzer에 반영
        self.is_running = False

        self.logger = logging.getLogger('MainService')
        logging.basicConfig(level=logging.INFO)

    def set_mode(self, camera_id, mode):
        """카메라별 분석 모드 변경 (0 낙상 / 1 보행 / 2 스트레칭)"""
        if camera_id in self.modes and mode in (MODE_FALL, MODE_GAIT, MODE_STRETCH):
            self.modes[camera_id] = mode
            self.logger.info(f"[{camera_id}] mode -> {mode}")

    def handle_command(self, cmd):
        """GUI에서 온 명령 처리 (GUI 명령 수신 스레드에서 실행)"""
        name = cmd.get('cmd')

        # ---- DB 요청: 결과를 같은 req_id로 GUI에 응답 ----
        if name == 'get_patients':
            patients = self.db.get_patients()
            self._reply(cmd, patients is not None, patients or [])
            return
        if name == 'get_patient_detail':
            detail = self.db.get_patient_detail(cmd.get('patient_id'))
            self._reply(cmd, detail is not None, detail)
            return
        if name == 'save_gait_session':
            ok = self.db.save_gait_session(cmd.get('patient_id'), cmd.get('cumulative_scores') or {},
                                           camera_id=cmd.get('camera_id'))
            self._reply(cmd, ok)
            return

        # ---- 카메라 제어 명령 ----
        camera_id = cmd.get('camera_id')
        if camera_id not in self.modes:
            self.logger.warning(f"Command with unknown camera_id: {cmd}")
            return

        if name == 'start_stretching':
            # 기준 영상 재생 시작 → 기준 자세 교체 + frame_idx 0부터 다시 매김 + 세션 시작
            reference = load_reference(cmd['reference'])
            course_id, course_name = parse_course(cmd['reference'])
            patient_id = cmd.get('patient_id')
            if patient_id is None:
                patient = self.db.get_patient_by_camera(camera_id)
                patient_id = patient['patient_id'] if patient else None
            last_idx = max((s['frame_index'] for s in reference['skeletons']), default=0)
            try:
                ref_fps = float(cmd.get('ref_fps') or DEFAULT_REF_FPS)
            except (TypeError, ValueError):
                ref_fps = DEFAULT_REF_FPS

            with self.state_lock:
                stopped = self.stretch_sessions.pop(camera_id, None)
                self.stretching_analyzer = StretchingAnalyzer(reference)
                self.analyzers[MODE_STRETCH] = self.stretching_analyzer
                self.frame_idx[camera_id] = 0
                self.modes[camera_id] = MODE_STRETCH
                # 기준 인덱스는 카메라 프레임 개수가 아니라 "세션 시작 후 경과 시간 × 기준영상 FPS"로
                # 매긴다 — 카메라(15fps)와 기준영상(예: 23.976fps)의 FPS가 달라도 GUI 좌측 영상과
                # 같은 시계로 비교/종료되게. ref_idx_map: frame_idx → 그 프레임이 도착한 시점의 ref_idx
                self.stretch_sessions[camera_id] = {
                    'patient_id': patient_id, 'course_id': course_id, 'course_name': course_name,
                    'last_idx': last_idx, 'scores': [],
                    'start_time': time.monotonic(), 'ref_fps': ref_fps, 'ref_idx_map': {},
                }
            self._finish_stretch_session(camera_id, stopped, completed=False)
            self.logger.info(f"[{camera_id}] stretching start: {course_name} ({len(reference['skeletons'])} ref frames "
                             f"@ {ref_fps:.3f}fps, "
                             f"patient={patient_id}), frame_idx reset")

        elif name == 'set_mode':
            if cmd.get('mode') != MODE_STRETCH:
                with self.state_lock:
                    stopped = self.stretch_sessions.pop(camera_id, None)
                self._finish_stretch_session(camera_id, stopped, completed=False)
            self.set_mode(camera_id, cmd.get('mode'))

        elif name == 'resolve_fall':
            # FallAnalyzer는 결과 처리 스레드만 건드리게 큐로 넘기고, DB 기록은 여기서 바로
            self.pending_resolves.put((camera_id, cmd.get('track_id')))
            ok = self.db.resolve_fall_event(camera_id, cmd.get('track_id'), cmd.get('caregiver_name'))
            self._reply(cmd, ok)

        else:
            self.logger.warning(f"Unknown command: {name}")

    def _reply(self, cmd, ok, data=None):
        """GUI 요청에 대한 응답. 분석 결과 메시지와 구분되게 'type': 'response'를 붙임"""
        self.net.send_to_gui({
            'type': 'response',
            'cmd': cmd.get('cmd'),
            'req_id': cmd.get('req_id'),
            'ok': ok,
            'data': data,
        })

    # ============ 스트레칭 세션 ============
    def _finish_stretch_session(self, camera_id, session, completed):
        """세션 마무리 (state_lock 밖에서 호출 — DB 저장/전송이 카메라 스레드를 막지 않게).
        기준 영상을 끝까지 했으면(completed) 평균 점수를 DB에 저장하고 GUI에 알림"""
        if session is None:
            return
        scores = session['scores']
        avg = sum(scores) / len(scores) if scores else None
        saved = False
        if completed and avg is not None and session['patient_id'] is not None:
            saved = self.db.save_stretch_session(session['patient_id'], camera_id, session['course_id'], avg)
        self.logger.info(f"[{camera_id}] stretching {'done' if completed else 'stopped'}: {session['course_name']} "
                         f"avg={avg if avg is None else round(avg, 1)} ({len(scores)} frames), saved={saved}")
        self.net.send_to_gui({
            'type': 'stretch_session_end',
            'camera_id': camera_id,
            'course_id': session['course_id'],
            'course_name': session['course_name'],
            'completed': completed,
            'avg_score': avg,
            'frames_scored': len(scores),
            'saved': saved,
        })

    def _track_stretch_score(self, camera_id, ref_idx, gui_data):
        """스트레칭 결과 1프레임을 세션에 누적. 기준 영상 마지막 프레임을 지나면 세션 완료 + 기본 모드로 복귀"""
        finished = None
        with self.state_lock:
            session = self.stretch_sessions.get(camera_id)
            if session is None:
                return
            tracking = (gui_data or {}).get('tracking_data') or {}
            if tracking:
                first = next(iter(tracking.values()))       # 화면에는 한 명만 표시하므로 첫 번째 사람 기준
                session['scores'].append(first['overall']['score'])
            if ref_idx >= session['last_idx']:
                finished = self.stretch_sessions.pop(camera_id)
                self.modes[camera_id] = self.default_mode
        self._finish_stretch_session(camera_id, finished, completed=True)

    def process_frame(self, camera_id, jpg_bytes):
        """카메라 프레임 수신 시 호출 → 같은 frame_idx로 AI Server와 GUI에 전송"""
        with self.state_lock:
            frame_idx = self.frame_idx[camera_id]
            self.frame_idx[camera_id] += 1
            mode = self.modes[camera_id]
            session = self.stretch_sessions.get(camera_id)
            if mode == MODE_STRETCH and session is not None:
                # 프레임이 main에 도착한 시각 기준으로 기준영상의 몇 번째 프레임인지 기록
                elapsed = time.monotonic() - session['start_time']
                session['ref_idx_map'][frame_idx] = int(elapsed * session['ref_fps'])
        self.frame_counter[camera_id] += 1

        self.net.send_frame(camera_id, mode, frame_idx, jpg_bytes)
        self.net.send_video_to_gui(camera_id, frame_idx, jpg_bytes)

    def update_gui(self):
        """AI 결과를 주기적으로 꺼내서 분석 후 GUI로 전송"""
        while self.is_running:
            # GUI에서 온 낙상 처리완료 → FallAnalyzer 반영 (분석기는 이 스레드에서만 건드림)
            while not self.pending_resolves.empty():
                camera_id, track_id = self.pending_resolves.get_nowait()
                self.fall_analyzer.resolve(camera_id, track_id)

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
                    ref_idx = None
                    if aioutput.get('mode') == MODE_STRETCH:
                        with self.state_lock:
                            session = self.stretch_sessions.get(camera_id)
                            if session is not None:
                                ref_idx = session['ref_idx_map'].pop(frame_idx, None)
                        if ref_idx is None:
                            continue    # 세션 밖(시작 전/종료 후) 프레임의 결과 → 버림
                        # StretchingAnalyzer는 aioutput['frame_idx']로 기준 스켈레톤을 찾으므로
                        # 분석에 쓰는 사본에만 ref_idx를 넣는다 (원본 frame_idx는 GUI 영상 매칭용)
                        analyze_input = {**aioutput, 'frame_idx': ref_idx}
                    else:
                        analyze_input = aioutput
                    try:
                        gui_data = self._analyze(analyze_input)
                    except Exception as e:
                        self.logger.error(f"[{camera_id}] Analyze error (mode={aioutput.get('mode')}): {e}")
                        continue
                    if gui_data is None:
                        continue
                    self.send_to_gui(camera_id, aioutput, gui_data, ref_idx)

                    mode = aioutput.get('mode')
                    if mode == MODE_FALL:
                        for event in gui_data.get('events', []):
                            self.db.log_fall_event(camera_id, event.get('track_id'), event.get('event', ''))
                    elif mode == MODE_STRETCH:
                        self._track_stretch_score(camera_id, ref_idx, gui_data)

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

    def send_to_gui(self, camera_id, aioutput, gui_data, ref_idx=None):
        message = {
            'camera_id': camera_id,
            'mode': aioutput.get('mode'),
            'frame_idx': aioutput.get('frame_idx'),
            'timestamp': aioutput.get('timestamp'),
            'data': gui_data,
        }
        if ref_idx is not None:
            message['ref_idx'] = ref_idx    # 스트레칭: 비교에 쓴 기준영상 프레임 번호 (GUI 종료 판단용)
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
        self.db.check_connection()
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
            self.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, default=MODE_FALL, choices=[0, 1, 2],
                        help='모든 카메라의 시작 모드 (0 낙상 / 1 보행 / 2 스트레칭)')
    parser.add_argument('--stretch-ref', default=None,
                        help='스트레칭 기준 자세 JSON 경로 (없으면 mode 2는 AI 결과 그대로 전달)')
    parser.add_argument('--ai-host', default='localhost')
    parser.add_argument('--gui-host', default='localhost')
    parser.add_argument('--no-db', action='store_true', help='DB 없이 실행 (환자 목록 비어 있음, 기록 저장 안 함)')
    args = parser.parse_args()

    service = MainService(
        default_mode=args.mode,
        stretch_ref_path=args.stretch_ref,
        ai_host=args.ai_host,
        gui_host=args.gui_host,
        use_db=not args.no_db,
    )
    service.run()
# main_service/main_networkmanager.py
import base64
import json
import socket
import threading
import time
import logging
import struct
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from config.settings import (
    CAMERA_PORTS,
    AI_SERVER_PORT,
    MAIN_SERVICE_PORT,
    GUI_INFO_PORT,
    GUI_VIDEO_PORT,
)

UDP_MAX_BYTES = 65000        # UDP 데이터그램 한계(65507)보다 약간 작게
RESULT_QUEUE_LEN = 60        # 카메라별 미처리 AI 결과 최대 보관 개수
GUI_RETRY_SEC = 2.0          # GUI 연결 실패 시 재시도 간격


def _to_json_safe(obj):
    """json.dumps가 못 바꾸는 numpy 타입 처리"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


class MainNetworkManager:
    """Main Service 네트워크 통신

    - 카메라 → Main : UDP  (CAMERA_PORTS, JPEG 바이트 그대로)
    - Main  → AI   : UDP  (AI_SERVER_PORT, JSON {camera_id, mode, frame(base64)})
    - AI    → Main : HTTP POST /ai_result (MAIN_SERVICE_PORT)
    - Main  → GUI  : TCP  (GUI_INFO_PORT, 한 줄에 JSON 하나)
    - Main  → GUI  : UDP  (GUI_VIDEO_PORT, [2바이트 헤더 길이][JSON 헤더][JPEG])
    - GUI   → Main : 같은 TCP 연결로 명령 수신 (한 줄에 JSON 하나)
    """

    def __init__(self, ai_host='localhost', ai_port=AI_SERVER_PORT,
                 main_port=MAIN_SERVICE_PORT,
                 gui_host='localhost', gui_port=GUI_INFO_PORT,
                 gui_video_port=GUI_VIDEO_PORT):
        self.ai_host = ai_host
        self.ai_port = ai_port
        self.main_port = main_port
        self.gui_host = gui_host
        self.gui_port = gui_port
        self.gui_video_port = gui_video_port

        # camera_id별 미처리 AI 결과 (도착 순서 유지)
        self.result_buffer = {cam_id: deque(maxlen=RESULT_QUEUE_LEN) for cam_id in CAMERA_PORTS}
        self.lock = threading.Lock()

        self.ai_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.gui_video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.gui_sock = None
        self.gui_lock = threading.Lock()
        self.gui_next_retry = 0.0
        self.gui_warned = False

        self.on_frame = None
        self.on_command = None
        self.http_server = None
        self.is_running = False

        self.logger = logging.getLogger('MainNetworkManager')
        logging.basicConfig(level=logging.INFO)

    # ============ 시작 / 종료 ============
    def start(self, on_frame, on_command=None):
        """카메라 UDP 수신 + /ai_result HTTP 서버 시작

        on_frame(camera_id, jpg_bytes): 카메라 프레임 도착 시 호출할 콜백
        on_command(cmd: dict): GUI에서 명령이 왔을 때 호출할 콜백
        """
        self.on_frame = on_frame
        self.on_command = on_command
        self.is_running = True

        for camera_id, port in CAMERA_PORTS.items():
            threading.Thread(
                target=self._camera_loop, args=(camera_id, port), daemon=True
            ).start()
            self.logger.info(f"{camera_id} listening on UDP port {port}")

        self.http_server = ThreadingHTTPServer(('0.0.0.0', self.main_port), self._make_handler())
        threading.Thread(target=self.http_server.serve_forever, daemon=True).start()
        self.logger.info(f"AI result server on http://0.0.0.0:{self.main_port}/ai_result")

    def stop(self):
        self.is_running = False
        if self.http_server is not None:
            self.http_server.shutdown()
        self.ai_sock.close()
        self.gui_video_sock.close()
        with self.gui_lock:
            if self.gui_sock is not None:
                self.gui_sock.close()
                self.gui_sock = None
        self.logger.info("MainNetworkManager stopped")

    # ============ 카메라 → Main (UDP) ============
    def _camera_loop(self, camera_id, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', port))
        sock.settimeout(1.0)

        while self.is_running:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as e:
                self.logger.error(f"[{camera_id}] Receive error: {e}")
                continue

            if self.on_frame is None:
                continue
            try:
                self.on_frame(camera_id, data)
            except Exception as e:
                self.logger.error(f"[{camera_id}] on_frame error: {e}")

        sock.close()

    # ============ Main → AI (UDP) ============
    def send_frame(self, camera_id, mode, frame_idx, jpg_bytes):
        """JPEG 바이트를 AINetworkManager 형식(JSON + base64)으로 전송"""
        jpg_bytes = self._fit_jpeg(jpg_bytes)
        if jpg_bytes is None:
            self.logger.warning(f"[{camera_id}] Frame too large or invalid, dropped")
            return False

        payload = {
            'camera_id': camera_id,
            'mode': mode,
            'frame_idx': frame_idx,
            'frame': base64.b64encode(jpg_bytes).decode('utf-8'),
        }
        data = json.dumps(payload).encode('utf-8')
        if len(data) > UDP_MAX_BYTES:
            self.logger.warning(f"[{camera_id}] Payload {len(data)}B exceeds UDP limit, dropped")
            return False

        try:
            self.ai_sock.sendto(data, (self.ai_host, self.ai_port))
            return True
        except OSError as e:
            self.logger.error(f"[{camera_id}] Send to AI failed: {e}")
            return False

    # ============ Main → GUI 영상 (UDP) ============
    def send_video_to_gui(self, camera_id, frame_idx, jpg_bytes):
        """GUI가 frame_idx로 결과와 매칭할 수 있게 헤더를 붙여서 전송"""
        jpg_bytes = self._fit_jpeg(jpg_bytes)
        if jpg_bytes is None:
            return False

        header = json.dumps({'camera_id': camera_id, 'frame_idx': frame_idx}).encode('utf-8')
        packet = struct.pack('!H', len(header)) + header + jpg_bytes
        try:
            self.gui_video_sock.sendto(packet, (self.gui_host, self.gui_video_port))
            return True
        except OSError as e:
            self.logger.error(f"[{camera_id}] Send video to GUI failed: {e}")
            return False

    def _fit_jpeg(self, jpg_bytes):
        """base64(+33%) 후에도 UDP 한 패킷에 들어가도록 JPEG 크기 조정"""
        max_jpg = UDP_MAX_BYTES * 3 // 4 - 200   # base64 증가분 + JSON 헤더 여유
        if len(jpg_bytes) <= max_jpg:
            return jpg_bytes

        frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return None
        for quality in (70, 50, 30):
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok and len(encoded) <= max_jpg:
                return encoded.tobytes()
        return None

    # ============ AI → Main (HTTP POST /ai_result) ============
    def _make_handler(self):
        manager = self

        class AIResultHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path != '/ai_result':
                    self._reply(404, {'status': 'not found'})
                    return

                length = int(self.headers.get('Content-Length', 0))
                try:
                    data = json.loads(self.rfile.read(length).decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self._reply(400, {'status': 'invalid json'})
                    return

                if manager._store_result(data):
                    self._reply(200, {'status': 'ok'})
                else:
                    self._reply(400, {'status': 'unknown camera_id'})

            def _reply(self, code, body):
                raw = json.dumps(body).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, format, *args):
                pass   # 요청마다 찍히는 기본 접속 로그 끄기

        return AIResultHandler

    def _store_result(self, data):
        camera_id = data.get('camera_id') if isinstance(data, dict) else None
        with self.lock:
            queue = self.result_buffer.get(camera_id)
            if queue is None:
                self.logger.warning(f"Unknown camera_id in AI result: {camera_id}")
                return False
            queue.append(data)
        return True

    def get_latest_results(self):
        """도착한 AI 결과를 전부 꺼내서 반환 (꺼낸 결과는 버퍼에서 제거)

        Returns: {camera_id: [aioutput(dict), ...]}  — 새 결과가 있는 카메라만
        """
        with self.lock:
            results = {cam_id: list(queue) for cam_id, queue in self.result_buffer.items() if queue}
            for queue in self.result_buffer.values():
                queue.clear()
        return results

    # ============ Main → GUI (TCP) ============
    def send_to_gui(self, message):
        """GUI로 JSON 한 줄 전송. GUI가 안 떠 있으면 버리고 주기적으로 재접속"""
        line = (json.dumps(message, ensure_ascii=False, default=_to_json_safe) + '\n').encode('utf-8')

        with self.gui_lock:
            if self.gui_sock is None:
                if time.monotonic() < self.gui_next_retry:
                    return False
                try:
                    self.gui_sock = socket.create_connection((self.gui_host, self.gui_port), timeout=0.5)
                    self.gui_sock.settimeout(1.0)
                    self.gui_warned = False
                    self.logger.info(f"Connected to GUI {self.gui_host}:{self.gui_port}")
                    threading.Thread(target=self._gui_command_loop, args=(self.gui_sock,), daemon=True).start()
                except OSError:
                    self.gui_next_retry = time.monotonic() + GUI_RETRY_SEC
                    if not self.gui_warned:
                        self.logger.warning(f"GUI not reachable ({self.gui_host}:{self.gui_port}), retrying")
                        self.gui_warned = True
                    return False

            try:
                self.gui_sock.sendall(line)
                return True
            except OSError as e:
                self.logger.warning(f"GUI connection lost: {e}")
                self.gui_sock.close()
                self.gui_sock = None
                self.gui_next_retry = time.monotonic() + GUI_RETRY_SEC
                return False

    # ============ GUI → Main (같은 TCP 연결로 명령 수신) ============
    def _gui_command_loop(self, sock):
        """GUI가 보낸 명령(JSON 한 줄)을 읽어서 on_command로 전달. 연결이 바뀌거나 끊기면 종료"""
        buffer = b''
        while self.is_running and self.gui_sock is sock:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break

            buffer += chunk
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                if not line.strip():
                    continue
                try:
                    cmd = json.loads(line.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    self.logger.warning("Invalid command from GUI")
                    continue
                if self.on_command is None:
                    continue
                try:
                    self.on_command(cmd)
                except Exception as e:
                    self.logger.error(f"Command error {cmd.get('cmd')}: {e}")
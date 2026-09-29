# admin_gui/network.py
"""main_server와의 통신: 영상(UDP 9998) 수신, 결과/명령(TCP 9999) 송수신.

기존 admin_gui.py 초안에서 그대로 옮긴 부분 — 탭 구조를 어떻게 바꾸든 이 모듈은
안 건드려도 된다 (FrameStore.pop / render 쪽 매칭 로직은 작업 지시서에서도
"수정하지 않는다, 필요하면 먼저 공유한다"고 되어 있음)."""
import json
import socket
import struct
import threading
from collections import OrderedDict

from config.settings import CAMERA_PORTS, GUI_VIDEO_PORT, GUI_INFO_PORT

FRAME_BUFFER_LEN = 150   # 카메라별 보관 프레임 수 (15FPS 기준 약 10초)

class FrameStore:
    """camera_id별 {frame_idx: jpg_bytes} 버퍼 (스레드 안전)"""

    def __init__(self, maxlen=FRAME_BUFFER_LEN):
        self.maxlen = maxlen
        self.buffers = {cam_id: OrderedDict() for cam_id in CAMERA_PORTS}
        self.latest_idx = {cam_id: None for cam_id in CAMERA_PORTS}
        self.lock = threading.Lock()

    def put(self, camera_id, frame_idx, jpg_bytes):
        with self.lock:
            buf = self.buffers.get(camera_id)
            if buf is None or frame_idx is None:
                return
            buf[frame_idx] = jpg_bytes
            while len(buf) > self.maxlen:
                buf.popitem(last=False)
            self.latest_idx[camera_id] = frame_idx

    def pop(self, camera_id, frame_idx):
        """결과와 같은 frame_idx의 프레임 꺼내기 (그보다 오래된 프레임은 삭제)"""
        with self.lock:
            buf = self.buffers.get(camera_id)
            if buf is None:
                return None
            jpg_bytes = buf.pop(frame_idx, None)
            if jpg_bytes is not None:
                for idx in [i for i in buf if i < frame_idx]:
                    del buf[idx]
            return jpg_bytes

    def latest(self, camera_id):
        with self.lock:
            idx = self.latest_idx.get(camera_id)
            buf = self.buffers.get(camera_id)
            if idx is None or buf is None:
                return None, None
            return idx, buf.get(idx)

    def get_latest_idx(self, camera_id):
        with self.lock:
            return self.latest_idx.get(camera_id)

def video_receiver(store, stop_event, port=GUI_VIDEO_PORT):
    """UDP 영상 수신: [2바이트 헤더 길이][JSON 헤더][JPEG]"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(1.0)
    print(f"[GUI] video listening on UDP {port}")

    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except OSError:
            break

        if len(data) < 2:
            continue
        header_len = struct.unpack('!H', data[:2])[0]
        try:
            header = json.loads(data[2:2 + header_len].decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        store.put(header.get('camera_id'), header.get('frame_idx'), data[2 + header_len:])

    sock.close()

class MainLink:
    """main_server와의 TCP 연결 (결과 수신 + 명령 송신에 같은 연결 사용)"""

    def __init__(self):
        self.conn = None
        self.lock = threading.Lock()

    def set_conn(self, conn):
        with self.lock:
            self.conn = conn

    def send(self, cmd):
        line = (json.dumps(cmd, ensure_ascii=False) + '\n').encode('utf-8')
        with self.lock:
            if self.conn is None:
                return False
            try:
                self.conn.sendall(line)
                return True
            except OSError:
                return False

def result_receiver(result_queue, stop_event, link, port=GUI_INFO_PORT):
    """TCP 결과 수신 서버: main_server가 접속해서 한 줄에 JSON 하나씩 보냄"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', port))
    server.listen(1)
    server.settimeout(1.0)
    print(f"[GUI] result listening on TCP {port}")

    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
        except socket.timeout:
            continue
        except OSError:
            break

        print(f"[GUI] main_server connected: {addr}")
        conn.settimeout(1.0)
        link.set_conn(conn)
        buffer = b''
        while not stop_event.is_set():
            try:
                chunk = conn.recv(65536)
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
                    result_queue.put(json.loads(line.decode('utf-8')))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
        link.set_conn(None)
        conn.close()
        print("[GUI] main_server disconnected")

    server.close()

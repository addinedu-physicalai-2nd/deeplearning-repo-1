# hardware/image_sender.py
"""테스트용 카메라 송신기: 웹캠/영상 파일 → JPEG → UDP → Main Service

카메라 여러 대를 한 번에 보낼 수 있다. --camera와 --source를 같은 순서로 1:1 대응시킨다.
    python -m hardware.image_sender                                   # 기본: CAM-01~04 ← 웹캠 0 2 4 6
    python -m hardware.image_sender --camera CAM-01 --source 0        # 1대만
    python -m hardware.image_sender --camera CAM-01 CAM-02 --source 0 video.mp4
카메라마다 스레드 1개가 읽기 → 인코딩 → 전송을 따로 돌린다 (한 대가 느려도 다른 카메라는 영향 없음).
"""
import argparse
import socket
import threading
import time

import cv2

from config.settings import CAMERA_PORTS

MAX_JPEG_BYTES = 48000    # Main이 base64(+33%)로 AI에 넘겨도 UDP 한 패킷(65507)에 들어가는 크기
MAX_READ_FAILS = 30       # 웹캠 연속 읽기 실패 허용 횟수
LOG_INTERVAL_SEC = 5.0

# 기본 송신 대상: CAM-01~04 ← 웹캠 0, 2, 4, 6
# (리눅스는 웹캠 1대당 /dev/video 번호가 2개씩 생겨서 보통 짝수 번호가 실제 영상 장치)
DEFAULT_CAMERAS = ['CAM-01', 'CAM-02', 'CAM-03', 'CAM-04']
DEFAULT_SOURCES = ['0', '2', '4', '6']


def encode_jpeg(frame, quality):
    """UDP 한 패킷에 들어갈 때까지 품질을 낮춰가며 인코딩"""
    qualities = [quality] + [q for q in (60, 40, 25) if q < quality]
    for q in qualities:
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok and len(encoded) <= MAX_JPEG_BYTES:
            return encoded.tobytes()
    return None


def open_capture(camera_id, source, args):
    """웹캠/영상 열기. 실패하면 None"""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"[{camera_id}] 카메라/영상을 열 수 없음: {source}")
        return None

    if not isinstance(source, str):
        # 웹캠은 MJPG로 받기 (YUYV 무압축은 USB 대역폭 한계로 프레임이 섞여서 깨짐)
        # FOURCC를 먼저 설정해야 해상도/FPS가 MJPG 기준으로 잡힘
        # 여러 대를 동시에 쓸 때는 MJPG가 아니면 USB 대역폭이 모자라서 아예 안 열리기 쉽다
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)

        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace')
        print(f"[{camera_id}] camera format: {fourcc} {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {capture.get(cv2.CAP_PROP_FPS):.0f}fps")
    return capture


def send_loop(camera_id, source, capture, args, stop_event):
    """카메라 1대 송신 루프 (카메라마다 스레드 1개)"""
    is_file = isinstance(source, str)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, CAMERA_PORTS[camera_id])
    interval = 1.0 / args.fps

    sent = 0
    fails = 0
    last_log = time.monotonic()
    print(f"[{camera_id}] source={source} -> udp://{target[0]}:{target[1]} @ {args.fps} FPS")

    try:
        while not stop_event.is_set():
            started = time.monotonic()

            ok, frame = capture.read()
            if not ok:
                if is_file:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 영상 끝나면 처음부터 반복
                    continue
                fails += 1
                if fails >= MAX_READ_FAILS:
                    # 이 카메라만 멈춘다 (다른 카메라 스레드는 계속 송신)
                    print(f"[{camera_id}] 카메라 읽기 {MAX_READ_FAILS}회 연속 실패 → 이 카메라 송신 중단")
                    return
                time.sleep(0.05)
                continue
            fails = 0

            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            jpg = encode_jpeg(frame, args.quality)
            if jpg is None:
                print(f"[{camera_id}] JPEG too large for UDP, frame dropped")
            else:
                sock.sendto(jpg, target)
                sent += 1

            now = time.monotonic()
            if now - last_log >= LOG_INTERVAL_SEC:
                print(f"[{camera_id}] sent {sent} frames in last {LOG_INTERVAL_SEC:.0f}s")
                sent = 0
                last_log = now

            time.sleep(max(0.0, interval - (time.monotonic() - started)))
    finally:
        capture.release()
        sock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', nargs='+', default=DEFAULT_CAMERAS, choices=list(CAMERA_PORTS),
                        help='보낼 카메라 ID들 (--source와 같은 순서로 1:1 대응)')
    parser.add_argument('--source', nargs='+', default=DEFAULT_SOURCES,
                        help='웹캠 번호 또는 영상 파일 경로들 (--camera와 같은 개수)')
    parser.add_argument('--host', default='127.0.0.1', help='Main Service IP')
    parser.add_argument('--fps', type=float, default=15.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--quality', type=int, default=80)
    args = parser.parse_args()

    if len(args.camera) != len(args.source):
        parser.error(f"--camera {len(args.camera)}개와 --source {len(args.source)}개의 개수가 달라요")
    if len(set(args.camera)) != len(args.camera):
        parser.error("--camera에 같은 카메라 ID가 중복됨")

    # 카메라는 메인 스레드에서 차례대로 연다 (여러 대를 동시에 열면 장치 초기화가 꼬이기 쉬움)
    targets = []
    for camera_id, source_arg in zip(args.camera, args.source):
        source = int(source_arg) if source_arg.isdigit() else source_arg
        capture = open_capture(camera_id, source, args)
        if capture is not None:
            targets.append((camera_id, source, capture))
    if not targets:
        raise SystemExit("열린 카메라가 없음")

    stop_event = threading.Event()
    threads = []
    for camera_id, source, capture in targets:
        thread = threading.Thread(target=send_loop, args=(camera_id, source, capture, args, stop_event),
                                  name=camera_id, daemon=True)
        thread.start()
        threads.append(thread)
    print(f"송신 시작: {', '.join(t[0] for t in targets)} ({len(targets)}/{len(args.camera)}대)")

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)


if __name__ == '__main__':
    main()

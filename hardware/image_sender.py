# hardware/image_sender.py
"""테스트용 카메라 송신기: 웹캠/영상 파일 → JPEG → UDP → Main Service"""
import argparse
import socket
import time

import cv2

from config.settings import CAMERA_PORTS

MAX_JPEG_BYTES = 48000    # Main이 base64(+33%)로 AI에 넘겨도 UDP 한 패킷(65507)에 들어가는 크기
MAX_READ_FAILS = 30       # 웹캠 연속 읽기 실패 허용 횟수
LOG_INTERVAL_SEC = 5.0


def encode_jpeg(frame, quality):
    """UDP 한 패킷에 들어갈 때까지 품질을 낮춰가며 인코딩"""
    qualities = [quality] + [q for q in (60, 40, 25) if q < quality]
    for q in qualities:
        ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok and len(encoded) <= MAX_JPEG_BYTES:
            return encoded.tobytes()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default='CAM-01', choices=list(CAMERA_PORTS))
    parser.add_argument('--source', default='0', help='웹캠 번호 또는 영상 파일 경로')
    parser.add_argument('--host', default='127.0.0.1', help='Main Service IP')
    parser.add_argument('--fps', type=float, default=15.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--quality', type=int, default=80)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    is_file = not isinstance(source, int)

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"카메라/영상을 열 수 없음: {source}")

    if not is_file:
        # 웹캠은 MJPG로 받기 (YUYV 무압축은 USB 대역폭 한계로 프레임이 섞여서 깨짐)
        # FOURCC를 먼저 설정해야 해상도/FPS가 MJPG 기준으로 잡힘
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)

        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, 'little').decode(errors='replace')
        print(f"camera format: {fourcc} {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {capture.get(cv2.CAP_PROP_FPS):.0f}fps")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, CAMERA_PORTS[args.camera])
    interval = 1.0 / args.fps

    sent = 0
    fails = 0
    last_log = time.monotonic()
    print(f"[{args.camera}] source={source} -> udp://{target[0]}:{target[1]} @ {args.fps} FPS")

    try:
        while True:
            started = time.monotonic()

            ok, frame = capture.read()
            if not ok:
                if is_file:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)   # 영상 끝나면 처음부터 반복
                    continue
                fails += 1
                if fails >= MAX_READ_FAILS:
                    raise SystemExit(f"카메라 읽기 {MAX_READ_FAILS}회 연속 실패")
                time.sleep(0.05)
                continue
            fails = 0

            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            jpg = encode_jpeg(frame, args.quality)
            if jpg is None:
                print("JPEG too large for UDP, frame dropped")
            else:
                sock.sendto(jpg, target)
                sent += 1

            now = time.monotonic()
            if now - last_log >= LOG_INTERVAL_SEC:
                print(f"[{args.camera}] sent {sent} frames in last {LOG_INTERVAL_SEC:.0f}s")
                sent = 0
                last_log = now

            time.sleep(max(0.0, interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        sock.close()


if __name__ == '__main__':
    main()
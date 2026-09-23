# config/settings.py
from pathlib import Path

# ============ 기준 경로 ============
# settings.py 위치(config/) 기준으로 한 단계 위 = 프로젝트 루트(SilverCare/)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'ai_server' / 'models'

# UDP 포트
CAMERA_PORTS = {
    'CAM-01': 9000,
    'CAM-02': 9001,
    'CAM-03': 9002,
    'CAM-04': 9003,
    'CAM-05': 9004,
}
AI_SERVER_PORT = 9100
GUI_VIDEO_PORT = 9998

# TCP 포트
MAIN_SERVICE_HOST = 'localhost'
MAIN_SERVICE_PORT = 5000
GUI_INFO_PORT = 9999

# ============ 모델 경로 ============
YOLO_POSE_MODEL = str(MODEL_DIR / 'yolov8n-pose.pt')

# Fall Detection
FALL_DETECTOR_MODEL = str(MODEL_DIR / 'fall_lstm.pt')

# Gait Analysis (6개 모델) 자체로딩

# Stretching (모델 불필요)
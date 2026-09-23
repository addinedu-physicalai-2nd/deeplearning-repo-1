# config/settings.py
"""
포트 설정 (통신 포트만 관리)
"""

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

# 모델 경로
YOLO_POSE_MODEL = 'ai_server/models/yolov8n-pose.pt'
FALL_DETECTOR_MODEL = 'ai_server/models/fall_lstm.pt'
GAIT_ANALYZER_MODEL = 'ai_server/models/gait_lstm.pt'
MODEL_DIR = 'ai_server/models'



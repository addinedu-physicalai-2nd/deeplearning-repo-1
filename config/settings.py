# config/settings.py
import os
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

# ============ 낙상(병실) 카메라 ↔ 환자 배정 ============
# DB patients 테이블에 카메라 컬럼이 없어서 여기서 1:1로 배정 (카메라 ID → patients.id)
# 낙상 기록에만 사용. 보행/스트레칭은 GUI에서 환자와 측정 카메라를 따로 고름
CAMERA_PATIENTS = {
    'CAM-01': 1,   # 김철수
    'CAM-02': 2,   # 이영희
    'CAM-03': 3,   # 박민수
    'CAM-04': 4,   # 정수진
    'CAM-05': 5,   # 최동욱
}

# ============ DB (MySQL) ============
# 비밀번호를 git에 올리지 않도록 환경변수 우선, 없으면 기본값 사용
#   export SILVERCARE_DB_PASSWORD=실제비번
DB_HOST = os.environ.get('SILVERCARE_DB_HOST', 'localhost')
DB_PORT = int(os.environ.get('SILVERCARE_DB_PORT', '3306'))
DB_USER = os.environ.get('SILVERCARE_DB_USER', 'root')
DB_PASSWORD = os.environ.get('SILVERCARE_DB_PASSWORD', '')
DB_NAME = os.environ.get('SILVERCARE_DB_NAME', 'hospital_monitoring')
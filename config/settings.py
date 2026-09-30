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
}

# ============ 카메라 역할 (설치 위치 기준 고정) ============
# 'fall'    : 침대(병실) 카메라 — 항상 낙상 감지. 낙상 탭에 표시
# 'gait'    : 보행 측정 카메라 — 보행 탭 "측정 카메라" 목록에만 표시
# 'stretch' : 스트레칭 측정 카메라 — 스트레칭 탭 "측정 카메라" 목록에만 표시
# 여기 없는 카메라(CAM-05)는 역할 미배정 → main은 낙상 모드로 두고, GUI 어느 탭에도 표시 안 함
CAMERA_ROLES = {
    'CAM-01': 'fall',
    'CAM-02': 'fall',
    'CAM-03': 'gait',
    'CAM-04': 'stretch',
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
# CAMERA_ROLES가 'fall'인 카메라만 배정 (CAM-03/04는 측정용이라 환자 고정 배정 없음)
CAMERA_PATIENTS = {
    'CAM-01': 1,   # 김철수
    'CAM-02': 2,   # 이영희
}

# ============ DB (MySQL) ============
# 비밀번호를 git에 올리지 않도록 환경변수 우선, 없으면 기본값 사용
#   export SILVERCARE_DB_PASSWORD=실제비번
DB_HOST = os.environ.get('SILVERCARE_DB_HOST', 'localhost')
DB_PORT = int(os.environ.get('SILVERCARE_DB_PORT', '3306'))
DB_USER = os.environ.get('SILVERCARE_DB_USER', 'root')
DB_PASSWORD = os.environ.get('SILVERCARE_DB_PASSWORD', '')
DB_NAME = os.environ.get('SILVERCARE_DB_NAME', 'hospital_monitoring')
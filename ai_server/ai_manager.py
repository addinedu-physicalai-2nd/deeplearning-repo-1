import threading

from .data_models import AIOutput
from .yolo_pose import YOLOPoseWrapper
from .fall_detector import FallDetectorSession, load_models as load_fall_model
from .gait_analyzer import GaitAnalyzerSession, load_models as load_gait_models  
from .stretching_manager import StretchingManager

from datetime import datetime, timezone
import numpy as np


# ============ 공유 모델 (프로세스 전체에서 1번만 로드) ============
_shared_models = None
_shared_lock = threading.Lock()


def get_shared_models(device: str):
    """Fall/Gait 모델을 처음 호출될 때 한 번만 로드하고, 이후엔 같은 걸 반환"""
    global _shared_models
    with _shared_lock:
        if _shared_models is None:
            fall_model, _ = load_fall_model(device=device)
            gait_models, _ = load_gait_models(device=device)   # ← 튜플이라 언패킹
            _shared_models = {
                'fall': fall_model,
                'gait': gait_models,   # {'normal': model, 'parkinsons': model, ...} 6개
            }
    return _shared_models

class AIManager:
    """camera_id별 AI 인스턴스"""
    
    MODE_FALL_ONLY = 0
    MODE_FALL_GAIT = 1
    MODE_FALL_STRETCH = 2
    
    def __init__(self, camera_id: str, device: str = 'cuda'):
        self.camera_id = camera_id
        self.device = device
        self.current_mode = self.MODE_FALL_ONLY

        # 공유 모델 (전 카메라 공통, 1번만 로드됨)
        models = get_shared_models(self.device)

        # YOLO는 트래커 상태를 가지므로 카메라별로 따로 생성
        self.yolo_pose = YOLOPoseWrapper(device=self.device)

        self.fall_detector = FallDetectorSession(
            camera_id=camera_id,
            lstm_model=models['fall'],
            device=self.device
        )

        self.gait_analyzer = GaitAnalyzerSession(
            camera_id=camera_id,
            models=models['gait'],
            device=self.device
        )

        self.stretching_mgr = StretchingManager(camera_id=camera_id)  # ← 기존 생성 코드 그대로 둬

        self.frame_idx = 0
    
    def set_mode(self, mode: int):
        """모드 변경"""
        try:
            mode = int(mode)
        except (TypeError, ValueError):
            return

        if mode in [self.MODE_FALL_ONLY, self.MODE_FALL_GAIT, self.MODE_FALL_STRETCH]:
            self.current_mode = mode
    
    def process_frame(self, frame: np.ndarray) -> str:
        """frame 처리 → JSON 반환"""
        keypoints_dict = self.yolo_pose.detect(frame)

        # 사람이 없는 프레임도 번호는 증가
        frame_idx = self.frame_idx
        self.frame_idx += 1

        detections = []

        if keypoints_dict:
            # 모드별로 하나의 모듈만 실행
            if self.current_mode == self.MODE_FALL_ONLY:
                detections = self.fall_detector.process_keypoints(frame_idx, keypoints_dict)

            elif self.current_mode == self.MODE_FALL_GAIT:
                detections = self.gait_analyzer.process_keypoints(frame_idx, keypoints_dict)

            elif self.current_mode == self.MODE_FALL_STRETCH:
                detections = self.stretching_mgr.process_keypoints(frame_idx, keypoints_dict)

        return self._make_output(detections)
    
    def _make_output(self, detections: list) -> str:
        output = AIOutput(
            camera_id=self.camera_id,
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            mode=self.current_mode,
            detections=detections
        )
        return output.to_json()
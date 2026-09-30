import threading

from .data_models import AIOutput
from .yolo_pose import YOLOPoseWrapper
from .keypoint_holder import KeypointHolder
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
        # 사라진 관절((0, 0))을 직전 좌표로 채움 — 보행/스트레칭 모드에서만 사용
        self.kpt_holder = KeypointHolder()

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
            # 보행 모드로 들어오거나 나갈 때 보행 세션 초기화 — 측정마다 버퍼 30프레임을
            # 새로 채운 뒤에 판단하고, 누적 점수도 이번 측정분만 쌓이게 한다.
            # (set_mode는 프레임마다 호출되므로 모드가 실제로 바뀔 때만)
            if mode != self.current_mode and self.MODE_FALL_GAIT in (mode, self.current_mode):
                self.gait_analyzer.reset()
            self.current_mode = mode
    
    def process_frame(self, frame: np.ndarray, frame_idx=None) -> str:
        """frame 처리 → JSON 반환"""
        yolo_result = self.yolo_pose.detect(frame)  
        keypoints_dict = yolo_result['keypoints']
        bbox_dict = yolo_result['bbox']

        # Main이 매긴 frame_idx가 있으면 그대로 사용, 없으면 자체 카운터 (사람 없는 프레임도 증가)
        if frame_idx is None:
            frame_idx = self.frame_idx
            self.frame_idx += 1

        detections = []

        # 보행/스트레칭: 신뢰도가 낮아 (0, 0)으로 온 관절을 직전 좌표로 채워서 스켈레톤이
        # 튀지 않게 한다 (keypoint_holder.py 참고). 낙상은 학습 때 입력 조건을 바꾸지 않으려고
        # YOLO 출력을 그대로 쓴다.
        if keypoints_dict and self.current_mode in (self.MODE_FALL_GAIT, self.MODE_FALL_STRETCH):
            keypoints_dict = self.kpt_holder.apply(keypoints_dict)

        if keypoints_dict:
            # 모드별로 하나의 모듈만 실행
            if self.current_mode == self.MODE_FALL_ONLY:
                detections = self.fall_detector.process_keypoints(keypoints_dict, bbox_dict)

            elif self.current_mode == self.MODE_FALL_GAIT:
                detections = self.gait_analyzer.process_keypoints(keypoints_dict, bbox_dict)

            elif self.current_mode == self.MODE_FALL_STRETCH:
                detections = self.stretching_mgr.process_keypoints(keypoints_dict, bbox_dict)

        return self._make_output(detections, frame_idx) 
    
    def _make_output(self, detections: list, frame_idx) -> str:
        output = AIOutput(
            camera_id=self.camera_id,
            timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            frame_idx = frame_idx,
            mode=self.current_mode,
            detections=detections
        )
        return output.to_json()
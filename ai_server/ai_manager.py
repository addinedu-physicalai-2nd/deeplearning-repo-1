# ai_server/ai_manager.py
from .data_models import AIOutput
from .fall_detector import FallDetectorSession
from .gait_analyzer import GaitAnalyzerSession
from .stretching_manager import StretchingManager

from datetime import datetime
from dataclasses import asdict
import numpy as np

class AIManager:
    """camera_id별 AI 인스턴스"""
    
    MODE_FALL_ONLY = 0
    MODE_FALL_GAIT = 1
    MODE_FALL_STRETCH = 2
    
    def __init__(self, camera_id: str, yolo_pose, device: str = 'cuda'):
        self.camera_id = camera_id
        self.device = device
        self.current_mode = self.MODE_FALL_ONLY
        
        # YOLOPose는 AIService에서 전달받음 (공유)
        self.yolo_pose = yolo_pose
        
        # 각 모듈은 settings에서 직접 로드
        self.fall_detector = FallDetectorSession(
            camera_id=camera_id,
            device=self.device
        )
        
        self.gait_analyzer = GaitAnalyzerSession(
            camera_id=camera_id,
            device=self.device
        )
        
        self.stretching_mgr = StretchingManager(
            camera_id=camera_id,
            device=self.device
        )
        
        self.frame_idx = 0
    
    def set_mode(self, mode: int):
        """모드 변경"""
        if mode in [self.MODE_FALL_ONLY, self.MODE_FALL_GAIT, self.MODE_FALL_STRETCH]:
            self.current_mode = mode
    
    def process_frame(self, frame: np.ndarray) -> str:
        """frame 처리 → JSON 반환"""
        keypoints_dict = self.yolo_pose.detect(frame)
        
        if not keypoints_dict:
            return self._make_empty_output()
        
        results = []
        
        # 모드별로 하나의 모듈만 실행
        if self.current_mode == self.MODE_FALL_ONLY:
            fall_output = self.fall_detector.process_keypoints(self.frame_idx, keypoints_dict)
            results.append(asdict(fall_output))
        
        elif self.current_mode == self.MODE_FALL_GAIT:
            gait_output = self.gait_analyzer.process_keypoints(self.frame_idx, keypoints_dict)
            results.append(asdict(gait_output))
        
        elif self.current_mode == self.MODE_FALL_STRETCH:
            stretch_output = self.stretching_mgr.process_keypoints(self.frame_idx, keypoints_dict)
            results.append(asdict(stretch_output))
        
        unified_output = AIOutput(
            camera_id=self.camera_id,
            timestamp=datetime.now().isoformat(),
            mode=self.current_mode,
            detections=results
        )
        
        self.frame_idx += 1
        return unified_output.to_json()
    
    def _make_empty_output(self) -> str:
        """keypoints 없을 때"""
        empty_output = AIOutput(
            camera_id=self.camera_id,
            timestamp=datetime.now().isoformat(),
            mode=self.current_mode,
            detections=[]
        )
        return empty_output.to_json()
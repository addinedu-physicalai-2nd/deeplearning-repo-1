# ai_server/yolo_pose.py
from config.settings import YOLO_POSE_MODEL
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple

class YOLOPoseWrapper:
    """YOLO Pose 모델 래퍼"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        # settings에서 모델 경로 가져와서 직접 로드
        self.model = YOLO(YOLO_POSE_MODEL)
        self.model.to(self.device)
        self.conf_threshold = 0.5
    
    def detect(self, frame: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """원본 keypoints만 반환 (정규화 없음)"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        keypoints_dict = []
        
        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue
            
            track_ids = result.boxes.id
            if track_ids is None:
                continue
            
            keypoints_array = result.keypoints.xy.cpu().numpy()
            
            for track_id, keypoints in zip(track_ids, keypoints_array):
                if keypoints.shape == (17, 2):
                    keypoints_dict.append((int(track_id), keypoints.astype(np.float32)))
        
        return keypoints_dict
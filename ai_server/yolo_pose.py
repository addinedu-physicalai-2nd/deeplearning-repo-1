# ai_server/yolo_pose.py
import numpy as np
from ultralytics import YOLO
from typing import Dict

from config.settings import YOLO_POSE_MODEL


class YOLOPoseWrapper:
    """YOLO Pose 모델을 래핑하여 frame → keypoints_dict 변환"""

    def __init__(self, device: str = 'cuda'):
        self.model = YOLO(YOLO_POSE_MODEL)
        self.device = device
        self.conf_threshold = 0.5

    def detect(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """
        영상에서 사람 탐지 + 추적 후 keypoints 추출 (정규화 없음)

        Args:
            frame: BGR 영상 (H, W, 3)

        Returns:
            {track_id: keypoints} 형식
            keypoints shape: (17, 2)
        """
        results = self.model.track(frame, persist=True, conf=self.conf_threshold,device=self.device, verbose=False)
        keypoints_dict = {}

        for result in results:
            if result.boxes is None or result.keypoints is None:
                continue

            track_ids = result.boxes.id
            if track_ids is None:
                continue

            keypoints_array = result.keypoints.xy.cpu().numpy()

            for track_id, keypoints in zip(track_ids, keypoints_array):
                track_id = int(track_id)

                if keypoints.shape == (17, 2):
                    keypoints_dict[track_id] = keypoints.astype(np.float32)

        return keypoints_dict
# ai_server/stretching_manager.py
import numpy as np
from .data_models import  RawDetection

class StretchingManager:
    """
    스트레칭 피드백 - AI Server 담당 부분
    사용자 keypoints 정규화 & feature 추출만 수행
    reference_poses와의 비교는 Main에서 담당
    """
    
    MODE_STRETCHING = 2
    
    def __init__(self, camera_id: str, device: str = 'cuda'):
        self.camera_id = camera_id
        self.device = device
        self.frame_idx = 0
        
        # Limb 정의
        self.limbs = [(5, 9), (6, 10), (11, 15), (12, 16)]  # (p1, p2)
        self.arm_points = (9, 10)  # 손목
        self.leg_points = (15, 16)  # 발목
    
    def process_keypoints(self, frame_idx: int, keypoints_dict) -> list:
        """
        사용자 keypoints 가공 (정규화 + feature 추출)
        
        Args:
            frame_idx: 프레임 번호
            keypoints_dict: {track_id: keypoints_array (17x2), ...}
        
        Returns:
            AIOutput (mode=2, detections=[RawDetection])
        """
        detections = []
        
        for track_id, keypoints in keypoints_dict.items():
            try:
                # Step 1: 신체 중심 기준 정규화 (선택사항)
                normalized_kp = self._normalize_keypoints(keypoints)
                
                # Step 2: feature 추출 (Main에서 사용할 데이터)
                features = self._extract_features(normalized_kp, keypoints)
                
                detection = RawDetection(
                    track_id=track_id,
                    bbox=[0, 0, 0, 0],
                    raw_data=features
                )
                detections.append(detection)
            
            except Exception as e:
                print(f"[StretchingManager] Error processing track {track_id}: {e}")
                continue
        
        self.frame_idx += 1
        
        return detections
    
    def _normalize_keypoints(self, keypoints):
        """신체 중심 기준 정규화 (어깨 중심)"""
        keypoints = np.array(keypoints, dtype=np.float32)
        
        # 어깨 중심 계산
        shoulder_center = (keypoints[5] + keypoints[6]) / 2
        
        # 중심 이동
        normalized = keypoints.copy()
        normalized[:, 0] -= shoulder_center[0]
        normalized[:, 1] -= shoulder_center[1]
        
        return normalized
    
    def _extract_features(self, normalized_kp, raw_kp):
        """
        필요한 feature 추출
        Main에서 기준 자세와 비교하기 위한 데이터
        """
        normalized_kp = np.array(normalized_kp, dtype=np.float32)
        raw_kp = np.array(raw_kp, dtype=np.float32)
        
        # ============ 각도 계산 ============
        angles = []
        for p1, p2 in self.limbs:
            vec = normalized_kp[p2] - normalized_kp[p1]
            angle = np.degrees(np.arctan2(vec[1], vec[0]))
            angles.append(angle)
        
        # ============ 거리 계산 ============
        arm_dist = float(np.linalg.norm(normalized_kp[9] - normalized_kp[10]))
        leg_dist = float(np.linalg.norm(normalized_kp[15] - normalized_kp[16]))
        
        # ============ 신뢰도 (마지막 column이 confidence인 경우) ============
        if raw_kp.shape[1] > 2:
            # keypoint에 confidence 포함
            confidence = float(np.mean([
                raw_kp[i, 2] for i in [5, 6, 9, 10, 11, 12, 15, 16]
            ]))
        else:
            confidence = 1.0
        
        return {
            'keypoints': normalized_kp.tolist(),  # Main에서 호모그래피 적용용
            'limb_angles': [float(a) for a in angles],
            'spread_distances': [arm_dist, leg_dist],
            'confidence': confidence
        }
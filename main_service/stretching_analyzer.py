import numpy as np

# 단계 기준 (GUI는 level만 보고 색을 정함)
GOOD_SCORE = 70
ADJUST_SCORE = 50

# 벌림 거리 비교: 픽셀 거리 대신 "몸통 길이 대비 비율"로 비교 (해상도/카메라 거리와 무관)
SPREAD_TOLERANCE = 0.7   # 비율 차이가 몸통 길이의 0.7배면 0점
MIN_BODY_SCALE = 10.0    # 몸통 길이가 이보다 작으면(px) 비교 불가로 보고 건너뜀

class StretchingAnalyzer:
    """Main - 기준 자세와 비교 후 GUI 데이터 생성"""
    
    def __init__(self, reference_data):
        """
        reference_data: {
            "movement_name": "...",
            "skeletons": [
                {"frame_index": 395, "keypoints": [[x,y], ...], "confidence": [0.99, ...]},
                ...
            ]
        }
        """
        self.skeletons = reference_data['skeletons']
    
    def analyze(self, aioutput: dict):
        """AIOutput (dict 형식) → GUI 데이터"""
        
        frame_idx = aioutput['frame_idx']
        detections = aioutput['detections']
        
        # frame_idx에 해당하는 skeleton 찾기
        ref_skeleton = None
        for skeleton in self.skeletons:
            if skeleton['frame_index'] == frame_idx:
                ref_skeleton = skeleton
                break
        
        if ref_skeleton is None:
            return {'frame_idx': frame_idx, 'tracking_data': {}}
        
        ref_kp = np.array(ref_skeleton['keypoints'], dtype=np.float32)
        ref_scale = self._body_scale(ref_kp)
        if ref_scale is None:
            return {'frame_idx': frame_idx, 'tracking_data': {}}
        
        tracking_data = {}
        
        for detection in detections:
            track_id = detection['track_id']
            bbox = detection['bbox']
            raw_data = detection['raw_data']
            
            user_limb_angles = raw_data['limb_angles']
            user_spread_distances = raw_data['spread_distances']
            keypoints_drawn = raw_data['keypoints']
            confidence = raw_data['confidence']
            
            # 몸통 길이 (원본 픽셀 좌표 우선, 없으면 정규화 좌표 사용)
            user_scale = self._body_scale(raw_data.get('keypoints_px', keypoints_drawn))
            if user_scale is None:
                continue
            
            # ============ 기준 자세의 각도/거리 ============
            ref_limb_angles = {
                'left_arm': self._compute_angle(ref_kp, (5, 9)),
                'right_arm': self._compute_angle(ref_kp, (6, 10)),
                'left_leg': self._compute_angle(ref_kp, (11, 15)),
                'right_leg': self._compute_angle(ref_kp, (12, 16))
            }
            ref_spread_ratios = {
                'arm_spread': float(np.linalg.norm(ref_kp[9] - ref_kp[10])) / ref_scale,
                'leg_spread': float(np.linalg.norm(ref_kp[15] - ref_kp[16])) / ref_scale
            }
            
            # ============ 각도 비교 ============
            limbs_data = {}
            limb_scores = []
            for limb_name in ['left_arm', 'right_arm', 'left_leg', 'right_leg']:
                angle_diff = abs(user_limb_angles[limb_name] - ref_limb_angles[limb_name])
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                score = max(0, 100 - (angle_diff / 90) * 100)
                limbs_data[limb_name] = {
                    'angle': float(user_limb_angles[limb_name]),
                    'score': float(score),
                    'level': self._get_level(score)
                }
                limb_scores.append(score)
            
            # ============ 거리 비교 ============
            spreads_data = {}
            spread_scores = []
            for spread_name in ['arm_spread', 'leg_spread']:
                user_ratio = user_spread_distances[spread_name] / user_scale
                ratio_diff = abs(user_ratio - ref_spread_ratios[spread_name])
                score = max(0, 100 - (ratio_diff / SPREAD_TOLERANCE) * 100)
                
                spreads_data[spread_name] = {
                    'ratio': float(user_ratio),
                    'score': float(score),
                    'level': self._get_level(score)
                }
                spread_scores.append(score)
            
            # ============ 전체 점수 (각도 60% + 거리 40%) ============
            limb_avg = np.mean(limb_scores)
            spread_avg = np.mean(spread_scores)
            overall_score = (limb_avg * 0.6) + (spread_avg * 0.4)
            
            # ============ 관절별 정확도 ============
            joint_accuracy = {
                'shoulder': {'score': float(confidence * 100), 'level': self._get_level(confidence * 100)},
                'elbow': {'score': float((limbs_data['left_arm']['score'] + limbs_data['right_arm']['score']) / 2),
                         'level': self._get_level((limbs_data['left_arm']['score'] + limbs_data['right_arm']['score']) / 2)},
                'wrist': {'score': spreads_data['arm_spread']['score'], 'level': spreads_data['arm_spread']['level']},
                'hip': {'score': float(confidence * 100), 'level': self._get_level(confidence * 100)},
                'knee': {'score': float((limbs_data['left_leg']['score'] + limbs_data['right_leg']['score']) / 2),
                        'level': self._get_level((limbs_data['left_leg']['score'] + limbs_data['right_leg']['score']) / 2)},
                'ankle': {'score': spreads_data['leg_spread']['score'], 'level': spreads_data['leg_spread']['level']}
            }
            
            # ============ 전체 피드백 (점수 + 단계) ============
            overall = {
                'score': float(overall_score),
                'level': self._get_level(overall_score)
            }
            
            tracking_data[track_id] = {
                'overall': overall,
                'limbs': limbs_data,
                'spreads': spreads_data,
                'joint_accuracy': joint_accuracy,
                'bbox': bbox,
                'keypoints_drawn': keypoints_drawn,
                'confidence': float(confidence)
            }
        
        return {
            'frame_idx': frame_idx,
            'tracking_data': tracking_data
        }
    
    def _body_scale(self, keypoints):
        """몸통 길이 (어깨 중심 ↔ 엉덩이 중심). 어깨/엉덩이를 못 찾았거나 너무 작으면 None"""
        kp = np.asarray(keypoints, dtype=np.float32)[:, :2]
        points = kp[[5, 6, 11, 12]]
        if np.any(np.all(points == 0, axis=1)):    # YOLO가 못 찾은 점은 (0, 0)
            return None
        shoulder_center = (kp[5] + kp[6]) / 2
        hip_center = (kp[11] + kp[12]) / 2
        scale = float(np.linalg.norm(shoulder_center - hip_center))
        return scale if scale >= MIN_BODY_SCALE else None
    
    def _compute_angle(self, keypoints, limb):
        vec = keypoints[limb[1]] - keypoints[limb[0]]
        return float(np.degrees(np.arctan2(vec[1], vec[0])))
    
    def _get_level(self, score):
        """점수 → 단계 ('good' / 'adjust' / 'check'). 색은 GUI가 단계를 보고 정함"""
        if score >= GOOD_SCORE:
            return 'good'
        elif score >= ADJUST_SCORE:
            return 'adjust'
        else:
            return 'check'
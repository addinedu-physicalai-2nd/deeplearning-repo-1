import numpy as np

class StretchingAnalyzer:
    """Main - 기준 자세와 비교 후 GUI 데이터 생성"""
    
    def __init__(self, reference_poses):
        """reference_poses: list of {'original': keypoints, 'confidence': [...]}"""
        self.reference_poses = reference_poses
    
    def analyze(self, detections, frame_idx):
        """AIOutput detections + frame_idx → GUI 데이터"""
        
        if frame_idx >= len(self.reference_poses):
            return {'frame_idx': frame_idx, 'tracking_data': {}}
        
        ref_data = self.reference_poses[frame_idx]
        ref_kp = np.array(ref_data['original'], dtype=np.float32)
        
        tracking_data = {}
        
        for detection in detections:
            track_id = detection['track_id']
            bbox = detection['bbox']
            raw_data = detection['raw_data']
            
            user_limb_angles = raw_data['limb_angles']
            user_spread_distances = raw_data['spread_distances']
            confidence = raw_data['confidence']
            
            # ============ 기준 자세의 각도/거리 ============
            ref_limb_angles = {
                'left_arm': self._compute_angle(ref_kp, (5, 9)),
                'right_arm': self._compute_angle(ref_kp, (6, 10)),
                'left_leg': self._compute_angle(ref_kp, (11, 15)),
                'right_leg': self._compute_angle(ref_kp, (12, 16))
            }
            ref_spread_distances = {
                'arm_spread': float(np.linalg.norm(ref_kp[9] - ref_kp[10])),
                'leg_spread': float(np.linalg.norm(ref_kp[15] - ref_kp[16]))
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
                    'color': self._get_color(score)
                }
                limb_scores.append(score)
            
            # ============ 거리 비교 ============
            spreads_data = {}
            spread_scores = []
            for spread_name in ['arm_spread', 'leg_spread']:
                dist_diff = abs(user_spread_distances[spread_name] - ref_spread_distances[spread_name])
                score = max(0, 100 - (dist_diff / 100) * 100)
                
                spreads_data[spread_name] = {
                    'distance': float(user_spread_distances[spread_name]),
                    'score': float(score),
                    'color': self._get_color(score)
                }
                spread_scores.append(score)
            
            # ============ 전체 점수 (각도 60% + 거리 40%) ============
            limb_avg = np.mean(limb_scores)
            spread_avg = np.mean(spread_scores)
            overall_score = (limb_avg * 0.6) + (spread_avg * 0.4)
            
            # ============ 관절별 정확도 ============
            joint_accuracy = {
                'shoulder': {'score': float(confidence * 100), 'color': self._get_color(confidence * 100)},
                'elbow': {'score': float((limbs_data['left_arm']['score'] + limbs_data['right_arm']['score']) / 2),
                         'color': self._get_color((limbs_data['left_arm']['score'] + limbs_data['right_arm']['score']) / 2)},
                'wrist': {'score': spreads_data['arm_spread']['score'], 'color': spreads_data['arm_spread']['color']},
                'hip': {'score': float(confidence * 100), 'color': self._get_color(confidence * 100)},
                'knee': {'score': float((limbs_data['left_leg']['score'] + limbs_data['right_leg']['score']) / 2),
                        'color': self._get_color((limbs_data['left_leg']['score'] + limbs_data['right_leg']['score']) / 2)},
                'ankle': {'score': spreads_data['leg_spread']['score'], 'color': spreads_data['leg_spread']['color']}
            }
            
            # ============ 피드백 ============
            if overall_score >= 70:
                feedback = 'GOOD'
                feedback_color = (0, 255, 0)
            elif overall_score >= 50:
                feedback = 'ADJUST'
                feedback_color = (0, 255, 255)
            else:
                feedback = 'CHECK'
                feedback_color = (0, 0, 255)
            
            tracking_data[track_id] = {
                'overall_score': float(overall_score),
                'feedback': feedback,
                'feedback_color': feedback_color,
                'limbs': limbs_data,
                'spreads': spreads_data,
                'joint_accuracy': joint_accuracy,
                'bbox': bbox,
                'keypoints_drawn': raw_data['keypoints'],
                'confidence': float(confidence)
            }
        
        return {
            'frame_idx': frame_idx,
            'tracking_data': tracking_data
        }
    
    def _compute_angle(self, keypoints, limb):
        vec = keypoints[limb[1]] - keypoints[limb[0]]
        return float(np.degrees(np.arctan2(vec[1], vec[0])))
    
    def _get_color(self, score):
        if score >= 70:
            return (0, 255, 0)
        elif score >= 50:
            return (0, 255, 255)
        else:
            return (0, 0, 255)
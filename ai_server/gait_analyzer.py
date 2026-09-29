import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any
from config.settings import MODEL_DIR

# ==========================================
# 0. 데이터 입출력 구조체 정의
# ==========================================
@dataclass
class RawDetection:
    track_id: int
    bbox: List[float]
    raw_data: Dict[str, Any]

# ==========================================
# 1. LSTM 딥러닝 모델 아키텍처
# ==========================================
class GaitLSTM(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=128, num_layers=2, num_classes=2):
        super(GaitLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ==========================================
# 2. 글로벌 설정 및 유틸리티 함수
# ==========================================
TARGET_DISEASES = ['normal', 'parkinsons', 'stroke', 'antalgic', 'myopathic', 'abnormal']
WINDOW_SIZE = 30

def load_models(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_dict = {}
    for disease in TARGET_DISEASES:
        model_path = os.path.join(MODEL_DIR, f'model_{disease}.pth')
        model = GaitLSTM(num_classes=2).to(device)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"Warning: {model_path} not found. Using randomly initialized weights.")
            
        model.eval()
        models_dict[disease] = model
        
    return models_dict, device

def normalize(kpts: np.ndarray) -> np.ndarray:
    pelvis = (kpts[11] + kpts[12]) / 2.0
    centered_kpts = kpts - pelvis
    
    width = np.ptp(kpts[:, 0]) + 1e-6
    height = np.ptp(kpts[:, 1]) + 1e-6
    
    centered_kpts[:, 0] /= width
    centered_kpts[:, 1] /= height
    
    return centered_kpts.flatten()

# ==========================================
# 3. 메인 분석 세션 클래스
# ==========================================
class GaitAnalyzerSession:
    def __init__(self, camera_id, models=None, device=None, fps=None):
        self.camera_id = camera_id
        self.models = models
        self.device = device if device else torch.device('cpu')
        self.fps = fps
        self.sequence_buffers = {}
        # 🔥 추가: 개별 인원(track_id)의 누적 확률을 기록할 딕셔너리 추가
        self.history_probs = {}

    def process_keypoints(self, keypoints_dict: dict, bbox_dict: dict) -> List[RawDetection]:
        detections = []
        
        # 1. 화면에서 이탈한 객체의 메모리 및 누적 데이터 정리
        active_tracks = set(keypoints_dict.keys())
        for track_id in list(self.sequence_buffers.keys()):
            if track_id not in active_tracks:
                del self.sequence_buffers[track_id]
                if track_id in self.history_probs:
                    del self.history_probs[track_id]
        
        # 2. 현재 활성화된 객체 처리
        for track_id, kpts in keypoints_dict.items():
            if track_id not in self.sequence_buffers:
                self.sequence_buffers[track_id] = deque(maxlen=WINDOW_SIZE)
                # 새로운 인원이 등장하면 질환별 누적 리스트 초기화
                self.history_probs[track_id] = {disease: [] for disease in TARGET_DISEASES}
                
            norm_kpts = normalize(kpts)
            self.sequence_buffers[track_id].append(norm_kpts)
            
            bbox = bbox_dict[track_id] 
            
            # 3. 30프레임 도달 시 추론 및 실시간/누적 확률 계산
            if len(self.sequence_buffers[track_id]) == WINDOW_SIZE:
                input_tensor = torch.tensor([list(self.sequence_buffers[track_id])], dtype=torch.float32).to(self.device)
                
                raw_probs = {}
                with torch.no_grad():
                    for disease_name, model in self.models.items():
                        outputs = model(input_tensor)
                        prob = F.softmax(outputs, dim=1)[0][1].item()
                        raw_probs[disease_name] = prob
                
                total_prob = sum(raw_probs.values())
                realtime_scores = {}
                cumulative_scores = {}
                
                # 실시간 확률 계산 및 누적 데이터 갱신
                for d_name, prob in raw_probs.items():
                    score = round(prob / total_prob, 3) if total_prob > 0 else 0.0
                    realtime_scores[d_name] = score
                    self.history_probs[track_id][d_name].append(score)
                
                # 누적 평균 확률 산출
                for d_name in TARGET_DISEASES:
                    history = self.history_probs[track_id][d_name]
                    avg_score = round(sum(history) / len(history), 3) if history else 0.0
                    cumulative_scores[d_name] = avg_score
                
                gait_score = round(realtime_scores.get('normal', 0.0) * 100, 1)
                
                detection = RawDetection(
                    track_id=track_id,
                    bbox=bbox,
                    raw_data={
                        'gait_score': gait_score,
                        'disease_scores': realtime_scores,         # 🔥 실시간 확률
                        'cumulative_scores': cumulative_scores,    # 🔥 누적 평균 확률
                        'confidence': 1.0,
                        'keypoints_px': np.asarray(kpts)[:, :2].tolist()   # GUI에서 영상 위에 그릴 원본 픽셀 좌표
                    }
                )
                detections.append(detection)
        
        # 🔥 AIOutput 객체 패키징을 제거하고 리스트(detections) 자체를 직접 반환
        return detections
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from config.settings import MODEL_DIR
from .data_models import RawDetection

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
# 🔥 직접 학습하신 6개 질환 클래스로 원복
TARGET_DISEASES = ['normal', 'parkinsons', 'stroke', 'antalgic', 'myopathic', 'abnormal']
WINDOW_SIZE = 30

def load_models(device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_dict = {}
    for disease in TARGET_DISEASES:
        # 모델 파일명 규칙을 이전 학습 코드와 동일하게 맞춤 (예: model_normal.pth)
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
    """
    17x2 형태의 픽셀 좌표를 골반 중심으로 0~1 스케일로 정규화
    """
    kpts = np.asarray(kpts, dtype=np.float32)
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

    def process_keypoints(self,frame_idx:int, keypoints_dict: dict) -> list:
        detections = []
        
        active_tracks = set(keypoints_dict.keys())
        for track_id in list(self.sequence_buffers.keys()):
            if track_id not in active_tracks:
                del self.sequence_buffers[track_id]
        
        for track_id, kpts in keypoints_dict.items():
            kpts = np.asarray(kpts, dtype=np.float32)
            if track_id not in self.sequence_buffers:
                self.sequence_buffers[track_id] = deque(maxlen=WINDOW_SIZE)
                
            norm_kpts = normalize(kpts)
            self.sequence_buffers[track_id].append(norm_kpts)
            
            bbox = [
                float(np.min(kpts[:, 0])), float(np.min(kpts[:, 1])),
                float(np.max(kpts[:, 0])), float(np.max(kpts[:, 1]))
            ]
            
            if len(self.sequence_buffers[track_id]) == WINDOW_SIZE:
                input_tensor = torch.from_numpy(
                    np.stack(self.sequence_buffers[track_id])
                ).unsqueeze(0).to(self.device)
                
                raw_probs = {}
                with torch.no_grad():
                    for disease_name, model in self.models.items():
                        outputs = model(input_tensor)
                        prob = F.softmax(outputs, dim=1)[0][1].item()
                        raw_probs[disease_name] = prob
                
                total_prob = sum(raw_probs.values())
                disease_scores = {}
                for d_name, prob in raw_probs.items():
                    disease_scores[d_name] = round(prob / total_prob, 3) if total_prob > 0 else 0.0
                
                # Gait Score: 정상(normal) 확률을 기반으로 산출하도록 수정
                gait_score = round(disease_scores.get('normal', 0.0) * 100, 1)
                
                detection = RawDetection(
                    track_id=track_id,
                    bbox=bbox,
                    raw_data={
                        'gait_score': gait_score,
                        'disease_scores': disease_scores,
                        'confidence': 1.0
                    }
                )
                detections.append(detection)
        
        return detections
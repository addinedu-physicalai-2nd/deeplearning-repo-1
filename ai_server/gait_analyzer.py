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

def combine_scores(raw_probs: dict) -> dict:
    """모델 6개의 확률 → 합이 1인 질환별 비율 (2단계 방식)

    모델들은 "이 질환이냐 아니냐"를 각자 따로 학습한 이진 분류기라, 확률을 그냥 더해서
    나누면 질환 모델들이 애매하게(0.3~0.5) 답할 때 정상 몫이 희석된다
    (예: 정상 0.9 / 나머지 각 0.4 → 정상 31%).
    그래서 1단계로 정상 모델의 확률을 정상 비율로 그대로 쓰고,
    2단계로 나머지(1 - 정상)를 질환 모델 5개의 확률 비율대로 나눈다.
      예: 정상 0.9 / 나머지 각 0.4 → 정상 90%, 질환 각 2%
    질환 모델 확률이 모두 0이면 나머지를 똑같이 나눈다.
    """
    normal = min(max(raw_probs.get('normal', 0.0), 0.0), 1.0)
    diseases = [d for d in TARGET_DISEASES if d != 'normal']
    disease_total = sum(raw_probs.get(d, 0.0) for d in diseases)
    rest = 1.0 - normal

    scores = {'normal': round(normal, 3)}
    for d in diseases:
        share = raw_probs.get(d, 0.0) / disease_total if disease_total > 0 else 1.0 / len(diseases)
        scores[d] = round(rest * share, 3)
    return scores

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

    def reset(self):
        """측정 세션 초기화 — 30프레임 버퍼와 누적 확률을 전부 비운다.
        보행 모드로 들어오거나 나갈 때 AIManager.set_mode()가 호출한다.
        (안 비우면 이전 측정의 프레임이 버퍼에 남아서, 다시 시작하자마자
        새 프레임 1장만으로 판단이 나오고 누적 점수도 이전 측정에 이어서 쌓인다)"""
        self.sequence_buffers.clear()
        self.history_probs.clear()

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
                
                # 실시간 비율 계산 (정상 모델 우선 2단계 방식 — combine_scores 참고) 및 누적 데이터 갱신
                realtime_scores = combine_scores(raw_probs)
                cumulative_scores = {}
                for d_name, score in realtime_scores.items():
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
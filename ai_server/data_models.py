# ai_server/data_models.py
from dataclasses import dataclass, field, asdict
import json

@dataclass
class RawDetection:
    """AI가 반환하는 원본 탐지 데이터"""
    track_id: int
    bbox: list
    raw_data: dict

@dataclass
class AIOutput:
    """AI 모듈의 최종 출력"""
    camera_id: str
    timestamp: str
    mode: int = 0
    detections: list = field(default_factory=list)
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
# admin_gui/patients.py
"""보행 탭 환자 목록 — DB 스키마가 나오기 전까지 쓰는 더미 데이터.

TODO(DB 연동 담당자): 여기에 작업하시면 됩니다.
  get_patients()의 내부만 실제 DB 조회로 교체하면 됩니다. 함수 시그니처(인자 없음,
  Patient 리스트 반환)와 Patient 필드는 GUI 쪽 코드(gait_tab.py)가 그대로 쓰고
  있으니 유지해 주세요. camera_id는 그 환자가 있는 병실을 찍는 카메라
  (config.settings.CAMERA_PORTS의 키, 예: 'CAM-01')와 1:1로 맞아야 합니다.
"""
from dataclasses import dataclass

@dataclass
class Patient:
    patient_id: str
    name: str
    age: int
    room: str
    camera_id: str

DUMMY_PATIENTS = [
    Patient('101018', '송길호', 76, '101호', 'CAM-01'),
    Patient('101021', '김영희', 78, '101호', 'CAM-01'),
    Patient('102003', '박철수', 87, '102호', 'CAM-02'),
    Patient('103010', '이순자', 81, '103호', 'CAM-03'),
    Patient('104007', '정미영', 70, '104호', 'CAM-04'),
]

def get_patients():
    """환자 목록 반환. TODO(DB 연동 담당자): DB 스키마 확정되면 이 함수 내부만 교체."""
    return list(DUMMY_PATIENTS)

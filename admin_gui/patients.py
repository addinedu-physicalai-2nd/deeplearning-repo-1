# admin_gui/patients.py
"""환자 목록 — main_server가 DB에서 조회해서 보내준 값을 들고 있는 캐시.

GUI가 main_server와 연결되면 app.py가 db_stub.request_patients()로 요청하고,
응답이 오면 set_patients()로 채운 뒤 각 탭의 set_patients()를 호출해 목록을 갱신한다.
(연결 전에는 빈 목록)
"""
from dataclasses import dataclass


@dataclass
class Patient:
    patient_id: str
    name: str
    age: int
    room: str
    camera_id: str
    caregiver_name: str = ''


_patients = []


def get_patients():
    """현재 캐시된 환자 목록 (시그니처 유지: 인자 없음, Patient 리스트 반환)"""
    return list(_patients)


def set_patients(rows):
    """main_server 응답(dict 리스트)으로 캐시 교체"""
    _patients[:] = [
        Patient(
            patient_id=str(r['patient_id']),
            name=r['name'],
            age=r.get('age') or 0,
            room=r.get('room') or '',
            camera_id=r['camera_id'],
            caregiver_name=r.get('caregiver_name') or '',
        )
        for r in rows
    ]
    return get_patients()
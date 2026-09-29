# admin_gui/db_client.py
"""main_server 요청 창구 — GUI는 DB에 직접 접속하지 않고, main_server에 요청을 보낸다.
main_server가 DBManager(main_service/db_manager.py)로 조회/저장한 뒤 결과를 TCP 9999로 돌려준다.

요청:  {"cmd": "<이름>", "req_id": n, ...인자}
응답:  {"type": "response", "cmd": "<이름>", "req_id": n, "ok": true/false, "data": ...}
       → app.py의 handle_server_message()가 받아서 처리

낙상 이벤트(주의/낙상 확정)는 main_server가 FallAnalyzer 결과를 받는 즉시 직접 기록하므로
GUI에서 따로 보낼 요청이 없다.
"""
import itertools
from dataclasses import dataclass


@dataclass
class Patient:
    """보행/스트레칭 탭 환자 목록에 쓰는 환자 정보 (main_server get_patients 응답 1건)"""
    patient_id: str
    name: str
    age: int
    room: str
    camera_id: str          # 낙상(병실) 카메라 배정. 없으면 None — 보행/스트레칭은 이 값을 안 씀
    caregiver_name: str = ''


_link = None
_req_ids = itertools.count(1)


def set_link(link):
    """app.py에서 MainLink를 한 번 연결해준다"""
    global _link
    _link = link


def request(cmd, **params):
    """main_server에 요청 전송. 보냈으면 req_id, 연결이 없으면 None"""
    if _link is None:
        return None
    req_id = next(_req_ids)
    if not _link.send({'cmd': cmd, 'req_id': req_id, **params}):
        return None
    return req_id


def request_patients():
    """환자 목록 요청 → 응답은 app.py가 to_patients()로 바꿔서 각 탭에 넘김"""
    return request('get_patients')


def to_patients(rows):
    """get_patients 응답(dict 리스트) → Patient 리스트"""
    return [
        Patient(
            patient_id=str(r['patient_id']),
            name=r['name'],
            age=r.get('age') or 0,
            room=r.get('room') or '',
            camera_id=r.get('camera_id'),
            caregiver_name=r.get('caregiver_name') or '',
        )
        for r in rows
    ]


def save_gait_session(patient_id, cumulative_scores, camera_id):
    """보행 결과 저장. camera_id: 측정한 카메라 (보행 탭에서 고른 카메라)"""
    return request('save_gait_session', patient_id=patient_id,
                   cumulative_scores=cumulative_scores, camera_id=camera_id)


def resolve_fall(camera_id, track_id, caregiver_name):
    """낙상 처리 완료 → main_server가 FallAnalyzer 상태 초기화 + fall_logs 처리완료 기록"""
    return request('resolve_fall', camera_id=camera_id, track_id=track_id,
                   caregiver_name=caregiver_name)
# admin_gui/db_stub.py
"""DB 요청 창구 — GUI는 DB에 직접 접속하지 않고, main_server에 요청을 보낸다.
main_server가 DBManager(main_service/db_manager.py)로 조회/저장한 뒤 결과를 TCP 9999로 돌려준다.

파일 이름과 함수 시그니처는 기존 스텁 그대로 유지 (fall_tab.py / gait_tab.py 호출부 수정 없음).

요청:  {"cmd": "<이름>", "req_id": n, ...인자}
응답:  {"type": "response", "cmd": "<이름>", "req_id": n, "ok": true/false, "data": ...}
       → app.py의 handle_server_message()가 받아서 처리
"""
import itertools

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
    """환자 목록 요청 → 응답은 app.py가 patients.set_patients()로 반영"""
    return request('get_patients')


def log_event(camera_id, track_id, event_name):
    """main_server가 FallAnalyzer 이벤트를 받는 즉시 DB에 직접 기록한다 → GUI에서는 할 일 없음"""
    pass


def resolve_fall_event(camera_id, track_id, patient_name, caregiver_name):
    """fall_tab.py가 이 호출 직후 보내는 'resolve_fall' 명령으로 main_server가 DB에 기록한다 → 할 일 없음"""
    pass


def save_gait_session(patient_id, cumulative_scores):
    return request('save_gait_session', patient_id=patient_id, cumulative_scores=cumulative_scores)
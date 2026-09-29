# admin_gui/db_stub.py
"""DB 연동 전 임시 스텁 — 실제 DB 스키마가 나오면 이 파일 안의 함수들만 교체하면 됩니다.

TODO(DB 연동 담당자): 여기에 작업하시면 됩니다.
  - log_event(): 낙상/주의 이벤트를 이벤트 테이블에 저장
  - save_gait_session(): 보행 분석 결과 저장
  - resolve_fall_event(): 낙상 처리 완료(환자명/요양보호사명) 기록
  함수 시그니처는 fall_tab.py / gait_tab.py가 그대로 호출하고 있으니
  인자 구성은 유지하고 내부 구현만 실제 DB 연동으로 바꿔주시면 됩니다.
"""
import time

def log_event(camera_id, track_id, event_name):
    print(f"[DB STUB] event camera={camera_id} track={track_id} event={event_name} "
          f"at={time.strftime('%H:%M:%S')}")

def resolve_fall_event(camera_id, track_id, patient_name, caregiver_name):
    print(f"[DB STUB] resolve camera={camera_id} track={track_id} "
          f"patient={patient_name} caregiver={caregiver_name} at={time.strftime('%H:%M:%S')}")

def save_gait_session(patient_id, cumulative_scores):
    print(f"[DB STUB] save gait patient={patient_id} scores={cumulative_scores} "
          f"at={time.strftime('%H:%M:%S')}")

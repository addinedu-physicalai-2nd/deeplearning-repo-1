# admin_gui/patient_tab_test.py
"""환자 관리 탭 단독 실행 (테스트용) — main_server 없이 MySQL에 직접 붙어서 화면 확인.

실행 (프로젝트 루트에서):
    export SILVERCARE_DB_PASSWORD=실제비번
    python -m admin_gui.patient_tab_test

합칠 때:
  - PatientTab은 그대로 app.py의 QTabWidget에 추가
  - DirectDBSource의 SQL은 main_service/db_manager.py의 get_patient_detail()로 옮기고,
    GUI 쪽은 db_client로 main_server에 요청해서 같은 dict 형식을 받는 source로 교체
    (GUI는 원래 DB에 직접 접속하지 않는 구조라서, 직접 연결은 이 테스트 파일에만 둔다)
"""
import sys
from datetime import date, datetime
from decimal import Decimal

from config.settings import CAMERA_PATIENTS, DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from .patient_tab import PatientTab
from .qt_compat import QApplication, QMainWindow

try:
    from .app import STYLE_SHEET, load_app_fonts
except Exception as e:       # app.py가 import하는 drawing.py가 작업 중이라 깨져 있어도 탭은 띄운다
    print(f"[PatientTabTest] app.py 스타일을 못 불러와서 기본 스타일로 실행: {e}")
    STYLE_SHEET = ""

    def load_app_fonts():
        pass


def _plain(row):
    """DB 값 → JSON으로도 보낼 수 있는 기본 타입 (나중에 main_server 경유로 바꿔도 형식이 같게)"""
    out = {}
    for key, value in row.items():
        if isinstance(value, (datetime, date)):
            out[key] = value.strftime('%Y-%m-%d %H:%M:%S') if isinstance(value, datetime) else value.isoformat()
        elif isinstance(value, Decimal):
            out[key] = float(value)
        else:
            out[key] = value
    return out


class DirectDBSource:
    """PatientTab용 데이터 소스 — hospital_monitoring DB 직접 조회 (테스트 전용)"""

    LOG_LIMIT = 20      # 기록 표에 보여줄 최근 건수

    def __init__(self):
        import pymysql
        self.pymysql = pymysql
        self.conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
                                    database=DB_NAME, charset='utf8mb4', autocommit=True,
                                    connect_timeout=3, cursorclass=pymysql.cursors.DictCursor)
        self.camera_by_patient = {pid: cam for cam, pid in CAMERA_PATIENTS.items()}

    def _query(self, sql, params=None):
        self.conn.ping(reconnect=True)
        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            return [_plain(r) for r in cur.fetchall()]

    def list_patients(self):
        try:
            rows = self._query("SELECT id, name, room_number FROM patients ORDER BY room_number, name")
        except Exception as e:
            print(f"[DirectDBSource] list_patients 실패: {e}")
            return None
        return [{'patient_id': r['id'], 'name': r['name'], 'room': r['room_number']} for r in rows]

    def get_patient_detail(self, patient_id):
        try:
            rows = self._query("SELECT * FROM patients WHERE id = %s", (patient_id,))
            if not rows:
                return None
            patient = rows[0]
            patient['camera_id'] = self.camera_by_patient.get(patient['id'])
            fall_logs = self._query(
                "SELECT log_id, camera_id, checking_at, alert_at, status, resolved_at FROM fall_logs "
                "WHERE patient_id = %s ORDER BY COALESCE(alert_at, checking_at) DESC LIMIT %s",
                (patient_id, self.LOG_LIMIT))
            gait_logs = self._query(
                "SELECT log_id, camera_id, checking_at, normal, abnormal, parkinsons, stroke, myopathic, antalgic "
                "FROM gait_logs WHERE patient_id = %s ORDER BY checking_at DESC LIMIT %s",
                (patient_id, self.LOG_LIMIT))
            stretch_logs = self._query(
                "SELECT log_id, camera_id, checking_at, straching_id, accuracy_id FROM stretch_logs "
                "WHERE patient_id = %s ORDER BY checking_at DESC LIMIT %s",
                (patient_id, self.LOG_LIMIT))
        except Exception as e:
            print(f"[DirectDBSource] get_patient_detail({patient_id}) 실패: {e}")
            return None
        return {'patient': patient, 'fall_logs': fall_logs,
                'gait_logs': gait_logs, 'stretch_logs': stretch_logs}


def main():
    app = QApplication(sys.argv)
    load_app_fonts()
    app.setStyleSheet(STYLE_SHEET)

    try:
        source = DirectDBSource()
    except Exception as e:
        print(f"[PatientTabTest] DB 연결 실패: {e}")
        print("  → SILVERCARE_DB_PASSWORD 환경변수와 MySQL 실행 상태를 확인하세요")
        sys.exit(1)

    tab = PatientTab(source)
    tab.refresh()

    window = QMainWindow()
    window.setWindowTitle("환자 관리 탭 테스트")
    window.setCentralWidget(tab)
    window.resize(1600, 900)
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
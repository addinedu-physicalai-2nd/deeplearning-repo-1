# main_service/db_manager.py
"""hospital_monitoring(MySQL) 접근 전담. main_server 프로세스만 DB에 접속하고,
GUI는 main_server에 요청 → main_server가 이 클래스로 조회/저장 → 결과를 GUI로 돌려준다.

모든 공개 메서드는 실패해도 예외를 밖으로 던지지 않는다 (로그만 남기고 None/[]/False 반환).
DB가 꺼져 있어도 모니터링(영상/분석)은 계속 돌아가야 하기 때문.
"""
import logging
import threading

from config.settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, CAMERA_PATIENTS

GAIT_COLUMNS = ['normal', 'abnormal', 'parkinsons', 'stroke', 'myopathic', 'antalgic']


class DBManager:
    def __init__(self, host=DB_HOST, port=DB_PORT, user=DB_USER,
                 password=DB_PASSWORD, database=DB_NAME, enabled=True):
        self.enabled = enabled                # False면 DB 없이 실행 (모든 메서드가 기본값 반환)
        self.config = dict(host=host, port=port, user=user, password=password,
                           database=database, charset='utf8mb4', autocommit=True,
                           connect_timeout=3)
        self.conn = None
        self.lock = threading.Lock()          # 커넥션 1개를 여러 스레드가 공유 → 쿼리 단위로 보호

        # 진행 중인 낙상 사건: (camera_id, track_id) → fall_logs.log_id
        # track_id는 AI 세션 한정 값이라 DB에 안 남기고 메모리에서만 추적
        self.open_falls = {}

        # 카메라 ↔ 환자 배정은 DB가 아니라 settings.CAMERA_PATIENTS에 있음
        self.camera_by_patient = {patient_id: cam_id for cam_id, patient_id in CAMERA_PATIENTS.items()}

        self.logger = logging.getLogger('DBManager')

    # ============ 공통 ============
    def _execute(self, sql, params=None, fetch=None):
        """쿼리 실행. fetch: None(쓰기) / 'one' / 'all'. 실패 시 예외를 그대로 올림 (호출부에서 처리)"""
        with self.lock:
            if self.conn is None:
                import pymysql                            # --no-db로 실행할 때는 설치 안 돼 있어도 되게 여기서 import
                self.conn = pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **self.config)
            else:
                self.conn.ping(reconnect=True)       # 끊겨 있으면 재접속
            with self.conn.cursor() as cur:
                cur.execute(sql, params)
                if fetch == 'one':
                    return cur.fetchone()
                if fetch == 'all':
                    return cur.fetchall()
                return cur.lastrowid

    def check_connection(self):
        if not self.enabled:
            self.logger.info("DB disabled (--no-db)")
            return False
        try:
            self._execute("SELECT 1", fetch='one')
            self.logger.info(f"DB connected: {self.config['host']}:{self.config['port']}/{self.config['database']}")
            return True
        except Exception as e:
            self.logger.warning(f"DB not available: {e}")
            return False

    def close(self):
        with self.lock:
            if self.conn is not None:
                self.conn.close()
                self.conn = None

    def _patient_row(self, row):
        return {
            'patient_id': row['id'],
            'name': row['name'],
            'age': row['age'],
            'room': row['room_number'],
            'camera_id': self.camera_by_patient.get(row['id']),
            'caregiver_name': row['caregiver_name'],
        }

    # ============ 환자 ============
    def get_patients(self):
        """전체 환자 목록. camera_id는 낙상(병실) 카메라 배정이고 없으면 None. 조회 실패 시 None (빈 목록과 구분)"""
        if not self.enabled:
            return None
        try:
            rows = self._execute(
                "SELECT id, name, age, room_number, caregiver_name FROM patients "
                "ORDER BY room_number, name", fetch='all')
            return [self._patient_row(r) for r in rows]
        except Exception as e:
            self.logger.error(f"get_patients failed: {e}")
            return None

    def get_patient_by_camera(self, camera_id):
        if not self.enabled:
            return None
        patient_id = CAMERA_PATIENTS.get(camera_id)
        if patient_id is None:
            return None
        try:
            row = self._execute(
                "SELECT id, name, age, room_number, caregiver_name FROM patients "
                "WHERE id = %s", (patient_id,), fetch='one')
            return self._patient_row(row) if row else None
        except Exception as e:
            self.logger.error(f"get_patient_by_camera({camera_id}) failed: {e}")
            return None

    # ============ 낙상 ============
    def log_fall_event(self, camera_id, track_id, event_name):
        """FallAnalyzer 이벤트 기록. '..._TO_CHECKING' → 사건 생성, '..._TO_ALERT' → 낙상 확정 시각 기록"""
        if not self.enabled:
            return False
        is_checking = event_name.endswith('_TO_CHECKING')
        is_alert = event_name.endswith('_TO_ALERT')
        if not (is_checking or is_alert):
            return False
        try:
            patient = self.get_patient_by_camera(camera_id)
            if patient is None:
                return False                          # 배정된 환자가 없는 카메라
            key = (camera_id, track_id)
            log_id = self.open_falls.get(key)

            if log_id is None:
                # 새 사건: checking이든 alert든 checking_at은 채움 (alert로 바로 오는 경우 대비)
                log_id = self._execute(
                    "INSERT INTO fall_logs (patient_id, camera_id, checking_at, alert_at, status) "
                    "VALUES (%s, %s, NOW(), " + ("NOW()" if is_alert else "NULL") + ", '대기')",
                    (patient['patient_id'], camera_id))
                self.open_falls[key] = log_id
            elif is_alert:
                self._execute("UPDATE fall_logs SET alert_at = NOW() WHERE log_id = %s", (log_id,))
            return True
        except Exception as e:
            self.logger.error(f"log_fall_event({camera_id}, {track_id}, {event_name}) failed: {e}")
            return False

    def resolve_fall_event(self, camera_id, track_id, caregiver_name):
        """처리 완료 기록. 추적 중인 사건이 없으면 그 카메라의 가장 최근 미처리 사건을 처리
        (fall_logs에 처리자 컬럼이 없어서 caregiver_name은 로그에만 남김)"""
        if not self.enabled:
            return False
        try:
            log_id = self.open_falls.pop((camera_id, track_id), None)
            if log_id is None:
                row = self._execute(
                    "SELECT log_id FROM fall_logs WHERE camera_id = %s AND resolved_at IS NULL "
                    "ORDER BY log_id DESC LIMIT 1", (camera_id,), fetch='one')
                if row is None:
                    return False
                log_id = row['log_id']
            self._execute(
                "UPDATE fall_logs SET status = '처리완료', resolved_at = NOW() WHERE log_id = %s", (log_id,))
            self.logger.info(f"fall log {log_id} resolved by {caregiver_name}")
            return True
        except Exception as e:
            self.logger.error(f"resolve_fall_event({camera_id}, {track_id}) failed: {e}")
            return False

    # ============ 보행 ============
    def save_gait_session(self, patient_id, cumulative_scores, camera_id=None):
        """cumulative_scores: gait_analyzer 출력 그대로 (0~1 비율) → DB에는 % (0~100)로 저장
        camera_id: 측정한 카메라 (GUI 보행 탭에서 고른 카메라). 없으면 병실 배정 카메라"""
        if not self.enabled:
            return False
        try:
            camera_id = camera_id or self.camera_by_patient.get(int(patient_id))
            percents = [round(float(cumulative_scores.get(col, 0.0)) * 100, 2) for col in GAIT_COLUMNS]
            self._execute(
                "INSERT INTO gait_logs (patient_id, camera_id, checking_at, "
                + ", ".join(GAIT_COLUMNS) + ") VALUES (%s, %s, NOW(), "
                + ", ".join(["%s"] * len(GAIT_COLUMNS)) + ")",
                (patient_id, camera_id, *percents))
            return True
        except Exception as e:
            self.logger.error(f"save_gait_session({patient_id}) failed: {e}")
            return False

    # ============ 스트레칭 ============
    def save_stretch_session(self, patient_id, camera_id, course_id, accuracy):
        """course_id: 루틴 번호 (파일명 앞 두 자리) → straching_id, accuracy: 세션 평균 점수 (0~100) → accuracy_id"""
        if not self.enabled:
            return False
        try:
            self._execute(
                "INSERT INTO stretch_logs (patient_id, camera_id, checking_at, straching_id, accuracy_id) "
                "VALUES (%s, %s, NOW(), %s, %s)",
                (patient_id, camera_id, course_id, round(float(accuracy), 2)))
            return True
        except Exception as e:
            self.logger.error(f"save_stretch_session({patient_id}) failed: {e}")
            return False
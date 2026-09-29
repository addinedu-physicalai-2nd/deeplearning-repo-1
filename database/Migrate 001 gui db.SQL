-- database/migrate_001_gui_db.sql
-- dbmanager 함수 명세서 기준 스키마 보강 (init_hospital_db.sql 실행 후 1회 적용)
USE hospital_monitoring;

-- 1. 환자 ↔ 카메라 1:1 배정 (CAM-01 ~ CAM-05, config/settings.py CAMERA_PORTS 키)
ALTER TABLE patients
    ADD COLUMN camera_id VARCHAR(20) NULL COMMENT '배정된 카메라 (CAM-01~CAM-05)';

-- 2. 낙상 실제 처리자 (배정된 요양보호사와 다를 수 있음)
ALTER TABLE fall_logs
    ADD COLUMN resolved_by VARCHAR(50) NULL COMMENT '실제 처리한 요양보호사';

-- 3. 스트레칭 루틴 번호 체계 변경 (01~23, stretching 폴더 파일명 앞 번호) + 이름
ALTER TABLE stretch_logs
    MODIFY COLUMN straching_id INT COMMENT '스트레칭 루틴 번호 (01~23, 파일명 앞 번호)';
ALTER TABLE stretch_logs
    ADD COLUMN course_name VARCHAR(50) NULL COMMENT '스트레칭 루틴 이름 (파일명 기반)';

-- 4. (테스트용) 카메라 배정 예시 — 실제 병상 배정에 맞게 수정
UPDATE patients SET camera_id = 'CAM-01' WHERE id = 1;
UPDATE patients SET camera_id = 'CAM-02' WHERE id = 2;
UPDATE patients SET camera_id = 'CAM-03' WHERE id = 3;
UPDATE patients SET camera_id = 'CAM-04' WHERE id = 4;
UPDATE patients SET camera_id = 'CAM-05' WHERE id = 5;
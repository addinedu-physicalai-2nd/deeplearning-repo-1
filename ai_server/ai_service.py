# ai_server/ai_service.py
from .ai_networkmanager import AINetworkManager
from .ai_manager import AIManager
from .yolo_pose import YOLOPoseWrapper
from config.settings import AI_SERVER_PORT, CAMERA_PORTS

import threading
import logging

class AIService:
    """AI 서비스 (병렬 처리)"""
    
    def __init__(self, device='cuda'):
        self.net = AINetworkManager(udp_port=AI_SERVER_PORT)
        
        # YOLOPose는 1번만 생성해서 모든 AIManager가 공유
        self.yolo_pose = YOLOPoseWrapper(device=device)
        
        self.ai_managers = {}
        self.device = device
        
        self.logger = logging.getLogger('AIService')
        logging.basicConfig(level=logging.INFO)
    
    def run(self):
        """메인 루프"""
        self.net.start()
        self.logger.info("AIService started on UDP port 9100")
        
        # camera_id별 스레드 생성
        for camera_id in CAMERA_PORTS.keys():
            ai_mgr = AIManager(
                camera_id=camera_id,
                yolo_pose=self.yolo_pose,  # ← 공유!
                device=self.device
            )
            self.ai_managers[camera_id] = ai_mgr
            
            threading.Thread(
                target=self._process_camera,
                args=(camera_id,),
                daemon=True
            ).start()
            self.logger.info(f"Started thread for {camera_id}")
        
        # 메인 스레드는 계속 실행 상태 유지
        while True:
            threading.Event().wait(1)
    
    def _process_camera(self, camera_id: str):
        """camera_id별 처리 스레드"""
        ai_mgr = self.ai_managers[camera_id]
        
        while True:
            try:
                payload = self.net.get_frame()
                
                # 자신의 카메라 frame만 처리
                if payload['camera_id'] != camera_id:
                    continue
                
                mode = payload['mode']
                frame = payload['frame']
                
                ai_mgr.set_mode(mode)
                json_result = ai_mgr.process_frame(frame)
                self.net.send_to_main(json_result)
                
            except Exception as e:
                self.logger.error(f"[{camera_id}] Error: {e}")

if __name__ == '__main__':
    service = AIService()
    service.run()
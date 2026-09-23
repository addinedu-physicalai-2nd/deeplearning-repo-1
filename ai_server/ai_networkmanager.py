# ai_server/ai_network_manager.py
import socket
import json
import requests
from queue import Queue
import threading
import logging
from config.settings import (
    AI_SERVER_PORT,
    MAIN_SERVICE_HOST,
    MAIN_SERVICE_PORT
)

class AINetworkManager:
    """네트워크 통신 (UDP 수신, HTTP 송신)"""
    
    REQUEST_TIMEOUT = 5
    
    def __init__(self, udp_port=AI_SERVER_PORT, 
                 main_host=MAIN_SERVICE_HOST, main_port=MAIN_SERVICE_PORT):
        self.udp_port = udp_port
        self.main_host = main_host
        self.main_port = main_port
        self.frame_queue = Queue()
        self.is_running = False
        
        self.logger = logging.getLogger('AINetworkManager')
        logging.basicConfig(level=logging.INFO)
    
    def start(self):
        """UDP 수신 시작"""
        self.is_running = True
        threading.Thread(target=self._receive_loop, daemon=True).start()
        self.logger.info(f"AINetworkManager listening on UDP port {self.udp_port}")
    
    def _receive_loop(self):
        """UDP 수신 루프"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.udp_port))
        
        while self.is_running:
            try:
                data, addr = sock.recvfrom(65535)
                payload = json.loads(data.decode('utf-8'))
                
                camera_id = payload.get('camera_id')
                mode = payload.get('mode')
                frame = payload.get('frame')
                
                if not all([camera_id, mode is not None, frame]):
                    self.logger.warning(f"[{addr}] Invalid payload")
                    continue
                
                self.frame_queue.put({
                    'camera_id': camera_id,
                    'mode': mode,
                    'frame': frame
                })
                self.logger.debug(f"Received frame from {camera_id} (mode={mode})")
                
            except json.JSONDecodeError:
                self.logger.error(f"[{addr}] JSON decode error")
            except Exception as e:
                self.logger.error(f"[{addr}] Receive error: {e}")
    
    def get_frame(self):
        """frame 가져오기 (blocking)"""
        return self.frame_queue.get()
    
    def send_to_main(self, json_result):
        """결과 송신 (HTTP POST)"""
        try:
            url = f"http://{self.main_host}:{self.main_port}/ai_result"
            response = requests.post(
                url,
                json=json_result,
                headers={'Content-Type': 'application/json'},
                timeout=self.REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                self.logger.debug(f"Sent result from {json_result.get('camera_id')}")
                return True
            else:
                self.logger.warning(f"Main returned status {response.status_code}")
                return False
                
        except requests.exceptions.Timeout:
            self.logger.error("Main connection timeout")
            return False
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Main connection refused ({self.main_host}:{self.main_port})")
            return False
        except Exception as e:
            self.logger.error(f"Send error: {e}")
            return False
    
    def stop(self):
        """종료"""
        self.is_running = False
        self.logger.info("AINetworkManager stopped")
import cv2
import numpy as np
import threading
import time
from typing import Optional, Dict


class SimpleCamera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._connected = False
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._read_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def connect(self) -> bool:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        backend_names = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "ANY"}
        
        for backend in backends:
            for attempt in range(3):
                try:
                    self._cap = cv2.VideoCapture(self._index, backend)
                    if self._cap.isOpened():
                        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
                        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        time.sleep(0.2)
                        ret, frame = self._cap.read()
                        if ret:
                            print(f"Camera {self._index} connected with {backend_names[backend]} backend")
                            self._latest_frame = frame
                            self._stop_event.clear()
                            self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
                            self._read_thread.start()
                            self._connected = True
                            return True
                    
                    if self._cap is not None:
                        self._cap.release()
                    
                    if attempt < 2:
                        time.sleep(0.3)
                except Exception as e:
                    if self._cap is not None:
                        self._cap.release()
        
        print(f"Failed to connect camera {self._index}")
        return False

    def _read_loop(self):
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if ret:
                with self._frame_lock:
                    self._latest_frame = frame
            time.sleep(0.001)

    def read(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
            return None

    def disconnect(self):
        self._stop_event.set()
        if self._read_thread is not None:
            self._read_thread.join(timeout=1.0)
            self._read_thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected


class CameraManager:
    def __init__(self, camera_configs: Dict[str, dict]):
        self._configs = camera_configs
        self._cameras: Dict[str, SimpleCamera] = {}
        self._connected = False

    def connect(self) -> bool:
        for name, cfg in self._configs.items():
            print(f"Connecting camera: {name}")
            cam = SimpleCamera(
                index=cfg.get("index", 0),
                width=cfg.get("width", 640),
                height=cfg.get("height", 480),
                fps=cfg.get("fps", 30)
            )
            if not cam.connect():
                print(f"  Failed to connect camera: {name}")
                continue
            self._cameras[name] = cam
            time.sleep(0.3)
        self._connected = len(self._cameras) > 0
        return self._connected

    def get_images(self) -> Dict[str, np.ndarray]:
        images = {}
        for name, cam in self._cameras.items():
            frame = cam.read()
            if frame is not None:
                images[name] = frame
        return images

    def disconnect(self):
        for cam in self._cameras.values():
            cam.disconnect()
        self._cameras.clear()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected


def test_camera(index: int = 0):
    print(f"Testing camera {index}...")
    
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    backend_names = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "ANY"}
    
    cap = None
    for backend in backends:
        for attempt in range(3):
            print(f"Trying {backend_names[backend]} backend, attempt {attempt + 1}...")
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                time.sleep(0.2)
                ret, frame = cap.read()
                if ret:
                    print(f"Camera opened with {backend_names[backend]} backend")
                    break
                cap.release()
                cap = None
            
            if attempt < 2:
                time.sleep(0.3)
        
        if cap is not None and cap.isOpened():
            break
    
    if cap is None or not cap.isOpened():
        print(f"Failed to open camera {index}")
        return
    
    print(f"Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print("Reading frames (press 'q' to quit)...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            cv2.imshow(f"Camera {index}", frame)
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: OK")
        else:
            print(f"Frame read failed at count {frame_count}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Total frames: {frame_count}")


if __name__ == "__main__":
    test_camera(0)

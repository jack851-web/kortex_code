# -*- coding: utf-8 -*-
"""
多相机同时预览工具 - 解决 Windows USB 带宽冲突问题
用法: python test_multi_cam.py
按 q 退出
"""

import cv2
import threading
import time


class ThreadedCamera:
    """独立线程读取摄像头，避免阻塞主循环"""

    def __init__(self, index, width=640, height=480, fps=15):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.frame = None
        self.running = False
        self.opened = False

    def start(self):
        # 尝试多种后端
        backends = [
            cv2.CAP_DSHOW,
            cv2.CAP_MSMF,
            cv2.CAP_ANY,
        ]
        for backend in backends:
            self.cap = cv2.VideoCapture(self.index, backend)
            if self.cap.isOpened():
                break

        if not self.cap.isOpened():
            print(f'  [FAIL] Camera {self.index}: 无法打开')
            return False

        # 关键：设置 MJPEG 硬件压缩，大幅降低带宽
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 验证实际参数
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = ''.join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        print(f'  [OK]   Camera {self.index}: {actual_w}x{actual_h} '
              f'(codec={fourcc_str}, backend={self.cap.getBackendName()})')

        self.running = True
        self.opened = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        return True

    def _read_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        if hasattr(self, 'cap'):
            self.cap.release()


def main():
    print('=' * 50)
    print('多相机测试 - 识别外接 vs 内置相机')
    print('=' * 50)
    print()

    # 扫描所有可能的 index（0-9），全部同时打开
    configs = []
    for idx in range(10):
        configs.append({'index': idx, 'name': f'Camera-{idx}', 'width': 480, 'height': 360})

    cameras = []
    for cfg in configs:
        cam = ThreadedCamera(
            index=cfg['index'],
            width=cfg['width'],
            height=cfg['height'],
            fps=10,
        )
        if cam.start():
            cameras.append((cfg['name'], cam))

    if len(cameras) == 0:
        print('\n没有找到任何相机')
        return

    print(f'\n成功打开 {len(cameras)} 个相机')
    print('请观察画面，确认哪些是 WHEELTEC 外接相机\n')

    try:
        while True:
            for name, cam in cameras:
                frame = cam.read()
                if frame is not None:
                    cv2.imshow(name, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        print('\n关闭所有相机...')
        for _, cam in cameras:
            cam.stop()
        cv2.destroyAllWindows()
        print('Done')


if __name__ == '__main__':
    main()

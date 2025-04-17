# This is client code to receive video frames
import sys
import cv2, socket
import numpy as np
import time
import base64
from threading import Thread
import json
import h264decoder
import glob
from patterntrack import *
from pattern_detection import *

class CameraStream:
    def __init__(self, path, host_ip='192.168.1.105'):
        self.initialize_socket(host_ip)
        self.initialize_camera_parameters(path)
        self.frame = np.zeros((1440, 1920, 3)).astype(np.uint8)
        self.stable = False
        self.start_camera_thread()

    def initialize_socket(self, host_ip):
        BUFF_SIZE = 65536
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
        port = 9999
        message = b'Hello'
        client_socket.sendto(message, (host_ip, port))
        self.client_socket = client_socket
        self.host_ip = host_ip
        self.port = port
        self.BUFF_SIZE = BUFF_SIZE
        self.fps, self.st, self.frames_to_count, self.cnt = (0, 0, 20, 0)

    def initialize_camera_parameters(self, path):
        homography = np.load(path + 'homography.npz')
        self.hmtx = homography['mtx']
        self.size = homography['size']
        intrinsics = np.load(path + "intrinsic_calib.npz")
        self.mtx = intrinsics['mtx']
        self.dist = intrinsics['dist']
        json_data = open(path + "per_instrinsic.json").read()
        data = json.loads(json_data)
        self.scaled_K, self.D, self.new_K, self.dim3 = np.asarray(data['scaled_K']), np.asarray(data['D']), np.asarray(data['new_K']), tuple(data['dim3'])
        self.scaled_K_ = np.copy(self.scaled_K)
        self.Knew = self.scaled_K.copy()
        self.new_K[(0, 1), (0, 1)] = 2 * self.Knew[(0, 1), (0, 1)]
        self.calibration_image = None

    def start_camera_thread(self):
        cam_thread = Thread(target=self.pull_frame_thread, args=[])
        cam_thread.start()

    def pull_frame_thread(self):
        decoder = h264decoder.H264Decoder()
        while True:
            self.request_frame()
            packet, _ = self.client_socket.recvfrom(self.BUFF_SIZE)
            frame = self.decode_frame(packet, decoder)
            self.process_frame(frame)

    def request_frame(self):
        message = b'giveme'
        self.client_socket.sendto(message, (self.host_ip, self.port))

    def decode_frame(self, packet, decoder):
        framedatas = decoder.decode(packet)
        frame = None
        for framedata in framedatas:
            (frame, w, h, ls) = framedata
            frame = np.frombuffer(frame, dtype=np.ubyte, count=len(frame))
            frame = frame.reshape((h, ls // 3, 3))
            frame = frame[:, :w, :]
        return frame

    def process_frame(self, frame):
        if self.cnt == self.frames_to_count:
            self.update_fps()
        self.cnt += 1
        if frame is not None:
            frame = self.prepare_frame(frame)
            self.frame = frame
            self.stable = True
        else:
            self.stable = False

    def update_fps(self):
        try:
            self.fps = round(self.frames_to_count / (time.time() - self.st))
            self.st = time.time()
            self.cnt = 0
        except:
            self.stable = False

    def prepare_frame(self, frame):
        frame = cv2.resize(frame, (1440, 810))
        frame = self.undistort(frame)
        frame = self.transform(frame)
        return frame

    def read(self):
        return True, self.frame

    def undistort(self, frame):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.scaled_K, self.D, np.eye(3), self.scaled_K_, self.dim3, cv2.CV_16SC2)
        u, v = map1[:, :, 0].astype(np.float32), map1[:, :, 1].astype(np.float32)
        undistorted = cv2.remap(frame, u, v, cv2.INTER_LINEAR)
        return undistorted

    def transform(self, image):
        img_transed = cv2.warpPerspective(image, self.hmtx, self.size)
        return img_transed

def get_newfile_num(path, subfix='', prefix=''):
    filenames = glob.glob(path + '*')
    num = 0
    while f'{path}{prefix}{num}{subfix}' in filenames:
        num += 1
    return num

# Test code showing the camera stream and pattern detection
def main():
    cam = CameraStream("../data/iphone1/")
    issave = False
    images = []
    center_stack = []
    video_save = None
    cnt = 0

    while True:
        ret, frame = cam.read()
        frame_ = np.copy(frame)
        frame_th = get_threshold_image(frame_)
        frame_filtered = frame_th / 255
        frame_filtered = cv2.resize(frame_filtered, (frame_.shape[1], frame_.shape[0]))
        frame_filtered = (frame_filtered * 255).astype(np.uint8)
        center, contours = detect_pattern(frame_th)
        frame_th = cv2.cvtColor(frame_th, cv2.COLOR_GRAY2RGB)
        frame_filtered = cv2.cvtColor(frame_filtered, cv2.COLOR_GRAY2RGB)
        frame_filtered = put_pattern_on_image(frame_filtered, center)
        frame_ = put_pattern_on_image(frame_, center)
        frame_th = put_pattern_on_image(frame_th, center, contours)
        frame_simple = put_pattern_on_image(np.zeros(frame_.shape).astype(np.uint8), center)
        frame_th_ = np.zeros(frame_th.shape)
        frame_th_[200:, :] = frame_th[200:, :]
        frame_th = frame_th_.astype(np.uint8)
        image = np.hstack((frame_, frame_filtered, frame_simple, frame_th))
        cv2.imshow('Camera Stream', image)

        if video_save is not None and cam.stable:
            print("recording video...")
            video_save.write(frame)
            cnt += 1
            if cnt > 400:
                cnt = 0
                video_save.release()
                video_save = None

        mkey = cv2.waitKey(1)
        if mkey == ord('q'):
            cam.client_socket.close()
            print('[User command] Exit')
            break
        elif mkey == ord('r'):
            video_save = handle_video_recording(video_save, frame)
        elif mkey == ord(' '):
            save_frame(frame)
        elif mkey == ord('c'):
            cam.calibration_image = frame

def handle_video_recording(video_save, frame):
    if video_save is None:
        num = get_newfile_num('../data/captured/ir/', '.mp4')
        name = '../data/captured/ir/{}.mp4'.format(str(num).zfill(5))
        video_save = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'avc1'), 10, (frame.shape[1], frame.shape[0]))
    else:
        video_save.release()
        video_save = None
    return video_save

def save_frame(frame):
    num = get_newfile_num('../data/captured/ir/', '.jpg')
    cv2.imwrite('../data/captured/ir/{}.jpg'.format(str(num).zfill(5)), frame)
    print("Saved", num)

if __name__ == "__main__":
    main()
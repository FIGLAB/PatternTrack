#! /usr/bin/env python
from subprocess import call
call(['espeak “Welcome to the world of Robots” 2>/dev/null'], shell=True)
import socket
import numpy as np
import time
import base64
import time
from rapid_processing import *
from threading import Thread

framerate = 18

ratio = 3/4


def data_send_thread(server_socket,encoded_buf,client_addr,):
    starttime = time.time()
    fps = []
    timeout_cnt = 0
    while True:
        if len(encoded_buf) == 0:
            continue
        elif len(encoded_buf) > framerate*2:
            del encoded_buf[:]
            print("Reset image queue")
            continue
        try:
            msg, client_addr = server_socket.recvfrom(BUFF_SIZE)
            timeout_cnt = 0
        except Exception as e:
            print("error:",e)
            if str(e) == "timed out":
                timeout_cnt += 1
                if timeout_cnt > 3:
                    print("Waiting for new connection")
                    break
        message = encoded_buf.pop(0)
        try:
            server_socket.sendto(message,client_addr)
        except Exception as e:
            print(e)
            message = b'\x00\x00\x00\x01'
            server_socket.sendto(message,client_addr)
        t = time.time()
        fps_ = 1/(t - starttime)
        fps.append(fps_)
        if len(fps) > 10:
            fps.pop(0)
        print("{} FPS, In queue: #{}".format(int(np.mean(fps)),len(encoded_buf)))
        starttime = time.time()


if __name__ == "__main__":

    # with picamera.PiCamera(resolution='VGA') as camera:
    with picamera.PiCamera() as camera:
        width,height = 1920,1080
        width,height = int(width*ratio),int(height*ratio)
        width,height = 640, 360
        camera.resolution = (width, height)
        camera.color_effects = (1,1)
        camera.framerate = framerate
        # camera.exposure_compensation = -25
        camera.shutter_speed = 10000
        # camera.shutter_speed = 60000
        # camera.shutter_speed = 100000
        camera.start_preview()
        time.sleep(2)
        output = ProcessOutput()
        # camera.start_recording(output, format='mjpeg')
        camera.start_recording(output, format='h264')
        while True:
            BUFF_SIZE = 655360
            server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,BUFF_SIZE)
            host_name = socket.gethostname()
            host_ip = '0.0.0.0'#  socket.gethostbyname(host_name)
            print(host_ip)
            port = 9999
            socket_address = (host_ip,port)
            server_socket.bind(socket_address)
            server_socket.settimeout(1.0)
            print('Listening at:',socket_address)

            try:
                msg,client_addr = server_socket.recvfrom(BUFF_SIZE)
            except Exception as e:
                print("error:",e)
                continue
            print('GOT connection from ',client_addr)

            del output.encoded_buf[:]
            t = Thread(target=data_send_thread, args=(server_socket,output.encoded_buf,client_addr,))
            t.start()
            while not output.done and t.is_alive():
                camera.wait_recording(1/(framerate*5))
            server_socket.close()
    camera.stop_recording()


import io
import time
import threading
import picamera
# import cv2
import numpy as np
from PIL import Image
from multiprocessing import Process
# import matplotlib.pyplot as plt

class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()
        #self.starttime = time.time()

    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    #Image.open(self.stream)
                    #...
                    #...
                    # Set done to True if you want the script to terminate
                    # at some point
                    #self.owner.done=True
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)
                #endtime = time.time()
                #print("frame rate: {} fps".format(round(1/(endtime - self.starttime))))
                #self.starttime = time.time()

class ProcessOutput(object):
    def __init__(self):
        self.done = False
        # Construct a pool of 4 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(4)]
        self.processor = None
        self.images = []
        self.encoded_buf = []
        #self.vis_thread = threading.Thread(target=self.visualize, args=(self.images,))
        #self.vis_thread.start()
        self.starttime = time.time()
        self.T = []

    def write(self, buf):
        # if buf.startswith(b'\xff\xd8'): # mjpeg 
        if buf.startswith(b'\x00\x00\x00\x01'): # h264
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.event.set()
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    self.processor = None
        if self.processor:
            self.processor.stream.write(buf)
            #image = Image.open(io.BytesIO(buf))
            #self.images.append(image)
            #img = np.asarray(image)
            #_,buf = cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,80]) # slow 0.025s
            self.encoded_buf.append(buf)
            #image = np.asarray(image)
            #cv2.imshow('image',image)
            #cv2.waitKey(1)
            endtime = time.time()
            t = endtime - self.starttime
            self.T.append(t)
            if len(self.T) > 30:
                self.T.pop(0)
            t = np.mean(np.array(self.T))
            #print("frame rate: {} fps".format(round(1/t)))
            self.starttime = time.time()

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass # pool is empty
            proc.terminated = True
            proc.join()
            
def visualize(images):
    #fig, ax = plt.subplots(1,1)
    del images[:]
    while True:
        #plt.cla()
        if len(images)>0:
            image = images.pop(0)
            # del images[:]
            #print(len(images))
            image = np.asarray(image)
            #ax.imshow(image)
            #plt.pause(0.0001)
            cv2.imshow("display",image)
            mkey = cv2.waitKey(1)
        

if __name__ == "__main__":
    # with picamera.PiCamera(resolution='VGA') as camera:
    with picamera.PiCamera() as camera:
        camera.resolution = (600, 400)
        camera.framerate = 60 # 5
        #camera.exposure_compensation = -10
        #camera.shutter_speed = 100000000
        camera.start_preview()
        time.sleep(2)
        output = ProcessOutput()
        #camera.start_recording(output, format='mjpeg')
        camera.start_recording(output, format='h264')
        #t = threading.Thread(target=visualize, args=(output.images,),daemon=True) 
        #t.start()
        starttime = None
        while not output.done:
            camera.wait_recording(1/60)
            #print(len(output.images))
            if len(output.images) == 0:
                continue
            frame = output.images.pop(0)
            #frame = np.asarray(frame)
            #cv2.imshow('frame',frame)
            #cv2.waitKey(1)
        camera.stop_recording()

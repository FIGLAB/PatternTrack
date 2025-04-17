import websockets
import random
import multiprocessing
import time
from websockets.sync.server import serve
from scipy.spatial.transform import Rotation as R
import numpy as np

class WebSocketServer:
    def __init__(self,port=8080,ismain=1):
        self.matrice = np.zeros((2,4,4))
        self.mtx_dev2dev = np.eye(4)
        self.ismain = ismain
        self.start_output_server(port=port)

    def send_data(self, data):
        self.send_queue.put(data)

    def receive_data(self):
        if self.receive_queue.empty():
            return None
        data = self.receive_queue.get()
        return data

    def send_and_receive(self, websocket, send_queue, receive_queue):
        while True:
            try:
                self._send_data(websocket, send_queue)
                self._receive_data(websocket,  receive_queue)
                time.sleep(0.01)
            except websockets.exceptions.ConnectionClosed:
                break

    def _send_data(self, websocket, queue):
        # if not queue.empty():
        ray_data = queue.get()  # Get data from the queue
        data_string = ",".join(map(str, ray_data))
        websocket.send(data_string)

    def _receive_data(self, websocket, queue):
        message = websocket.recv()
        float_array = [float(x) for x in message.split(',')]
        float_array = np.array(float_array)
        try:
            float_array = float_array.reshape(2,4,4)
            float_array[0] = float_array[0].T
            float_array[1] = float_array[1].T
            queue.put(float_array)
        except:
            print("Wrong shape of received data")

    def handle_connection(self, websocket, send_queue, receive_queue):
        print("Client connected")
        try:
            self.send_and_receive(websocket, send_queue, receive_queue)
        finally:
            print("Client disconnected")

    def run_server(self, send_queue,receive_queue,port):
        def wrapped_handle_connection(websocket):
            return self.handle_connection(websocket, send_queue, receive_queue)
        
        with serve(wrapped_handle_connection, "0.0.0.0", port) as server:
            print(f"WebSocket server started on ws://0.0.0.0:{port}")
            server.serve_forever()

    def start_output_server(self, port):
        # Create a queue for inter-process communication
        send_queue = multiprocessing.Queue(maxsize=3)
        receive_queue = multiprocessing.Queue(maxsize=3)
        # Start the WebSocket server in a separate process
        server_process = multiprocessing.Process(target=self.run_server, args=(send_queue,receive_queue,port,),daemon=True)
        server_process.start()
        # return send_queue, receive_queue
        self.send_queue = send_queue
        self.receive_queue = receive_queue

    def format_ray_data(self, pos, rmat):
        r = R.from_matrix(rmat)
        rot = r.as_rotvec()
        pos = np.array(pos)
        pos = pos/1000
        start = np.array([pos[0],-pos[1],-pos[2]])
        end = rmat @ np.array([0,0,1])
        vec = np.array([end[0],-end[1],-end[2]])
        data = np.concatenate((start, vec, rmat.flatten()))
        return data

    def format_object_transform(self, matrice):
        mtx_dev_in_world = matrice[0]
        mtx_obj_in_world = matrice[1]
        try:
            m = np.linalg.inv(self.matrice[0]) @ self.matrice[1]
            r = self.mtx_dev2dev[:3,:3] @ m[:3,:3]
            t = self.mtx_dev2dev[:3,-1] + m[:3,-1]
            pose = np.eye(4)
            pose[:3,:3] = r
            pose[:3,-1] = t
            # pose = self.mtx_dev2dev @ np.linalg.inv(mtx_dev_in_world) @ mtx_obj_in_world
        except Exception as e:
            pose = np.zeros((4,4))
        return pose

    def generate_data(self, pos, rmat):
        _rmat = np.eye(3)
        _rmat[:,0], _rmat[:,1], _rmat[:,2] = rmat[:,1], rmat[:,0], rmat[:,2]
        # _rmat = _rmat.T
        self.mtx_dev2dev[:3,:3] = _rmat
        tvec = pos
        self.mtx_dev2dev[:3,-1] = np.array([tvec[0],-tvec[1],-tvec[2]])/1000
        ray_data = self.format_ray_data(pos, rmat)
        pose = self.format_object_transform(self.matrice)
        data = np.concatenate((ray_data,pose.T.flatten(),[self.ismain]))
        return data

    def update_matrice(self,matrice):
        if np.all(matrice[1] == self.matrice[1]):
            self.ismain = 1
        else:
            self.ismain = 0
        self.matrice = matrice

def generate_random_ray_data():
    # return [random.uniform(-1, 1) for _ in range(6)]
    ray = np.concatenate((np.zeros(3), np.array([0,0,-1])*0.1))
    return ray

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update3dplot(mtx):
    ax.cla()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    p = mtx[:3,-1]
    x,y,z = mtx[:3,0]*0.1,mtx[:3,1]*0.1,mtx[:3,2]*0.1
    ax.plot([p[0],p[0]+x[0]],[p[1],p[1]+x[1]],[p[2],p[2]+x[2]],c='red')
    ax.plot([p[0],p[0]+y[0]],[p[1],p[1]+y[1]],[p[2],p[2]+y[2]],c='green')
    ax.plot([p[0],p[0]+z[0]],[p[1],p[1]+z[1]],[p[2],p[2]+z[2]],c='blue')

    ax.plot([0,0.1],[0,0],[0,0],c='darkred')
    ax.plot([0,0],[0,0.1],[0,0],c='darkgreen')
    ax.plot([0,0],[0,0],[0,0.1],c='darkblue')
    ax.set_aspect('equal')
    plt.pause(0.01)

if __name__ == "__main__":
    ws_main = WebSocketServer(port=8080,ismain=1)
    ws_follower = WebSocketServer(port=4000,ismain=0)
    print("Main process is generating ray data and doing other computations")

    cnt = 0
    try:
        while True:
            # Generate ray data in the main process
            # ray_data = generate_random_ray_data()
            
            # Put the data in the queue for the server process
            cnt = (cnt + 1) % 10
            # pos = np.zeros(3)
            rmat = np.eye(3)
            rmat = R.from_rotvec([0, np.pi/180*30, 0]).as_matrix()
            # rmat = np.array([[-1,    0     ,0],   
            #                  [0,    -1     ,0],
            #                  [0,     0     ,1]])
            # rmat = np.array([[1,    0     ,0],   
            #                  [0,    0.7071, -0.7071],
            #                  [0,    0.7071,  0.7071]])
            # rmat = np.array([[1,    0     ,0],   
            #                  [0,    0.7071, 0.7071],
            #                  [0,    -0.7071,  0.7071]])
            pos = np.array([-120-cnt/10,0,0])
            # rmat = np.array([[-0.72083165,0.67531824,-0.07635498],
            #                  [-0.52902886,-0.63160557,-0.55854796],
            #                  [-0.43053254,-0.36978925, 0.81203075]])
            # pose = np.array([[ 0.00899117 ,-0.79268542 , 0.60956452 , 0.00717257],
            #                  [ 0.99985117 , 0.01610503 , 0.00619524 ,-0.00383975],
            #                  [-0.01472793 , 0.60941809 , 0.79271227 ,-0.3920132 ],
            #                  [ 0.         , 0.         , 0.         , 0.99999994]])
            data_main = ws_main.generate_data(pos, rmat)
            data_follower = ws_follower.generate_data(-pos, rmat.T)

            ws_main.send_data(data_main)
            ws_follower.send_data(data_follower)

            matrice1 = ws_main.receive_data()
            matrice2 = ws_follower.receive_data()
            if matrice1 is not None:
                ws_follower.update_matrice(matrice1)
            if matrice2 is not None:
                ws_main.update_matrice(matrice2)

            # Perform other computations here
            # print("Generated new ray data and doing other work in the main process...")
            time.sleep(0.01)  # Generate new data every second

    except KeyboardInterrupt:
        print("Main process interrupted. Stopping server...")
        # server_process.terminate()
        # server_process.join()
        # print("Server stopped.")
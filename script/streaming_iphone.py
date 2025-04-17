# This is client code to receive video frames
import sys
sys.path.append('../')
import cv2, socket
import numpy as np
import open3d as o3d
import time
import base64
from threading import Thread
import json
import io
import h264decoder
from threading import Thread
import matplotlib.pyplot as plt
import glob
import zlib

# host = "10.0.0.178" # home
# host = "192.168.0.202" # lab
host = "0.0.0.0"
port = 12005  # initiate port no above 1024

# fig = plt.figure("Pointcloud")
# ax = fig.add_subplot(111, projection='3d')

# Camera parameters
size_rgb = (1920,1440)
size_depth = (960,760)
scaleRes = np.array([size_rgb[0]/size_depth[0], size_rgb[1]/size_depth[1]])
droprate = 5
size_pc = (int(size_depth[0]/droprate), int(size_depth[1]/droprate))
# size_pc = (96, 76)

# Initialize visualizer
vis = o3d.visualization.Visualizer()
vis.create_window("iPhone Point Cloud",width=1000, height=1000)

# Initial point cloud for setting up
pcd = o3d.geometry.PointCloud()
pcd_pattern = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(np.random.rand(190 * 240, 3)*1000+np.array([-500,-500,0])) # 760 960
pcd.points = o3d.utility.Vector3dVector(np.random.rand(size_pc[1] * size_pc[0], 3)*2000+np.array([-1000,-1000,0])) # 760 960
pcd_pattern.points = o3d.utility.Vector3dVector(np.random.rand(64*9, 3))
# origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
# origin.scale(200, center=origin.get_center())


# origin = o3d.geometry.TriangleMesh.create_box(width=220, height=120, depth=40)
origin = o3d.geometry.TriangleMesh.create_arrow(cylinder_split=40, cone_split=10)
origin.scale(20, center=origin.get_center())
origin.translate([-110,-60,-20])
origin.compute_vertex_normals()
origin.paint_uniform_color([0,0.635,1])

vis.add_geometry(origin)
vis.add_geometry(pcd)
vis.add_geometry(pcd_pattern)

# set camera view of vis
ctr = vis.get_view_control()
camera_params = ctr.convert_to_pinhole_camera_parameters()
camera_params.extrinsic = np.array([
[-4.67363562e-02, -9.49732206e-01,  3.09555244e-01, -9.94521213e+01],
[ 9.30939351e-01, -1.53769351e-01, -3.31220336e-01,  3.65518440e+02],
[ 3.62170729e-01,  2.72697127e-01,  8.91329703e-01,  2.27826140e+03],
[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
ctr.convert_from_pinhole_camera_parameters(camera_params)

view1 = np.array([[-4.67363562e-02, -9.49732206e-01,  3.09555244e-01, -9.94521213e+01],
                 [ 9.30939351e-01, -1.53769351e-01, -3.31220336e-01,  3.65518440e+02],
                 [ 3.62170729e-01,  2.72697127e-01,  8.91329703e-01,  2.27826140e+03],
                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
view2 = np.array([[-4.86993181e-02, -8.56147077e-01,  5.14432269e-01, -4.79636062e+02],
                 [ 9.97857509e-01, -1.91746573e-02,  6.25517786e-02,  1.76350899e+01],
                 [-4.36894600e-02,  5.16376331e-01,  8.55246582e-01,  1.32014415e+03],
                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
view3 = np.array([[ 1.68406000e-01,  6.81981443e-01,  7.11716749e-01, -7.97013936e+02],
                 [-1.46421518e-02, -7.20218576e-01,  6.93592683e-01, -7.96654052e+02],
                 [ 9.85608962e-01, -1.27226234e-01, -1.11303454e-01,  2.48041229e+03],
                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
view4 = np.array([[-4.30900509e-01, -6.35852426e-01,  6.40325264e-01, -6.90871678e+02],
                 [ 8.18914368e-01,  2.25497975e-02,  5.73472549e-01, -4.46272200e+02],
                 [-3.79083117e-01,  7.71481172e-01,  5.10991968e-01,  1.67877988e+03],
                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
view_cnt = 0
view_inc = 1

def camera_animation():
    global view_cnt, view_inc
    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()
    view = view1 + (view4-view1)*view_cnt/100
    camera_params.extrinsic = view
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    view_cnt += view_inc
    if view_cnt > 100:
        view_inc = -1
    if view_cnt < 0:
        view_inc = 1
    # print(camera_params.extrinsic)

# def get_index_in_depth(idx_ir):
#     idx_depth = np.copy(idx_ir)
#     idx_depth[:,0] = np.around(idx_depth[:,0] / scaleRes[0]).astype(int)
#     idx_depth[:,1] = np.around(idx_depth[:,1] / scaleRes[1]).astype(int)
#     return idx_depth

def get_index_in_depth(xys):
    # turn a list of (x, y) rgb pixel coords into indices for the pointcloud
    xs = [int(round(x / scale_res[0])) for x, y in xys]
    ys = [int(round(y / scale_res[1])) for x, y in xys]
    return (ys, xs)

def get_pattern_in_3d(point2d, pointcloud):
    point3d = []
    for p in point2d:
        point3d.append(pointcloud[p[1],p[0]])
    point3d = np.array(point3d)[:,:3]
    return point3d

def rotate_point(point, angle):
    """Rotate a point around the origin (0, 0, 0)"""
    x, y, z = point
    # Rotate around Y axis
    x_rotated = x * np.cos(angle) - z * np.sin(angle)
    z_rotated = x * np.sin(angle) + z * np.cos(angle)
    return (x_rotated, y, z_rotated)

def project_point(point):
    """Project a 3D point onto a 2D surface"""
    x, y, z = point
    # Simple perspective projection
    fov = 500
    distance = 5
    projected_x = fov * (x / (z + distance)) + WIDTH // 2
    projected_y = fov * (y / (z + distance)) + HEIGHT // 2
    return (int(projected_x), int(projected_y))

def prepare_camera_param():
    cameraIntrinsics = np.array([
        [1537.6326, 0.0, 0.0],
        [0.0, 1537.6326, 0.0],
        [959.2445, 729.5761, 1.0]])
    cameraIntrinsics[0][0] /= scaleRes[0]
    cameraIntrinsics[1][1] /= scaleRes[1]
    cameraIntrinsics[2][0] /= scaleRes[0]
    cameraIntrinsics[2][1] /= scaleRes[1]
    return cameraIntrinsics

def generate_image_id(w,h):
    image_id = np.indices((h,w))
    image_id = np.swapaxes(image_id,0,1)
    image_id = np.swapaxes(image_id,1,2)
    return image_id

def get_pointcloud(image_depth):
    (h,w) = image_depth.shape
    cameraIntrinsics = prepare_camera_param()
    depth = image_depth * 1000
    # depth = jet_to_grayscale(image_depth)
    image_idx = generate_image_id(w,h)
    xrw = (image_idx[:,:,1] - cameraIntrinsics[2][0]) * depth / cameraIntrinsics[0][0]
    yrw = (image_idx[:,:,0] - cameraIntrinsics[2][1]) * depth / cameraIntrinsics[1][1]
    xyzw = np.dstack([xrw, yrw, depth, np.ones(depth.shape)])
    # print(np.mean(depth))
    return xyzw

def visualize_pointcloud(point3d, color=np.zeros((size_pc[1] * size_pc[0], 3))):
    point3d = point3d[::droprate,::droprate,:3].reshape(-1,3)
    point3d = point3d + np.array([0,0,10]) 
    pcd.points = o3d.utility.Vector3dVector(point3d)
    # pcd_pattern.points = o3d.utility.Vector3dVector(pattern)

    pcd.colors = o3d.utility.Vector3dVector(color)
    # pcd_pattern.paint_uniform_color([1, 0, 0])

    # Update the visualizer
    vis.update_geometry(pcd)
    # vis.update_geometry(pcd_pattern)
    vis.poll_events()
    vis.update_renderer()

prev_rot = np.eye(3)
def visualize_lidar(loc,axis):
    global prev_rot
    if loc is not None and axis is not None and ~np.isnan(loc).all() and ~np.isnan(axis).all():
        loc = np.array(loc)
        centroid = lidar.get_center()
        lidar.rotate(prev_rot.T,center=centroid)
        lidar.translate(loc-centroid)
        lidar.rotate(axis.T,center=loc)
        prev_rot = axis.T
        vis.update_geometry(lidar)

objects = []
def reset_objects():
    for obj in objects:
        vis.remove_geometry(obj,reset_bounding_box=False)
    objects.clear()

geometries_line = []
# geometries_point = []
def reset_line():
    while len(geometries_line) > 0:
        line_set = geometries_line.pop(0)
        vis.remove_geometry(line_set,reset_bounding_box=False)
    # while len(geometries_point) > 0:
    #     point = geometries_point.pop(0)
    #     vis.remove_geometry(point,reset_bounding_box=False)

def visualize_line(points=[[0, 0, 0],[1000, 0, 0]],lines=[[0, 1]],color=[1, 0, 0]):
    colors = [color for i in range(len(lines))]
    points = np.array(points)+np.array([0,0,-10])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    # line_set.scale(100, center=line_set.get_center())
    line_set.colors = o3d.utility.Vector3dVector(colors)
    geometries_line.append(line_set)

    vis.add_geometry(line_set,reset_bounding_box=False)

def visualize_coordsys(rmat,tvec,color=[1,0.4,0.3]):
    # camera_animation(vis)
    cube = o3d.geometry.TriangleMesh.create_arrow(cylinder_split=40, cone_split=10)
    # coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # cube = o3d.geometry.TriangleMesh.create_box(width=120, height=220, depth=40)
    cube.compute_vertex_normals()
    cube.paint_uniform_color(color)
    cube.scale(20, center=cube.get_center())
    tvec = np.array(tvec)
    # coord.translate(tvec)
    # coord.rotate(rmat,center=coord.get_center())
    cube.translate(tvec+np.array([-60,-110,-20]))
    cube.rotate(rmat,center=cube.get_center())
    # objects.append(coord)
    objects.append(cube)
    # vis.add_geometry(coord,reset_bounding_box=False)
    vis.add_geometry(cube,reset_bounding_box=False)
    vis.poll_events()
    vis.update_renderer()

drop = 1
cnt = 0
def receive_data_h264(conn,queue):
    global cnt
    decoder = h264decoder.H264Decoder()
    while True:
        t = time.time()
        h264_bytes,_ = conn.recvfrom(65536)
        if len(queue) > 5:
            while len(queue) > 3:
                queue.pop(0)
        framedatas = decoder.decode(h264_bytes)
        for framedata in framedatas:
            (frame, w, h, ls) = framedata
            frame = np.frombuffer(frame, dtype=np.ubyte, count=len(frame))
            frame = frame.reshape((h, ls//3, 3))
            frame = frame[:,:w,:]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if cnt == 0:
                queue.append(frame)

def receive_data(conn,queue):
    global cnt
    while True:
        header, _ = conn.recvfrom(4)
        if header == b'\x01\x23\x45\x67':
            if len(queue) > 5:
                while len(queue) > 3:
                    queue.pop(0)
            framebytes, _ = conn.recvfrom(4)
            try:
                length = np.frombuffer(framebytes, dtype=np.uint32)[0]
            except:
                continue
            if length > 1000000:
                print(length,framebytes)
            else:
                length_ = 0
                outputbytes = b''
                while length_ < length:
                    framebytes = conn.recv(length-length_)
                    length_ += len(framebytes)
                    outputbytes += framebytes
                decrypted_data = zlib.decompress(outputbytes,-15)
                data = np.frombuffer(decrypted_data, dtype=np.float32)
                frame = data.reshape((192,256))
                if cnt == 0:
                    queue.append(frame)
                cnt = (cnt + 1) % drop

def make_connection():
    server_socket = socket.socket()
    server_socket.bind((host, port))
    queue = []
    for i in range(2):
        server_socket.listen(2)
        conn, address = server_socket.accept()
        print("Connection from: " + str(address))
        q = []
        queue.append(q)
        if i == 0:
            t = Thread(target=receive_data_h264, args=(conn,q,))
        else:
            t = Thread(target=receive_data, args=(conn,q,))
        t.start()
    return queue

def reconstruct_depth(frame):
    b = frame[:,:,0].astype(np.uint32)
    g = frame[:,:,1].astype(np.uint32)
    r = frame[:,:,2].astype(np.uint32)
    # r = (r << 16)
    # g = (g << 8)
    # r = (r / np.iinfo(np.uint16).max)
    # g = (g / np.iinfo(np.uint8).max)
    print(np.min(r),np.max(r))
    print(np.min(g),np.max(g))
    print(np.min(b),np.max(b))
    print("-----")
    gray = (r + g + b) / np.iinfo(np.uint16).max
    # print(np.min(gray),np.max(gray))
    frame = np.dstack([gray,gray,gray]).astype(np.uint8)
    return frame

def get_newfile_num(path,subfix='',prefix=''):
    filenames = glob.glob(path+'*')
    num = 0
    while f'{path}{prefix}{num}{subfix}' in filenames:
        num += 1
    return num

def get_color_in_3d(rgb_image):
    color = cv2.resize(rgb_image,(size_pc[0],size_pc[1])).reshape(-1,3)/255
    color[:,[0, 2]] = color[:,[2, 0]]
    return color

def server_program():
    queue = make_connection()
    starttime = time.time()
    rgb_image, depth_image = None, None
    point3d_recent = []
    while True:
        for q, text in zip(queue,['rgb','depth']):
            if len(q) > 0:
                # print(text,len(q))
                frame = q.pop(0)
                if text == 'rgb':
                    rgb_image = np.copy(frame)
                if text == 'depth':
                    # frame = reconstruct_depth(frame)
                    frame = cv2.resize(frame,size_depth)
                    point3d = get_pointcloud(frame)
                    depth_image = np.copy(frame)
                    point3d_recent.append(point3d)
                    while len(point3d_recent) > 10:
                        point3d_recent.pop(0)
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                cv2.imshow(text,frame)
                if text == 'rgb':
                    # print(1/(time.time()-starttime),'fps')
                    starttime = time.time()

        try:
            color = get_color_in_3d(rgb_image)
            visualize_pointcloud(point3d,color=color)
        except:
            pass

        # try:
        #     color = get_color_in_3d(rgb_image)
        #     pattern = np.array([ [105.29819251,  99.07547949, 227.16055298],
        #                          [ 85.13714226, 101.106547  , 238.55732727],
        #                          [122.14604954,  44.14450402, 217.18942261],
        #                          [102.91867677,  43.61574876, 228.43717957],
        #                          [124.74333535,  -5.72801824, 220.78642273],
        #                          [127.19392585, -26.55666794, 222.05654907],
        #                          [108.02035509, -46.95327277, 233.68882751],
        #                          [134.63942166, -66.94607972, 226.31835938],
        #                          [137.05125777, -88.99873428, 224.96209717]])
        #     # pattern = np.array([[ -39.78597518, -156.0855081,   570.43682861],
        #     #                      [ -37.87950603, -166.1813919,   564.14398193],
        #     #                      [ 198.23289401, -104.53925503,  327.48596191],
        #     #                      [ -20.19926896, -200.6089397,   542.56835938]])
        #     reset_line()
        #     visualize_line()
        #     visualize_lidar(np.ones(3)*10,np.eye(3))
        #     visualize_pointcloud(point3d,pattern,color=color)
        #     # visualize_pointcloud(point3d,np.zeros((64*9,3)),color=color)
        # except Exception as e:
        #     print(e)

        mkey = cv2.waitKey(1)
        if mkey == ord('q'):
            print('[User command] Exit')
            break
        # elif mkey == ord(' '):
        #     num = get_newfile_num('../data/captured/rgb/','jpg')
        #     cv2.imwrite(f'../data/captured/rgb/{str(num).zfill(5)}.jpg',rgb_image)
        #     np.save(f'../data/captured/depth/{str(num).zfill(5)}.npy', point3d)
        #     # cv2.imwrite('../data/captured/depth/{}.jpg'.format(str(num).zfill(5)),depth_image)
        #     print("Saved",num)
        elif mkey == ord(' '):
            num = get_newfile_num('../data/figures/','.jpg')
            cv2.imwrite(f'../data/figures/{num}.jpg',rgb_image)
            np.save(f'../data/figures/{num}.npy', point3d_recent)
        elif mkey == 'a':
            num = get_newfile_num('../data/captured/rgb/','.jpg')
            np.savez('../data/captured/{}.npz'.format(str(num).zfill(5)),
                     rgb=rgb_image,
                     depth=depth_image,
                     pointcloud=point3d)
    print("saved")

    for conn_ in conn:
        conn_.close()  # close the connection


if __name__ == '__main__':
    server_program()
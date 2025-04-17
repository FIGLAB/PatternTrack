from streaming_iphone import *
from streaming import *
import cv2
import numpy as np
import time
import coremltools as ct
from patterntrack import *
from output_server import WebSocketServer

raspi_ip = "192.168.1.105"
calibration_path = "../data/iphone1/"

def initialize_camera_and_server(calibration_path, raspi_ip):
    cam = CameraStream(calibration_path, host_ip=raspi_ip)
    ws_main = WebSocketServer(port=8080, ismain=1)
    return cam, ws_main

def initialize_video_writer():
    return cv2.VideoWriter(f'../data/figures/livedemo1.mp4', cv2.VideoWriter_fourcc(*'avc1'), 30.0, (9600, 1920))

### This order is important
cam, ws_main = initialize_camera_and_server(calibration_path, raspi_ip)
(queue_rgb,queue_d) = make_connection()
### This order is important
green = [54,216,97]
out = initialize_video_writer()
isRecord = False

def resize_images(image_rgb, image_ir, image_depth, img_th, pattern_img):
    image_rgb = cv2.resize(image_rgb, (int(image_rgb.shape[1] / 2.5), int(image_rgb.shape[0] / 2.5)))
    image_ir = cv2.resize(image_ir, (image_rgb.shape[1], image_rgb.shape[0]))
    image_depth = cv2.resize(image_depth, (image_rgb.shape[1], image_rgb.shape[0]))
    img_th = cv2.resize(img_th, (image_rgb.shape[1], image_rgb.shape[0]))

    pattern_img = cv2.rotate(pattern_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ratio = pattern_img.shape[1] / pattern_img.shape[0]
    pattern_img = cv2.resize(pattern_img, (image_rgb.shape[1], int(image_rgb.shape[1] / ratio)))
    ratio = image_depth.shape[1] / image_depth.shape[0]
    image_depth = cv2.resize(image_depth, (image_rgb.shape[1], int(image_rgb.shape[1] / ratio)))
    return image_rgb, image_ir, image_depth, img_th, pattern_img

def show_combined_frame(vis, image_rgb, image_ir, image_depth, img_th, pattern_img):
    image_rgb, image_ir, image_depth, img_th, pattern_img = resize_images(image_rgb, image_ir, image_depth, img_th, pattern_img)
    frame = np.vstack((image_rgb, image_ir, image_depth, img_th, pattern_img))
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('Combined', frame)

image_rgb_old = [np.zeros((1440, 1920, 3)).astype(np.uint8)]
depth_old = [np.zeros((760, 960)).astype(np.float32)]
def process_frame(cam, queue_rgb, queue_d, vis, ws_main):
    ret, image_ir = cam.read()
    pattern_img = get_pattern_image()
    try:
        image_rgb, depth = queue_rgb.pop(0), queue_d.pop(0)
        depth = cv2.resize(depth, size_depth)
        image_rgb_old[0] = np.copy(image_rgb)
        depth_old[0] = np.copy(depth)
    except Exception as e:
        image_rgb = image_rgb_old[0]
        depth = depth_old[0]

    image_depth = np.clip(np.dstack([depth * 100, depth * 100, depth * 100]), 0, 255).astype(np.uint8)
    img_th = get_threshold_image(image_ir)
    observed, _ = detect_pattern(img_th)

    pointcloud = get_pointcloud(depth)
    color = get_color_in_3d(image_rgb)
    visualize_pointcloud(pointcloud, color=color)

    img_th = cv2.cvtColor(img_th, cv2.COLOR_GRAY2RGB)
    if len(observed) < 12:
        print("Not enough points")
        show_combined_frame(vis, image_rgb, image_ir, image_depth, img_th, pattern_img)
        return None, None, image_rgb, image_ir, image_depth, img_th, pattern_img

    try:
        center = np.array(center)
        pattern = get_index_in_depth(center)
        pattern = get_pattern_in_3d(pattern, pointcloud)
    except:
        pattern = np.zeros((64 * 9, 3))

    rectangles = search_rectangles(observed)
    observed = np.unique(np.array(rectangles).reshape(-1, 2), axis=0)
    image_rgb = put_pattern_on_image(image_rgb, observed, rectangles, size=10, color=(255, 255, 255), color2=(255, 255, 255))
    obj_observed = get_pattern_in_3d(get_index_in_depth(observed), pointcloud)

    rmats, tvecs, scores, reprojs, ris, recs, match_indice = greedy_solution_search(rectangles, obj_observed, pointcloud)

    if len(rmats) == 0:
        print("No solution found")
        show_combined_frame(vis, image_rgb, image_ir, image_depth, img_th, pattern_img)
        return None, None, image_rgb, image_ir, image_depth, img_th, pattern_img

    idx = 0
    best_score = scores[idx]
    rmat = rmats[idx]
    tvec = tvecs[idx]
    reproj = reprojs[idx]
    match_idx = match_indice[idx]
    rec = recs[idx]
    ri = ris[idx]

    best_score = round(best_score, 1)
    pattern_img = cv2.putText(pattern_img, f'Score: {best_score}', (60, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
    pattern_img = put_pattern_on_image(pattern_img, [], [pattern_model[diamond[ri]]], size=10, color=green, color2=(0, 0, 255))
    pattern_img = put_pattern_on_image(pattern_img, reproj.astype(int), [], color2=(0, 255, 0))

    rot = np.linalg.inv(rmat)
    pos = np.matmul(rot, -tvec).T[0]

    return pos, rot, image_rgb, image_ir, image_depth, img_th, pattern_img

def main():
    out = initialize_video_writer()
    outputs = []
    outputs_rot = []
    prev = [None, None]
    pos, rot = None, None

    while True:
        starttime = time.time()
        pos, rot, image_rgb, image_ir, image_depth, img_th, pattern_img = process_frame(cam, queue_rgb, queue_d, vis, ws_main)
        if pos is None or rot is None:
            continue

        prev_pos = prev[0]
        prev[0] = pos
        prev[1] = rot
        if prev_pos is not None and np.linalg.norm(prev_pos - pos) > 200:
            pos = prev[0]
            rot = prev[1]

        outputs.append(pos)
        outputs_rot.append(rot)
        if len(outputs) > 30:
            outputs.pop(0)
            outputs_rot.pop(0)
        pos = np.mean(outputs, axis=0)
        rot = np.mean(outputs_rot, axis=0)


        if pos is not None and rot is not None:        
            data_main = ws_main.generate_data(pos, rot)
            ws_main.send_data(data_main)

            matrice1 = ws_main.receive_data()

        reset_objects()
        visualize_coordsys(rot, pos)
        show_combined_frame(vis, image_rgb, image_ir, image_depth, img_th, pattern_img)
        endtime = time.time()

    pool.close()
    pool.join()
    cv2.destroyAllWindows()
    vis.destroy_window()

if __name__ == "__main__":
    main()

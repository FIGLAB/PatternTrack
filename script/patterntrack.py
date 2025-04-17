import numpy as np
from pattern_detection import *
from lambdatwist import *
import multiprocessing
from sklearn.neighbors import KDTree
from sparse_diamonds import *

size_rgb = (1920,1440)
size_depth = (960,760)
scaleRes = np.array([size_rgb[0]/size_depth[0], size_rgb[1]/size_depth[1]])

def get_pattern_in_3d(point2d, pointcloud):
    point3d = []
    for p in point2d:
        point3d.append(pointcloud[p[1],p[0]])
    
    point3d = np.array(point3d)
    if len(point3d) > 0:
        point3d = point3d[:, :3]
    return point3d

def get_index_in_depth(idx_ir):
    idx_depth = np.copy(idx_ir)
    idx_depth[:,0] = np.around(idx_depth[:,0] / scaleRes[0]).astype(int)
    idx_depth[:,1] = np.around(idx_depth[:,1] / scaleRes[1]).astype(int)
    return idx_depth

def rotation_matrix_z(objPoints,theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    rot_matrix = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    objPoints = np.dot(objPoints, rot_matrix.T)
    return objPoints

multiprocessing.set_start_method('fork')

#### Load Model Pattern ####
def to_image_coords(xyzs):
    # divide out the z coordinate to obtain image plane coordinates
    return xyzs[..., :2] / xyzs[..., 2:3]

def get_depth_inds(xys):
    # turn a list of (x, y) rgb pixel coords into indices for the pointcloud
    xs = [int(round(x / scaleRes[0])) for x, y in xys]
    ys = [int(round(y / scaleRes[1])) for x, y in xys]
    return (ys, xs)

def project_points(pts, rmat, tvec):
    """ pts: Nx3, rmat: 3x3, tvec: 3 or 1x3 or 3x1 """
    return to_image_coords((rmat @ pts.T).T + tvec.flatten())

def get_pattern_image():
    pattern_img = np.zeros((1000, 1000, 3), dtype="uint8")
    # pattern_img = put_pattern_on_image(pattern_img,pattern_model,[],size=1,color2=(100,100,100))
    pattern_img = put_pattern_on_image(pattern_img,pattern_model[diamond.flatten()],[],size=5,color2=(255,255,255))
    return pattern_img

pat_pts = to_image_coords(np.array(sphere_pts))
pattern_model = np.array(pattern_model)
diamond = np.array(diamond3)
pattern_single = pattern_model[np.unique(diamond.flatten())]
tree = KDTree(pattern_single,leaf_size=1)

#### Search and Match ####
def get_score(pat_observed):
    dis, idx = tree.query(pat_observed, k=1)
    dis = dis[:,0]
    idx = idx[:,0]
    unique_idx_count = len(np.unique(idx))
    # if len(pat_observed) - unique_idx_count > 5:
    if len(pat_observed) - unique_idx_count > unique_idx_count/5:
        score = 9999
    else:
        # score = np.mean(dis, axis=0)
        score = np.quantile(dis, 0.75)
    # score = np.mean(dis, axis=0)
    return score, idx

def search_parallel(arg):
    pat_rect, obj_rect, obj_observed, ri, rec, r_id = arg
    best_score = 9999
    best_rmat = None
    best_tvec = None
    reproj = None
    best_match_idx = None

    res, rmats_, tvecs_ = p4p(objPoints=obj_rect, imagePoints=pat_rect)

    for i in range(res):
        rmat = rmats_[i]
        tvec = tvecs_[i]
        if np.linalg.det(rmat)==0:
            continue
        pat_observed = project_points(obj_observed, rmat, tvec)
        pat_observed = (pat_observed * 600 + 500).astype(int)
        score, match_idx = get_score(pat_observed)
        if best_score > score:
            best_score = score
            best_rmat = rmat
            best_tvec = tvec
            reproj = pat_observed
            best_match_idx = match_idx
            # if best_score < 30:
            #     break
    return best_rmat, best_tvec, best_score, reproj, ri, rec, best_match_idx, r_id

pool = multiprocessing.Pool()
def greedy_solution_search(rectangles, obj_observed, pointcloud):
    global pool
    args_list = []
    for r_id, rectangle in enumerate(rectangles):
        # for _ in range(4):
        rectangle = np.roll(rectangle,1,axis=0)
        obj_rect = get_pattern_in_3d(get_index_in_depth(rectangle),pointcloud)
        for ri in range(81):
            pattern_rect = np.copy(diamond[ri])
            pat_rect = np.copy(pat_pts[pattern_rect])
            args_list.append((pat_rect,obj_rect,np.copy(obj_observed),ri,np.copy(rectangle),r_id))
    results = pool.map(search_parallel, args_list)
    rmats, tvecs, scores, reprojs, ris, recs, match_indice, r_ids = [], [], [], [], [], [], [], []
    for i in range(len(results)):
        rmat = results[i][0]
        tvec = results[i][1]
        score = results[i][2]
        reproj = results[i][3]
        ri = results[i][4]
        rec = results[i][5]
        match_idx = results[i][6]
        r_id = results[i][7]
        if rmat is None or tvec is None or reproj is None or score==9999:
            continue
        rmats.append(rmat)
        tvecs.append(tvec)
        scores.append(score)
        reprojs.append(reproj)
        ris.append(ri)
        recs.append(rec)
        match_indice.append(match_idx)
        r_ids.append(r_id)
    if len(rmats) == 0:
        return [], [], [], [], [], [], []
    rmats = np.array(rmats)
    tvecs = np.array(tvecs)
    scores = np.array(scores)
    reprojs = np.array(reprojs)
    ris = np.array(ris)
    recs = np.array(recs)
    match_indice = np.array(match_indice)
    r_ids = np.array(r_ids)

    sel = np.argsort(scores)
    scores = scores[sel]
    rmats = rmats[sel]
    tvecs = tvecs[sel]
    ris = ris[sel]
    reprojs = reprojs[sel]
    recs = recs[sel]
    match_indice = match_indice[sel]

    return rmats, tvecs, scores, reprojs, ris, recs, match_indice

def search_rectangles(observed):
    rectangles = []
    for i in range(len(observed)):
        rectangles_, _ = find_rectangle(observed, i)
        rectangles.append(rectangles_)
    if len(rectangles) == 0:
        return []
    rectangles = np.concatenate(rectangles)
    if len(rectangles) == 0:
        return []
    centroids = calculate_centroids(rectangles)
    angles = calculate_angles(rectangles, centroids)
    idx = np.argsort(angles,axis=1)
    first_dim_indices = np.arange(rectangles.shape[0])[:, None]  # Shape (35, 1) to broadcast along the second dimension

    # Use advanced indexing to reorder `a`
    rectangles = rectangles[first_dim_indices, idx, :]
    valid = check_valid_parallelograms_corrected(rectangles)
    rectangles = rectangles[valid]
    rectangles = np.unique(rectangles,axis=0)

    if len(rectangles) == 0:
        return []

    area = get_area(rectangles)
    threshold = np.quantile(np.array(area),0.5)
    threshold2 = np.quantile(np.array(area),0.4)
    std = np.std(np.array(area))
    rectangles = np.array([rectangle for rectangle, a in zip(rectangles, area) if (a < threshold) & (a > threshold2)])
    # rectangles = np.array([rectangle for rectangle, a in zip(rectangles, area) if (a < threshold2)])
    return rectangles

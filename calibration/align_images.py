from __future__ import print_function
import cv2
import numpy as np
import argparse
from math import sqrt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, help='calibration image path')
# parser.add_argument('--reduce', action=argparse.BooleanOptionalAction)
args = parser.parse_args()


def get_undistorted(image):
    (height, width, _) = image.shape
    newcameramatrix, _ = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (width, height), 1, (width, height)
    )
    undistorted_image = cv2.undistort(
        image, mtx, dist, None, mtx
    )
    return undistorted_image


def put_pattern_on_image(image,pattern):
    for i, c in enumerate(pattern):
        x = int(c[0][0])
        y = int(c[0][1])
        image = cv2.circle(image, (x,y), 2, (255, 0, 0), 2)
        image = cv2.putText(image, str(i), (x+2,y+2), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, (255, 0, 0), 1, cv2.LINE_AA)
    return image

def capture_click(event, x_click, y_click, flags, params):
    global X_CAPT, Y_CAPT
    if event == cv2.EVENT_LBUTTONDOWN:
        xy_click = np.float32([x_click, y_click])
        xy_click = xy_click.reshape(-1, 1, 2)
        #print(xy_click)
        refined_xy = cv2.cornerSubPix(I, xy_click, dW, (-1, -1), criteria)
        #print(refined_xy)
        X_CAPT = np.append(X_CAPT, refined_xy[0, 0, 0])
        Y_CAPT = np.append(Y_CAPT, refined_xy[0, 0, 1])
        cv2.drawMarker(color_img, (int(X_CAPT[-1]), int(Y_CAPT[-1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30)

###################################
##  Align WFoV and NFoV Images   ##
##    (Get homography matrix)    ##
###################################

## [load]
# intrinsic_params = np.load("{}/intrinsic_calib.npz".format(args.path))
# mtx, dist = intrinsic_params['mtx'], intrinsic_params['dist']

iphone_img_file = "./{}/rgb/00002.jpg".format(args.path)
ir_img_file = "./{}/ir/00001.jpg".format(args.path)
img1 = cv2.imread(ir_img_file) # source
img2 = cv2.imread(iphone_img_file) # destination

# img1 = get_undistorted(img1)

img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

cv2.imshow('IR',img1_gray)
cv2.imshow('iPhone',img2_gray)
cv2.waitKey(0)

if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)
## [load]

# ## [AKAZE]
# akaze = cv2.AKAZE_create()
# kpts1, desc1 = akaze.detectAndCompute(img1, None)
# kpts2, desc2 = akaze.detectAndCompute(img2, None)
# ## [AKAZE]

# ## [2-nn matching]
# matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
# nn_matches = matcher.knnMatch(desc1, desc2, 2)
# print(nn_matches)
# ## [2-nn matching]

# ## [ratio test filtering]
# matched1 = []
# matched2 = []
# nn_match_ratio = 0.8 # Nearest neighbor matching ratio
# for m, n in nn_matches:
#     if m.distance < nn_match_ratio * n.distance:
#         matched1.append(kpts1[m.queryIdx])
#         matched2.append(kpts2[m.trainIdx])
# ## [ratio test filtering]

CHECKERBOARD = (6,8)
ret, matched1 = cv2.findChessboardCorners(img1_gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
ret, matched2 = cv2.findChessboardCorners(img2_gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

cv2.imshow('IR',put_pattern_on_image(img1,matched1))
cv2.imshow('iPhone',put_pattern_on_image(img2,matched2))
cv2.waitKey(0)

## [calculate homography matrix]
# src_pts = np.float32([ m.pt for m in matched1 ]).reshape(-1,1,2)
# dst_pts = np.float32([ m.pt for m in matched2 ]).reshape(-1,1,2)
src_pts = np.float32([ m for m in matched1 ]).reshape(-1,1,2)
dst_pts = np.float32([ m for m in matched2 ]).reshape(-1,1,2)
homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
print(homography)
np.savez(
    '{}/homography.npz'.format(args.path),
    mtx=homography,
    size=(img2.shape[1],img2.shape[0]))
print("******************************")
# print(dst_pts[:,:,0].astype(np.int)) # x position
## [calculate homography matrix]

#################################
##        Check mapping        ##
##  (Check homography matrix)  ##
#################################

# ## [load]
homography = np.load('{}/homography.npz'.format(args.path))
hmtx = homography['mtx']
size = homography['size']
# ## [load]

## [save image]
img_transed = cv2.warpPerspective(img1, hmtx, size)
cv2.imshow('result', img_transed)
res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv2.waitKey(0)
# cv2.waitKey()
## [save image]

## [homography check]
# inliers1 = []
# inliers2 = []
# good_matches = []
# inlier_threshold = 2.5 # Distance threshold to identify inliers with homography check
# for i, m in enumerate(matched1):
#     col = np.ones((3,1), dtype=np.float64)
#     col[0:2,0] = m.pt

#     col = np.dot(homography, col)
#     col /= col[2,0]
#     dist = sqrt(pow(col[0,0] - matched2[i].pt[0], 2) +\
#                 pow(col[1,0] - matched2[i].pt[1], 2))

#     if dist < inlier_threshold:
#         good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
#         inliers1.append(matched1[i])
#         inliers2.append(matched2[i])
# ## [homography check]

# ## [draw final matches]
# res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
# cv2.drawMatches(img1, inliers1, img2, inliers2, good_matches, res)

# inlier_ratio = len(inliers1) / float(len(matched1))
# print('A-KAZE Matching Results')
# print('*******************************')
# print('# Keypoints 1:                        \t', len(kpts1))
# print('# Keypoints 2:                        \t', len(kpts2))
# print('# Matches:                            \t', len(matched1))
# print('# Inliers:                            \t', len(inliers1))
# print('# Inliers Ratio:                      \t', inlier_ratio)

# cv2.imshow('result', res)
# cv2.imwrite('./output/camera_align_result.jpg',res)
# cv2.waitKey()
## [draw final matches]
import numpy as np
import cv2
from itertools import combinations

def calculate_centroids(rectangles):
    # rectangles.shape should be (n_rectangles, n_points_per_rectangle, 2)
    centroids = np.mean(rectangles, axis=1)
    return centroids

def calculate_angles(rectangles, centroids):
    # Expanding dimensions for broadcasting
    expanded_centroids = centroids[:, np.newaxis, :]
    
    # Calculating differences for angle calculation
    diff = rectangles - expanded_centroids
    
    # Calculating angles using arctan2, which handles broadcasting
    angles = np.arctan2(diff[..., 1], diff[..., 0])
    return angles

def check_valid_parallelograms_corrected(points):
    # Calculate vectors
    v1 = points[:, 0, :] - points[:, 1, :]
    v2 = points[:, 1, :] - points[:, 2, :]
    v3 = points[:, 2, :] - points[:, 3, :]
    v4 = points[:, 3, :] - points[:, 0, :]
    
    # Calculate lengths of vectors
    lengths1 = np.linalg.norm(v1, axis=1)
    lengths2 = np.linalg.norm(v2, axis=1)
    lengths3 = np.linalg.norm(v3, axis=1)
    lengths4 = np.linalg.norm(v4, axis=1)
    
    # Avoid division by zero for very short lengths
    valid_lengths = (lengths1 > 0.1) & (lengths2 > 0.1) & (lengths3 > 0.1) & (lengths4 > 0.1)
    
    # Normalize vectors where lengths are valid
    v1_norm = np.divide(v1, lengths1[:, np.newaxis], where=valid_lengths[:, np.newaxis])
    v2_norm = np.divide(v2, lengths2[:, np.newaxis], where=valid_lengths[:, np.newaxis])
    v3_norm = np.divide(v3, lengths3[:, np.newaxis], where=valid_lengths[:, np.newaxis])
    v4_norm = np.divide(v4, lengths4[:, np.newaxis], where=valid_lengths[:, np.newaxis])
    
    # Calculate dot products
    d13 = np.einsum('ij,ij->i', v1_norm, v3_norm)
    d24 = np.einsum('ij,ij->i', v2_norm, v4_norm)
    d12 = np.einsum('ij,ij->i', v1_norm, v2_norm)
    d23 = np.einsum('ij,ij->i', v2_norm, v3_norm)
    d34 = np.einsum('ij,ij->i', v3_norm, v4_norm)
    d41 = np.einsum('ij,ij->i', v4_norm, v1_norm)

    a12 = np.arccos(d12)*180/np.pi
    a23 = np.arccos(d23)*180/np.pi
    a34 = np.arccos(d34)*180/np.pi
    a41 = np.arccos(d41)*180/np.pi

    
    # Ensure dot products are within valid range
    valid_dots = (d13 <= 0.96) & (d24 <= 0.96) & (np.abs(d12) < 0.96) & (np.abs(d34) < 0.96)
    # valid_dots = (np.abs(a12 - a34) < 20) & (np.abs(a23 - a41) < 20)  
    # print(sum(valid_lengths & valid_dots))
    
    # Compute angles for further validation if needed
    # Angles are computed only if both valid_lengths and valid_dots conditions are met
    # This part is commented out as the current checks already suffice for parallelogram validation
    
    # Return whether each set of points forms a valid parallelogram
    return valid_lengths & valid_dots

def check_valid_parallelogram(points):
    v1 = points[0] - points[1]
    v2 = points[1] - points[2]
    v3 = points[2] - points[3]
    v4 = points[3] - points[0]

    length1 = np.linalg.norm(v1)
    length2 = np.linalg.norm(v2)
    length3 = np.linalg.norm(v3)
    length4 = np.linalg.norm(v4)

    if length1 < 0.1 or length2 < 0.1 or length3 < 0.1 or length4 < 0.1:
        return False

    v1 = v1 / length1
    v2 = v2 / length2
    v3 = v3 / length3
    v4 = v4 / length4

    d13 = np.dot(v1,v3)
    d24 = np.dot(v2,v4)
    d12 = np.dot(v1,v2)
    d34 = np.dot(v3,v4)

    if d13 >= 1 or d24 >= 1 or d12 >= 1 or d34 >= 1:
        return False
    if d13 <= -1 or d24 <= -1 or d12 <= -1 or d34 <= -1:
        return False

    # angle between v1 and v3
    angle1 = np.arccos(d13)*180 / np.pi
    angle2 = np.arccos(d24)*180 / np.pi
    # angle between v1 and v2
    angle3 = np.arccos(d12)*180 / np.pi
    angle4 = np.arccos(d34)*180 / np.pi

    valid = (angle1 > 172) and (angle2 > 172) and (angle3 < 172) and (angle4 < 172) and (angle3 > 8) and (angle4 > 8)
    # valid = (angle1 > 172) and (angle2 > 172) and (angle3 < 130) and (angle4 < 130) and (angle3 > 50) and (angle4 > 50)
    return valid

def get_area(polygons):
    area = []
    for polygon in polygons:
        # calculate the area of the parallelogram p
        v1 = polygon[1] - polygon[0]
        v2 = polygon[2] - polygon[0]
        area.append(np.abs(np.cross(v1,v2)))
    return area

def find_rectangle(points, i):
    distance = np.linalg.norm(points - points[i],axis=1)
    idx = np.argsort(distance)[:5]
    points_candidate = points[idx]
    rectangle_candidate = np.array(list(combinations(points_candidate, 4)))
    return rectangle_candidate, points[idx]

def get_threshold_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_gray = cv2.filter2D(img_gray, -1, kernel)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(40, 40))
    reconst_img = clahe.apply(img_gray)
    reconst_img = cv2.bitwise_not(reconst_img)
    reconst_img = (reconst_img * 255).astype(np.uint8)
    reconst_img[reconst_img < 120] = 0
    reconst_img[reconst_img >= 120] = 255
    reconst_img = cv2.cvtColor(reconst_img, cv2.COLOR_GRAY2RGB)
    reconst_img = cv2.resize(reconst_img, (img.shape[1], img.shape[0]))
    return reconst_img[:,:,0]

def threshold_pattern(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(image,cv2.CV_8U,ksize=3,scale=2)
    image_threshold = laplacian
    # grad_x = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
    # grad_y = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)
    # grad = np.sqrt(grad_x**2 + grad_y**2)
    # grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
    # image_threshold = grad_norm
    return image_threshold

def detect_pattern(image_threshold):
    # image_threshold = threshold_pattern_v1(image)
    contours, hierarchy = cv2.findContours(image=image_threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    contour_list = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,False),True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 4) & (area > 60)  & (area < 4000) ):
        # if ((len(approx) > 4) & (area > 100)  & (area < 2000) ):
            contour_list.append(contour)
    contours = contour_list

    center = []
    for c in contours:
        c = np.array(c)[:,0]
        c = np.mean(c,axis=0)
        c = np.around(c).astype(int)
        center.append(c)
    center = np.array(center)
    return center, contours

def put_pattern_on_image(image,pattern,contours=None, color=(255, 0, 0), size=5, color2 = (0, 0, 255)):
    image = np.copy(image)
    if contours is not None:
        cv2.drawContours(image=image, contours=contours, contourIdx=-1,
                    color=color, thickness=4, lineType=cv2.LINE_AA)
    for i, pt in enumerate(pattern):
        x, y = pt[0], pt[1]
  
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(image, (x, y), 3, color2, size)
        # Visualize ordered number on image
        # image = cv2.putText(image, str(i), (x+2,y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # for i, pt in enumerate(patterns):
    #     x, y = pt[0], pt[1]
    #     cv2.circle(image, (x, y), 1, (255, 0, 0), 5)
    return image

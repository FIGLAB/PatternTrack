import ctypes
import numpy as np
import time

_dll = ctypes.CDLL("_lambdatwist.so")
_p3p = _dll.p3p
# not POINTER(c_double) for efficiency reasons, since __array_interface__["data"][0] is an integer
_p3p.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
_p3p.restype = ctypes.c_int

_p4p = _dll.p4p
# not POINTER(c_double) for efficiency reasons, since __array_interface__["data"][0] is an integer
_p4p.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
_p4p.restype = ctypes.c_int

def p3p(objPoints, imagePoints):
    """ p3p(3x3 object points, 3x2 image points) -> (R, Rx3x3 rotation matrices, Rx3x1 translation vectors) """
    objPoints = np.ascontiguousarray(objPoints, dtype="float64")
    assert objPoints.size == 9
    imagePoints = np.ascontiguousarray(imagePoints, dtype="float64")
    assert imagePoints.size == 6

    rmats = np.empty((4, 3, 3), dtype="float64", order="C")
    tvecs = np.empty((4, 3, 1), dtype="float64", order="C")

    res = _p3p(
        objPoints.__array_interface__["data"][0],
        imagePoints.__array_interface__["data"][0],
        rmats.__array_interface__["data"][0],
        tvecs.__array_interface__["data"][0]
    )

    return res, rmats[:res], tvecs[:res]

def p4p(objPoints, imagePoints):
    """ p3p(3x3 object points, 3x2 image points) -> (R, Rx3x3 rotation matrices, Rx3x1 translation vectors) """
    objPoints = np.ascontiguousarray(objPoints, dtype="float64")
    assert objPoints.size == 12
    imagePoints = np.ascontiguousarray(imagePoints, dtype="float64")
    assert imagePoints.size == 8

    rmats = np.empty((4, 3, 3), dtype="float64", order="C")
    tvecs = np.empty((4, 3, 1), dtype="float64", order="C")

    res = _p4p(
        objPoints.__array_interface__["data"][0],
        imagePoints.__array_interface__["data"][0],
        rmats.__array_interface__["data"][0],
        tvecs.__array_interface__["data"][0]
    )

    return res, rmats[:res], tvecs[:res]

def rotation_matrix_z(objPoints,theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    rot_matrix = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    objPoints = np.dot(objPoints, rot_matrix.T)
    return objPoints


def rotation_matrix_y(objPoints,theta):
    c, s = np.cos(np.radians(theta)), np.sin(np.radians(theta))
    rot_matrix = np.array([
        [c, 0, -s],
        [0, 1, 0],
        [s, 0, c]
    ])
    objPoints = np.dot(objPoints, rot_matrix.T)
    return objPoints
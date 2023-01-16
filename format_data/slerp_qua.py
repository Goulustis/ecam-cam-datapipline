"""
contains functions interpolating extrinsics

SLERP taken from: https://github.com/apple/ml-neuman/blob/eaa4665908ba1e39de5f40ef1084503d1b6ab1cb/geometry/transformations.py
"""
import numpy as np
import math
from colmap_find_scale.read_write_model import qvec2rotmat, rotmat2qvec
from collections import namedtuple

_EPS = np.finfo(float).eps * 4.0
Rmtx = namedtuple("Rmtx", ["flat"])


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
    >>> v0 = np.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> np.allclose(v1, v0 / np.linalg.norm(v0))
    True
    >>> v0 = np.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=2)), 2)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / np.expand_dims(np.sqrt(np.sum(v0*v0, axis=1)), 1)
    >>> np.allclose(v1, v2)
    True
    >>> v1 = np.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> np.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data
    return None

def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.
    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quaternion_slerp(q0, q1, 0)
    >>> np.allclose(q, q0)
    True
    >>> q = quaternion_slerp(q0, q1, 1, 1)
    >>> np.allclose(q, q1)
    True
    >>> q = quaternion_slerp(q0, q1, 0.5)
    >>> angle = math.acos(np.dot(q0, q))
    >>> np.allclose(2, math.acos(np.dot(q0, q1)) / angle) or \
        np.allclose(2, math.acos(-np.dot(q0, q1)) / angle)
    True
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    if fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def ext_to_qua(ext_mtx, ret_t = True):
    """
    input:
        ext_mtx(np.array): 4x4 extrinsic matrix
    output:
        qua (np.array): quanterion
    """
    R = Rmtx(ext_mtx[:3,:3].reshape(-1))
    if ret_t:
        return rotmat2qvec(R), ext_mtx[:,3]
    else:
        return rotmat2qvec(R)


def qua_to_ext(qua, t):
    """
    input:
        qua (np.array) : quanteriion
        t (np.array): camera translation

    return:

        ext: extrinsic matrix (4x4)
    """
    if len(t.shape) == 1:
        t = t[...,None]
        
    R = qvec2rotmat(qua)
    bot = np.zeros((1,3))
    R = np.concatenate([R, bot], axis = 0)
    return np.concatenate([R, t], axis = 1)


def create_interpolated_ecams(eimg_ts, triggers, ecams_trig):
    """
    input:
        eimg_ts (np.array): starting times at which the image is accumulated
        triggers (np.array): col img starting time
        ecam_trigs (np.array): extrinsics of event camera at trigger times

    returns:
        ecams_int (np.array): interpolated extrinsic positions
    """

    ecams_int = []
    trig_idx = 1

    trig_st = triggers[trig_idx - 1]
    trig_end = triggers[trig_idx]
    trig_diff = trig_end - trig_st

    qua_st, tr_st = ext_to_qua(ecams_trig[trig_idx - 1])
    qua_end, tr_end = ext_to_qua(ecams_trig[trig_idx])

    for eimg_t in eimg_ts:
        if eimg_t > trig_end:
            trig_idx += 1
            if trig_idx >= len(triggers):
                break

            qua_st, tr_st = ext_to_qua(ecams_trig[trig_idx - 1])
            qua_end, tr_end = ext_to_qua(ecams_trig[trig_idx])

            trig_st = triggers[trig_idx - 1]
            trig_end = triggers[trig_idx]
            trig_diff = trig_end - trig_st
        
        frac = (eimg_t - trig_st)/trig_diff
        interp_qua = quaternion_slerp(qua_st,qua_end,frac)
        ecams_int.append(qua_to_ext(interp_qua, (1-frac)*tr_st + frac*tr_end))
    
    return np.stack(ecams_int)
import numpy as np
from .types import Vec3D


def normalize(input: Vec3D):
    return input / np.linalg.norm(input)


def mix(x, y, a):
    return x * (1.0 - a) + y * a


def is_incident(v1, v2, epsilon=0.999):
    return np.abs(np.dot(v1, v2)) > epsilon

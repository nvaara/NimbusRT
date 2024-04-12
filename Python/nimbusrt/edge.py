import numpy as np
from .types import Vec3D
from .utils import normalize, mix
from .material import Material


class EdgeFace:
    def __init__(self, normal, tangent):
        self._normal = normal
        self._tangent = tangent
        self._material = None

    def to(self, device):
        self._normal.to(device)
        self._tangent.to(device)

    @property
    def normal(self):
        return self._normal

    @property
    def tangent(self):
        return self._tangent

    @property
    def material(self):
        return self._material

    @material.setter
    def material(self, material):
        self._material = material


class Edge:
    def __init__(
        self,
        start: Vec3D,
        end: Vec3D,
        normal0: Vec3D,
        normal1: Vec3D,
        material_index0: int,
        material_index1: int,
    ):
        self._start = np.array(start, dtype=np.float32)
        self._end = np.array(end, dtype=np.float32)
        self._forward = normalize(self._end - self._start)
        self._material_index0 = material_index0
        self._material_index1 = material_index1
        self._edge_face0 = None
        self._edge_face1 = None
        self._n = None
        self._compute_vectors(np.array(normal0), np.array(normal1))

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def forward(self):
        return self._forward

    @property
    def material_index0(self):
        return self._material_index0

    @property
    def material_index1(self):
        return self._material_index1

    @property
    def edge_face0(self):
        return self._edge_face0

    @property
    def edge_face1(self):
        return self._edge_face1

    @property
    def n(self):
        return self._n

    def link_materials(self, materials: np.ndarray[Material]):
        self.edge_face0.material = materials[self.material_index0]
        self.edge_face1.material = materials[self.material_index1]
        return self

    def _compute_vectors(self, normal0, normal1):
        mix_normal = mix(normal0, normal1, 0.5)
        tangent0 = normalize(np.cross(normal0, self._forward))
        tangent1 = normalize(np.cross(normal1, self._forward))
        tangent0 = tangent0 if np.dot(tangent0, mix_normal) < 0.0 else -tangent0
        tangent1 = tangent1 if np.dot(tangent1, mix_normal) < 0.0 else -tangent1

        self._edge_face0 = EdgeFace(normal0, tangent0)
        self._edge_face1 = EdgeFace(normal1, tangent1)

        self._n = 2.0 - np.abs(np.arccos(np.dot(tangent0, tangent1))) / np.pi


class EdgeHelper:  # For writing to file
    def __init__(
        self,
        start,
        end,
        normal0,
        normal1,
        material_index0,
        material_index1,
    ):
        self.start = start.tolist()
        self.end = end.tolist()
        self.normal0 = normal0.tolist()
        self.normal1 = normal1.tolist()
        self.material_index0 = material_index0
        self.material_index1 = material_index1

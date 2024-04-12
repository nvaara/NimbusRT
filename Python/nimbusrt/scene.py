import numpy as np
from .types import Vec3D
from .edge import Edge
from .path import PathStorage
from ._C import (
    NativeScene,
    InputData,
    NativeObject3D,
    NativeEdge,
)
from plyfile import PlyData
from .io import load_point_cloud
from .material import Material
from .antenna import Antenna


class Scene(NativeScene):
    def __init__(self):
        super().__init__()
        self._types = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("label", "u4"),
            ("material", "u4"),
        ]
        self._point_cloud = {}
        self._transmitters = {}
        self._receivers = {}
        self._edges = []
        self._materials = None
        self._native_edges = []
        self._native_transmitters = {}
        self._native_receivers = {}

    def set_point_cloud(self, cloud):
        if isinstance(cloud, str):
            self._point_cloud = load_point_cloud(cloud, self._types)
        elif isinstance(PlyData):
            self._point_cloud = cloud
        else:
            raise Exception("Input should be path to point cloud or 'plyfile.PlyData'.")
        for t in self._types:
            if not t[0] in self._point_cloud["vertex"]:
                raise Exception(f"Field '{t[0]}' not found in point cloud.")

        material_count = np.max(self._point_cloud["vertex"]["material"]) + 1
        self._materials = np.empty((material_count), dtype=Material)

        for i in range(material_count):
            self._materials[i] = Material()

        for edge in self._edges:
            edge.link_materials(self._materials)

    def add_transmitter(self, name: str, position: Vec3D) -> None:
        self._transmitters[name] = Antenna(position)
        self._native_transmitters[name] = NativeObject3D(position)

    def add_receiver(self, name: str, position: Vec3D) -> None:
        self._receivers[name] = Antenna(position)
        self._native_receivers[name] = NativeObject3D(position)

    def add_edges(self, edges: np.ndarray[Edge]):
        for edge in edges:
            self._native_edges.append(
                NativeEdge(
                    edge.start,
                    edge.end,
                    edge.edge_face0.normal,
                    edge.edge_face1.normal,
                )
            )
            if self._materials is not None:
                edge.link_materials(self._materials)
            self._edges.append(edge)

    @property
    def path_storage(self):
        return self._path_storage

    @property
    def materials(self):
        return self._materials

    def compute_paths(self, input_data: InputData):
        number_of_points = self._point_cloud["vertex"].data.shape[0]
        rt_point_cloud = np.empty(number_of_points, dtype=self._types)
        for t in self._types:
            rt_point_cloud[t[0]] = self._point_cloud["vertex"][t[0]]

        self._path_storage = PathStorage(
            super()._compute_paths(
                input_data,
                rt_point_cloud,
                self._native_edges,
                self._native_transmitters,
                self._native_receivers,
            ),
            self._materials,
            self._edges,
        )

import numpy as np
from ._C import NativeTraceData, NativeInteraction, NativeInteractionType


class Interaction:
    def __init__(self, ia: NativeInteraction, edge_or_material):
        self._label = ia.label
        self._type = ia.type
        self._position = ia.position
        self._normal = ia.normal
        self._edge_or_material = edge_or_material

    @property
    def label(self):
        return self._label

    @property
    def type(self):
        return self._type

    @property
    def position(self):
        return self._position

    @property
    def normal(self):
        return self._normal

    @property
    def material(self):
        return self._edge_or_material

    @property
    def edge(self):
        return self._edge_or_material


class Path:
    def __init__(self, path: NativeTraceData, materials, edges):
        self._interactions = np.empty(path.num_interactions, dtype=Interaction)
        self._time_delay = path.time_delay
        index = 0
        for ia_index in range(path.num_interactions):
            ia = path.interactions[ia_index]
            if ia.type == NativeInteractionType.REFLECTION:
                self._interactions[index] = Interaction(ia, materials[ia.materialID])
            elif ia.type == NativeInteractionType.DIFFRACTION:
                self._interactions[index] = Interaction(ia, edges[ia.label])
            else:
                assert False, f"Bad interaction type: {ia.type}"
            index += 1

    @property
    def interactions(self):
        return self._interactions

    @property
    def time_delay(self):
        return self._time_delay

    def __getitem__(self, index):
        return self._interactions[index]


class PathStorage:
    def __init__(self, paths, materials, edges):
        self._paths = {}
        for tx in paths:
            self._paths[tx] = {}
            for rx in paths[tx]:
                self._paths[tx][rx] = np.array(
                    [Path(trace_data, materials, edges) for trace_data in paths[tx][rx]]
                )

    def __getitem__(self, tx_name):
        return self._paths[tx_name]

from .types import Vec3D


class Antenna:
    def __init__(self, position: Vec3D):
        self._position = position

    @property
    def position(self):
        return self._position

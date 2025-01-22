from sionna.rt.solver_paths import SolverPaths
from sionna.rt.utils import normalize

class NimbusSolverPaths(SolverPaths):
    #def __init__(self, scene, dtype):
    #    super().__init__(scene, dtype)
    
    @property
    def wedges_e_hat(self):
        print("wedges_e_hat")
        return super()._wedges_e_hat

    @wedges_e_hat.setter
    def wedges_e_hat(self, value):
        super()._wedges_e_hat = value

    @property
    def wedges_objects(self):
        return super()._wedges_objects

    @wedges_objects.setter
    def wedges_objects(self, value):
        super()._wedges_objects = value

    @property
    def wedges_normals(self):
        return super()._wedges_normals

    @wedges_normals.setter
    def wedges_normals(self, value):
        super().wedges_normals = value

    def wedges_e_hat_from_start_end(self, start, end):
        super().wedges_e_hat = normalize(start - end)
        return super().wedges_e_hat

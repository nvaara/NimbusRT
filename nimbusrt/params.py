from ._C import NativeRTParams

class RTParams(NativeRTParams):
    def __init__(self,
                 max_depth=2,
                 los=True,
                 reflection=True,
                 scattering=False,
                 diffraction=False,
                 ris=False,
                 refine_max_iterations=50,
                 refine_max_correction_iterations=10,
                 refine_convergence_threshold=1e-4,
                 refine_alpha=0.4,
                 refine_beta=0.4,
                 refine_angle_degrees_threshold=1.0,
                 refine_distance_threshold=0.002,
                 ray_bias=0.05):

        super().__init__()
        self._max_depth = max_depth
        self._los = los
        self._reflection = reflection
        self._scattering = scattering
        self._diffraction = diffraction
        self._ris = ris
        self._refine_max_iterations = refine_max_iterations
        self._refine_max_correction_iterations = refine_max_correction_iterations
        self._refine_convergence_threshold = refine_convergence_threshold
        self._refine_alpha = refine_alpha
        self._refine_beta = refine_beta
        self._refine_angle_degrees_threshold = refine_angle_degrees_threshold
        self._refine_distance_threshold = refine_distance_threshold
        self._ray_bias = ray_bias

    @property
    def max_depth(self):
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value):
        self._max_depth = value

    @property
    def los(self):
        return self._los

    @los.setter
    def los(self, value):
        self._los = value

    @property
    def reflection(self):
        return self._reflection

    @reflection.setter
    def reflection(self, value):
        self._reflection = value

    @property
    def scattering(self):
        return self._scattering

    @scattering.setter
    def scattering(self, value):
        self._scattering = value

    @property
    def diffraction(self):
        return self._diffraction

    @diffraction.setter
    def diffraction(self, value):
        self._diffraction = value

    @property
    def ris(self):
        return self._ris

    @ris.setter
    def ris(self, value):
        self._ris = value

    @property
    def refine_max_iterations(self):
        return self._refine_max_iterations

    @refine_max_iterations.setter
    def refine_max_iterations(self, value):
        self._refine_max_iterations = value

    @property
    def refine_max_correction_iterations(self):
        return self._refine_max_correction_iterations

    @refine_max_correction_iterations.setter
    def refine_max_correction_iterations(self, value):
        self._refine_max_correction_iterations = value

    @property
    def refine_convergence_threshold(self):
        return self._refine_convergence_threshold

    @refine_convergence_threshold.setter
    def refine_convergence_threshold(self, value):
        self._refine_convergence_threshold = value
    
    @property
    def refine_beta(self):
        return self._refine_beta

    @refine_beta.setter
    def refine_beta(self, value):
        self._refine_beta = value

    @property
    def refine_alpha(self):
        return self._refine_alpha
    
    @refine_alpha.setter
    def refine_alpha(self, value):
        self._refine_alpha = value

    @property
    def refine_angle_degrees_threshold(self):
        return self._refine_angle_degrees_threshold

    @refine_angle_degrees_threshold.setter
    def refine_angle_degrees_threshold(self, value):
        self._refine_angle_degrees_threshold = value

    @property
    def refine_distance_threshold(self):
        return self._refine_distance_threshold

    @refine_distance_threshold.setter
    def refine_distance_threshold(self, value):
        self._refine_distance_threshold = value

    @property
    def ray_bias(self):
        return self._ray_bias

    @ray_bias.setter
    def ray_bias(self, value):
        self._ray_bias = value
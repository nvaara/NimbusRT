from ._C import NativeRTParams

class RTParams(NativeRTParams):
    def __init__(self,
                 max_num_interactions=2,
                 scattering=False,
                 diffraction=False,
                 sample_radius=0.015,
                 variance_factor=2.0,
                 sdf_threshold=0.003,
                 refine_max_iterations=50,
                 refine_max_correction_iterations=10,
                 refine_convergence_threshold=1e-4,
                 refine_alpha=0.4,
                 refine_beta=0.4,
                 refine_angle_degrees_threshold=1.0,
                 refine_distance_threshold=0.002,
                 ray_bias=0.05):

        super().__init__()
        self._max_num_interactions = max_num_interactions
        self._scattering = scattering
        self._diffraction = diffraction
        self._sample_radius = sample_radius
        self._variance_factor = variance_factor
        self._sdf_threshold = sdf_threshold
        self._refine_max_iterations = refine_max_iterations
        self._refine_max_correction_iterations = refine_max_correction_iterations
        self._refine_convergence_threshold = refine_convergence_threshold
        self._refine_alpha = refine_alpha
        self._refine_beta = refine_beta
        self._refine_angle_degrees_threshold = refine_angle_degrees_threshold
        self._refine_distance_threshold = refine_distance_threshold
        self._ray_bias = ray_bias

    @property
    def max_num_interactions(self):
        return self._max_num_interactions

    @max_num_interactions.setter
    def max_num_interactions(self, value):
        self._max_num_interactions = value

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
    def sample_radius(self):
        return self._sample_radius

    @sample_radius.setter
    def sample_radius(self, value):
        self._sample_radius = value

    @property
    def variance_factor(self):
        return self._variance_factor

    @variance_factor.setter
    def variance_factor(self, value):
        self._variance_factor = value

    @property
    def sdf_threshold(self):
        return self._sdf_threshold

    @sdf_threshold.setter
    def sdf_threshold(self, value):
        self._sdf_threshold = value

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
import nimbusrt as nrt
from sionna.rt import Transmitter, Receiver, PlanarArray
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    scene = nrt.Scene()
    scene.set_triangle_mesh("Data/CorridorSyntheticSimple_TR.ply", use_face_normals=True)
    scene.set_itu_material_for_label(0, "itu_plasterboard")
    scene.frequency = 60e9
    params = nrt.RTParams(max_num_interactions=2,
                          scattering=False,
                          diffraction=False,
                          refine_convergence_threshold=1e-5)

    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")

    scene.rx_array = scene.tx_array
    scene.add(Transmitter(name="tx", position=[5.79, -2.93, 1.82]))
    scene.add(Receiver(name="rx", position=[8.7, 1.15, 0.95]))
    result_paths = scene.compute_paths(params)
    result_paths.normalize_delays = False
    a, tau = result_paths.cir()
    c = a.numpy().flatten()
    t = tau.numpy().flatten()
    valid_paths = np.nonzero(t > 0.0)
    c, t = np.abs(c[valid_paths])**2, t[valid_paths]
    c_n = c / np.max(c)
    
    plt.stem(t * 1e9, 10 * np.log10(c_n), bottom=-50)
    plt.xlabel("Time delay (ns)")
    plt.ylabel("Path Loss (dB)")
    plt.show()
    
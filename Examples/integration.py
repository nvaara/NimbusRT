import nimbusrt as nrt
from sionna.rt import Transmitter, Receiver, PlanarArray, DirectivePattern, RIS
from sionna.constants import PI
import numpy as np
import matplotlib.pyplot as plt
import warnings 
if __name__ == "__main__":
    scene = nrt.Scene()
    #scene.set_triangle_mesh("Data/CorridorSyntheticSimple_TR.ply")
    scene.set_point_cloud("Data/TestModelPCD.ply", point_radius=0.03)
    for mat_label in range(scene.num_material_labels):
        scene.set_itu_material_for_label(mat_label, "itu_plasterboard", scattering_coefficient=0.2, scattering_pattern=DirectivePattern(100))

    scene.frequency = 60e9
    params = nrt.RTParams(max_depth=3,
                          los=True,
                          reflection=True,
                          scattering=False,
                          diffraction=False,
                          ris=False,
                          refine_convergence_threshold=1e-4)

    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    scene.rx_array = scene.tx_array
    scene.synthetic_array = True
    scene.add(Transmitter(name="tx", position=[2.93, 5.79, 1.82]))
    scene.add(Receiver(name="rx", position=[-1.15, 8.7, 0.95]))
    #scene.add(Transmitter(name="tx", position=[5.79, -2.93, 1.82]))
    #scene.add(Receiver(name="rx", position=[8.7, 1.15, 0.95]))

    width = 1
    num_rows = num_cols = int(width/(0.5*scene.wavelength))
    ris = RIS(name="ris",
          position=[-1.4, 5.79, 1.0],
          num_rows=num_rows,
          num_cols=num_cols)

    #scene.add(ris)
    #ris.phase_gradient_reflector(scene.get("tx").position, scene.get("rx").position)

    cm3d = scene.coverage_map_3d(params, cm_voxel_size=0.5)
    fig = cm3d.show()
    fig.axes[0].yaxis.labelpad = 20
    plt.show()
    #c.show()
    #plt.show()
    #result_paths_tmp = scene.trace_paths(params)
    #spec_paths, diff_paths, scat_paths, ris_paths, spec_paths_tmp, diff_paths_tmp, scat_paths_tmp, ris_paths_tmp = result_paths_tmp
    #result_paths = scene.compute_fields(result_paths_tmp)
#
    #result_paths.normalize_delays = False
    #a, tau = result_paths.cir()
    #print(a)
    #print(tau)
    #c = a.numpy().flatten()
    #t = tau.numpy().flatten()
#
    #c_max = np.max(np.abs(c))
    #c_n = np.abs(c) / c_max
#
    #print(np.abs(c))
    #print(c_n)
    #print(t*1e9)
#
    #plt.stem(t*1e9, np.abs(c), markerfmt='')
    #plt.title("Sionna Paths")
    #plt.xlabel("Time (ns)")
    #plt.ylabel("$\\text{Normalized } |h(t)|^2$")
    #plt.show()
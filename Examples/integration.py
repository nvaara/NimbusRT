import nimbusrt as nrt
from sionna.rt import Transmitter, Receiver, PlanarArray, DirectivePattern
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    scene = nrt.Scene()
    scene.set_triangle_mesh("Data/CorridorSyntheticSimple_TR.ply", use_face_normals=True)
    for mat_label in range(len(scene.radio_materials)):
        scene.set_itu_material_for_label(mat_label, "itu_plasterboard", scattering_coefficient=0.1, scattering_pattern=DirectivePattern(100))
    scene.frequency = 60e9
    params = nrt.RTParams(max_num_interactions=1,
                          scattering=True,
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
    result_paths_tmp = scene.trace_paths(params)

    spec_paths, diff_paths, scat_paths, ris_paths, spec_paths_tmp, diff_paths_tmp, scat_paths_tmp, ris_paths_tmp = result_paths_tmp
    print(scat_paths.vertices.shape)
    print(scat_paths.objects.shape)
    print(scat_paths.mask.shape)
    print(scat_paths.theta_t.shape)
    print(scat_paths.theta_r.shape)
    print(scat_paths.phi_t.shape)
    print(scat_paths.phi_r.shape)
    print(scat_paths.tau.shape)
    print("?")
    print(scat_paths_tmp.k_tx.shape)
    print(scat_paths_tmp.k_rx.shape)
    print(scat_paths_tmp.total_distance.shape)
    print(scat_paths_tmp.k_i.shape)
    print(scat_paths_tmp.k_r.shape)
    print(scat_paths_tmp.num_samples)
    print(scat_paths_tmp.scat_keep_prob)
    print(scat_paths_tmp.scat_last_objects.shape)
    print(scat_paths_tmp.scat_last_vertices.shape)
    print(scat_paths_tmp.scat_last_k_i.shape)
    print(scat_paths_tmp.scat_k_s.shape)
    print(scat_paths_tmp.scat_last_normals.shape)
    print(scat_paths_tmp.scat_src_2_last_int_dist.shape)
    print(scat_paths_tmp.scat_2_target_dist.shape)
    result_paths = scene.compute_fields(result_paths_tmp)

    result_paths.normalize_delays = False
    a, tau = result_paths.cir()
    #c = a.numpy().flatten()
    #t = tau.numpy().flatten()
    #valid_paths = np.nonzero(t > 0.0)
    #c, t = np.abs(c[valid_paths])**2, t[valid_paths]
    #c_n = c / np.max(c)
    
    #plt.stem(t * 1e9, 10 * np.log10(c_n), bottom=-50)
    #plt.xlabel("Time delay (ns)")
    #plt.ylabel("Path Loss (dB)")
    #plt.show()
    
    c = a.numpy().flatten()
    t = tau.numpy().flatten()
    valid_paths = np.nonzero(t > 0.0)
    c, t = c[valid_paths], t[valid_paths]

    #print( tau * 1e9)
    #c = np.abs(c)**2
    indices = np.argsort(t)
    c_max = np.max(np.abs(c))
    c[np.nonzero(c == c_max)] = 0.0
    c_min = 0.0
    c_n = (np.abs(c) - c_min)  / (c_max - c_min)
    #print(c_n)
    plt.stem(t*1e9, c_n, markerfmt='')
    plt.title("Sionna Paths")
    plt.xlabel("Time (ns)")
    plt.ylabel("$\\text{Normalized } |h(t)|^2$")
    plt.show()
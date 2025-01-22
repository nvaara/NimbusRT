import sionna
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, RIS

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement
import tensorflow as tf

scene = load_scene("Data/corridor.xml")

# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

scene.rx_array = scene.tx_array

# Create transmitter
tx = Transmitter(name="tx",
                 position=[5.79,-2.93,1.82])

# Add transmitter instance to scene
scene.add(tx)

# Create a receiver
rx = Receiver(name="rx",
              position=[8.7, 1.15, 0.95])

rx2 = Receiver(name="rx2",
              position=[5.79, 1.15, 0.95])
rx3 = Receiver(name="rx3",
              position=[8.7, 1.15, 0.95])

width = 1 # Width [m] as described in [1]
num_rows = num_cols = int(width/(0.5*scene.wavelength))

#ris = RIS(name="ris",
#          position=[5.79, 1.2, 0.95],
#          orientation=[0,-PI/2,0],
#          num_rows=num_rows,
#          num_cols=num_cols)
#scene.add(ris)
# Add receiver instance to scene
#scene.add(rx)
scene.add(rx)
#scene.add(rx2)
tx.look_at(rx) # Transmitter points towards receiver

scene.frequency = 60e9 # in Hz; implicitly updates RadioMaterials
scene.objects["_unnamed_1"].radio_material = "itu_plasterboard"
from sionna.rt import DirectivePattern
scene.synthetic_array = True
#print(scene.objects["_unnamed_1"].radio_material.complex_relative_permittivity)
scene.objects["_unnamed_1"].radio_material.scattering_coefficient = 0.2
scattering_pattern = DirectivePattern(1)
scene.objects["_unnamed_1"].scattering_pattern = scattering_pattern
alpha_rs = np.array([1,2,3,5,10,30,50,100], np.int32)
scattering_pattern.alpha_r = 100

spec_paths, diff_paths, scat_paths, ris_paths, spec_paths_tmp, diff_paths_tmp, scat_paths_tmp, ris_paths_tmp = scene.trace_paths(max_depth=1, num_samples=1e6, scattering=False, diffraction=True)

#print(spec_paths.vertices.shape)
print(diff_paths.vertices.shape)
print(diff_paths.objects.shape)
print(diff_paths.mask.shape)
print(diff_paths.theta_t.shape)
print(diff_paths.theta_r.shape)
print(diff_paths.phi_t.shape)
print(diff_paths.phi_r.shape)
print(diff_paths.tau.shape)
#print("?")
#print(scat_paths_tmp.k_tx.shape)
#print(scat_paths_tmp.k_rx.shape)
#print(scat_paths_tmp.total_distance.shape)
#print(scat_paths_tmp.k_i.shape)
#print(scat_paths_tmp.k_r.shape)
#print(scat_paths_tmp.num_samples)
#print(scat_paths_tmp.scat_keep_prob)
#print(scat_paths_tmp.scat_last_objects.shape)
#print(scat_paths_tmp.scat_last_vertices.shape)
#print(scat_paths_tmp.scat_last_k_i.shape)
#print(scat_paths_tmp.scat_k_s.shape)
#print(scat_paths_tmp.scat_last_normals.shape)
#print(scat_paths_tmp.scat_src_2_last_int_dist.shape)
#print(scat_paths_tmp.scat_2_target_dist.shape)

paths = scene.compute_fields(spec_paths, diff_paths, scat_paths, ris_paths, spec_paths_tmp, diff_paths_tmp, scat_paths_tmp, ris_paths_tmp)        

paths.normalize_delays = False

a, tau = paths.cir()

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
#print(spec_paths.objects.shape)
#print(scat_paths.objects.shape)
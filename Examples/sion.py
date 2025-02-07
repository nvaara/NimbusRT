import sionna
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, RIS
from sionna.constants import PI

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement
import tensorflow as tf

scene = load_scene("Data/corridor.xml")
scene.frequency = 60e9
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="iso",
                             polarization="V")

scene.rx_array = scene.tx_array

tx = Transmitter(name="tx",
                 position=[5.79,-2.93,1.82])

scene.add(tx)

rx = Receiver(name="rx",
              position=[8.7, 1.15, 0.95])
scene.add(rx)

width = 1 # Width [m] as described in [1]
num_rows = num_cols = int(width/(0.5*scene.wavelength))
ris = RIS(name="ris",
          position=[5.79, 1.4, 1.0],
          orientation=[-PI/2,0,0],
          num_rows=num_rows,
          num_cols=num_cols)

scene.add(ris)
ris.phase_gradient_reflector(tx.position, rx.position)

scene.objects["_unnamed_1"].radio_material = "itu_plasterboard"
from sionna.rt import DirectivePattern
scene.synthetic_array = False
scene.objects["_unnamed_1"].radio_material.scattering_coefficient = 0.2
scattering_pattern = DirectivePattern(100)
scene.objects["_unnamed_1"].scattering_pattern = scattering_pattern

spec_paths, diff_paths, scat_paths, ris_paths, spec_paths_tmp, diff_paths_tmp, scat_paths_tmp, ris_paths_tmp = scene.trace_paths(max_depth=1, num_samples=1e6, reflection=False, scattering=False, diffraction=False, ris=False)
final_paths = scene.compute_fields(spec_paths, diff_paths, scat_paths, ris_paths, spec_paths_tmp, diff_paths_tmp, scat_paths_tmp, ris_paths_tmp)

final_paths.normalize_delays = False
a, tau = final_paths.cir()
print(a)
print(tau)
c = a.numpy().flatten()
t = tau.numpy().flatten()
c_max = np.max(np.abs(c))
c_n = np.abs(c) / c_max

print(np.abs(c))
print(c_n)
print(t*1e9)

plt.stem(t*1e9, c_n, markerfmt='')
plt.title("Sionna Paths")
plt.xlabel("Time (ns)")
plt.ylabel("$\\text{Normalized } |h(t)|^2$")
plt.show()
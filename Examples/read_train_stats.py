import numpy as np
import sys

if __name__ == "__main__":
    data = np.load(sys.argv[1])
    iters = data["eval_iters"]
    print("Iteration \t MRE e_r \t MRE C \t MRE S \t RE t_rms")
    for i in range(iters.shape[0]):
        it = iters[i] 
        r  = data["MRE_rel"][i]
        c  = data["MRE_con"][i]
        s  = data["MRE_sca"][i]
        d  = data["RE_ds"][i]
        print(f"{it} \t\t {r:.4f} \t\t {c:.4f} \t {s:.4f} \t {d:.4f}")

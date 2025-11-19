import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
from tqdm import tqdm

from utils import Bands_def_all_short, PhotometryGenerator


def clip(low, high, mu, std, rng):
    while True:
        point = rng.normal() * std + mu
        if point > low and point < high:
            return point


def generate(num, id, dire, rng):
    """
    dire - where to save
    p_core - probability of beeing in the core
    p2 - probability of beeing a pop2 star
    """
    generator = PhotometryGenerator(Bands_def_all_short)
    for j in range(len(id)):
        what = np.random.permutation(len(generator))[:how_many]
        parameters = np.zeros([batch, 5])
        mags = np.zeros([batch, how_many])
        for k in range(batch):
            ###BOSZ
            g = rng.random() * 5.4 + 0.1
            t_min, t_max = generator.get_boundaries(g)
            ###BOSZ
            T = rng.random() * (t_max - t_min) + t_min
            is_core = rng.choice(a=2, p=[1 - generator.p_core, generator.p_core])
            is_pop2 = rng.choice(a=2, p=[1 - generator.p_pop2, generator.p_pop2])
            if is_core:
                Z_m = 0
                Z_std = 0.2
            else:
                """
                taken from the isochrone package
                """
                if is_pop2:
                    Z_m = -1.49
                    Z_std = 0.4
                else:
                    Z_m = -0.1
                    Z_std = 0.2
            if "basel" in str(type(generator.lib_stell)):
                FEH = rng.random() * 4.5 - 3.5
            elif "kurucz" in str(type(generator.lib_stell)):
                FEH = rng.random() * 3 - 2.5
            elif "phoenix" in str(type(generator.lib_stell)):
                # FEH = rng.random()*5 - 4
                FEH = clip(-4, 1, Z_m, Z_std, rng)
            elif "bosz" in str(type(generator.lib_stell)):
                if g <= 0.5:
                    low_FEH = -1.5
                elif g <= 1:
                    low_FEH = -2.0
                else:
                    low_FEH = -2.5
                FEH = clip(low_FEH, 0.5, Z_m, Z_std, rng)
            else:
                FEH = rng.random() * 3 - 2.5
            Z = generator.to_Z(FEH)
            if is_core:
                AV = clip(0, 1, generator.AV_core_mean, generator.AV_core_std, rng)
            else:
                AV = 0.003 + rng.exponential(generator.AV_exp)
            RV = clip(2.5, 4.5, generator.RV_mean, generator.RV_std, rng)
            # magnitudes = generator.generate_gaia_diff(T,g,Z,AV,RV,what)
            magnitudes = generator.generate_data(T, g, Z, AV, RV, what)
            parameters[k, 0] = 10**T / 1000
            parameters[k, 1] = g
            parameters[k, 2] = FEH
            parameters[k, 3] = AV
            parameters[k, 4] = RV
            mags[k, :] = magnitudes
        np.save(os.path.join(dire, "batch_{}_X".format(id[j])), mags)
        np.save(os.path.join(dire, "batch_{}_Y".format(id[j])), parameters)
        np.save(os.path.join(dire, "batch_{}_bands".format(id[j])), what)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data with pystellibs")
    parser.add_argument(
        "-M", "--M", type=int, default=1, help="number of processors to generate data"
    )
    parser.add_argument("-d", "--dir", type=str, help="directory to store the data")
    args = parser.parse_args()
    data_dir = os.path.join(args.dir)

    data_dir_train = os.path.join(args.dir, "train")
    data_dir_test = os.path.join(args.dir, "test")

    N = 6000
    N_test = 1200
    batch = 256
    how_many = len(Bands_def_all_short)
    generator = PhotometryGenerator(Bands_def_all_short)
    M = args.M
    print(generator.lib_stell)
    print(N, N_test, batch)
    rng = np.random.default_rng()
    rngs = rng.spawn(M)
    processes = []
    for i in range(M):
        idx = np.arange(i, N, M)
        p = mp.Process(target=generate, args=(i, idx, data_dir_train, rngs[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    means = np.zeros([5])
    squares = np.zeros([5])
    data = glob.glob(os.path.join(data_dir_train, "batch_*_Y*"))
    for name in data:
        file = np.load(name)
        means += np.mean(file, axis=0) / len(data)
        squares += np.mean(file**2, axis=0) / len(data)
    std = np.sqrt(squares - means**2)

    summary_dict = {
        "bands": len(generator),
        "N": N,
        "B": batch,
        "how_many": how_many,
        "means": list(means),
        "stds": list(std),
        "type": str(type(generator.lib_stell)),
    }

    with open(os.path.join(data_dir, "train.json"), "w") as f:
        f.write(json.dumps(summary_dict))
    print(means, std)

    processes = []
    rng = np.random.default_rng()
    rngs = rng.spawn(M)
    for i in range(M):
        idx = np.arange(i, N_test, M)
        p = mp.Process(target=generate, args=(i, idx, data_dir_test, rngs[i]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    summary_dict_test = {
        "bands": len(generator),
        "N": N_test,
        "B": batch,
        "how_many": how_many,
    }
    with open(os.path.join(data_dir, "test.json"), "w") as f:
        f.write(json.dumps(summary_dict_test))

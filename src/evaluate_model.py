import json
import numpy as np
import torch
import pandas as pd
from models import SEDTransformer, SED_NDE
from utils import (
    PhotometryGenerator,
    Bands_def_all_short,
    DataBundle,
    is_set,
    M5,
    retry,
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
from Iris import Star, Galactic
from tqdm import tqdm
from sklearn.decomposition import PCA
import corner
import emcee
import multiprocessing as mp
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy
from chainconsumer import Chain, Truth, ChainConsumer, PlotConfig, ChainConfig
from chainconsumer.plotting import plot_contour, plot_truths
import pystellibs
import argparse

example_dir = "./examples/"


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


@torch.no_grad()
def evaluate_dataset(
    name_model,
    name,
    directory,
    to_dump,
    B=256,
    scale=1,
    eta=0,
    low=0,
    how_many=64,
    IS=False,
    CUDA=1,
):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("name: ", name)
    print("weights: ", name_model)
    data = DataBundle(name, B=B)
    data.load(directory)
    print("len: ", len(data.file))
    with open("SED_NDE_{}.json".format(name_model), "r") as f:
        params = json.loads(f.read())
    model = SED_NDE(params["MAF"], params["TF"])
    model.load_state_dict(
        torch.load("SED_NDE_{}.tc".format(name_model), weights_only=True)
    )
    dev = "cuda" if CUDA else "cpu"
    model.to(dev)
    model.eval()

    def helper_function(out, index, *args):
        if len(args[0]) == 0:
            return
        output = generator.log_prob(*args)
        out[index] = output

    if IS:
        generator = PhotometryGenerator(Bands_def_all_short)

    for bands, mags, errors, id in tqdm(data, "sampling"):
        bands = bands.T.int().to(dev)
        mags = mags.T.float().to(dev)
        errors_new = torch.clip(errors, low, torch.inf)
        errors_new = torch.sqrt(
            torch.clip(errors_new.T.float().to(dev), 0, torch.inf) ** 2 * scale**2
            + eta**2
        )
        with torch.autocast("cuda", torch.float16, enabled=params["MAF"]["autocast"]):
            samples, log_probs = model.sample(
                how_many, bands, mags, errors_new, return_log_prob=True
            )
            log_probs = log_probs.cpu().numpy()
        if IS:
            cores = IS
            band_size = bands.shape[0]  # number of unique bands
            samples_batch = np.array_split(samples.cpu().numpy(), cores)
            bands_batch = np.array_split(bands.T.cpu().numpy(), cores)
            mags_batch = np.array_split(mags.T.cpu().numpy(), cores)
            errors_new_batch = np.array_split(errors_new.T.cpu().numpy(), cores)

            processes = []
            man = mp.Manager()
            shared = man.list([None] * cores)  # Shared list to store ordered results
            for i in range(cores):
                p = mp.Process(
                    target=helper_function,
                    args=(
                        shared,
                        i,
                        samples_batch[i],
                        bands_batch[i],
                        mags_batch[i],
                        errors_new_batch[i],
                    ),
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            shared = [x for x in shared if x is not None]
            log_prob_prim = np.concatenate(shared, axis=0)

            weights = scipy.special.softmax(log_prob_prim - log_probs, axis=1)
            samples = samples.cpu().numpy()
            samples[:, :, 0] *= 1000
            best = np.sum(
                weights[:, :, np.newaxis] * samples, axis=1
            )  # /np.sum(weights,axis = 1)
        else:
            samples = samples.cpu().numpy()
            samples[:, :, 0] *= 1000
            best = np.mean(samples, axis=1)
        log_probs = np.mean(log_probs, axis=-1)
        data.upload(id, samples, best, log_probs)
    data.save(directory, save_hp=False)
    RMSE_temp = RMSE(data.file["T_true"].values, data.file["T_50th"].values)
    RMSE_temp_best = RMSE(data.file["T_true"].values, data.file["T_best"].values)
    RMSE_logg = RMSE(data.file["logg_true"].values, data.file["logg_50th"].values)
    RMSE_metal = RMSE(data.file["[M/H]_true"].values, data.file["[M/H]_50th"].values)
    RMSE_logg_best = RMSE(data.file["logg_true"].values, data.file["logg_best"].values)
    RMSE_metal_best = RMSE(
        data.file["[M/H]_true"].values, data.file["[M/H]_best"].values
    )
    print("temp err: ", RMSE_temp, RMSE_temp_best)
    print("metal err: ", RMSE_metal, RMSE_metal_best)
    print("logg err: ", RMSE_logg, RMSE_logg_best)
    MAD_temp = MAD(data.file["T_true"].values, data.file["T_50th"].values)
    MAD_temp_best = MAD(data.file["T_true"].values, data.file["T_best"].values)
    MAD_logg = MAD(data.file["logg_true"].values, data.file["logg_50th"].values)
    MAD_metal = MAD(data.file["[M/H]_true"].values, data.file["[M/H]_50th"].values)
    MAD_logg_best = MAD(data.file["logg_true"].values, data.file["logg_best"].values)
    MAD_metal_best = MAD(data.file["[M/H]_true"].values, data.file["[M/H]_best"].values)
    print("MAD temp: ", MAD_temp, MAD_temp_best)
    print("MAD metal: ", MAD_metal, MAD_metal_best)
    print("MAD logg: ", MAD_logg, MAD_logg_best)
    name_to_log = name.split("_")[1]
    to_dump["{}/MAD_temp".format(name_to_log)] = MAD_temp
    to_dump["{}/MAD_logg".format(name_to_log)] = MAD_logg
    to_dump["{}/MAD_metal".format(name_to_log)] = MAD_metal
    to_dump["{}/RMSE_temp".format(name_to_log)] = RMSE_temp
    to_dump["{}/RMSE_logg".format(name_to_log)] = RMSE_logg
    to_dump["{}/RMSE_metal".format(name_to_log)] = RMSE_metal
    return RMSE_temp


def optimize(names, **kwargs):
    def fun(X):
        print(X)
        eta, scale, low = X
        value = 0
        for name in names:
            value += evaluate_dataset(
                name=name, eta=eta, low=low, scale=scale, **kwargs
            )
        return value

    start = np.array([0.05, 1.3, 0.005])
    out = scipy.optimize.minimize(fun, x0=start, method="Nelder-Mead")
    print(out.x)


@torch.no_grad()
def process_cluster(
    name_model, name, directory, B=256, scale=1, eta=0, low=0, how_many=64, IS=False
):
    print("name: ", name)
    print("weights: ", name_model)
    data = DataBundle(name, B=B)
    data.load(directory)
    print("len: ", len(data.file))
    with open("SED_NDE_{}.json".format(name_model), "r") as f:
        params = json.loads(f.read())
    model = SED_NDE(params["MAF"], params["TF"])
    model.load_state_dict(
        torch.load("SED_NDE_{}.tc".format(name_model), weights_only=True)
    )
    dev = "cuda"
    model.to(dev)
    model.eval()
    if IS:
        generator = PhotometryGenerator(Bands_def_all_short)

    for bands, mags, errors, id in tqdm(data, "sampling"):
        bands = bands.T.int().to(dev)
        mags = mags.T.float().to(dev)
        # print(mags.shape,bands.shape)
        errors_new = torch.clip(errors, low, torch.inf)
        errors_new = torch.sqrt(
            torch.clip(errors_new.T.float().to(dev), 0, torch.inf) ** 2 * scale**2
            + eta**2
        )
        with torch.autocast("cuda", torch.float16, enabled=params["MAF"]["autocast"]):
            samples, log_probs = model.sample(
                how_many, bands, mags, errors_new, return_log_prob=True
            )
            log_probs = log_probs.cpu().numpy()
        if IS:
            log_prob_prim = 5 * generator.log_prob(
                samples.cpu().numpy(),
                bands.T.cpu().numpy(),
                mags.T.cpu().numpy(),
                errors_new.T.cpu().numpy(),
            )
            weights = scipy.special.softmax(log_prob_prim - log_probs, axis=1)
            # weights = np.exp(log_prob_prim - log_probs)
            samples = samples.cpu().numpy()
            samples[:, :, 0] *= 1000
            best = np.sum(
                weights[:, :, np.newaxis] * samples, axis=1
            )  # /np.sum(weights,axis = 1)
        else:
            samples = samples.cpu().numpy()
            samples[:, :, 0] *= 1000
            best = np.mean(samples, axis=1)
        log_probs = np.mean(log_probs, axis=-1)
        data.upload(id, samples, best, log_probs)
    data.save(directory, save_hp=False)
    idx = np.logical_and(
        data.file["Entropy"].values > -np.inf, data.file["leng"].values > 10
    )
    MH = data.file["[M/H]_50th"].values[idx]
    sigma = (
        -data.file["[M/H]_16th"].values[idx] + data.file["[M/H]_84th"].values[idx]
    ) / 2
    th = 2
    while True:
        sigma_prim = 1 / np.sqrt(np.sum(1 / sigma**2))
        mu_prim = np.sum(MH / sigma**2) * sigma_prim**2
        idx = (MH - mu_prim) < th * np.sqrt(sigma_prim**2 + sigma**2)
        if np.sum(idx) == len(idx):
            break
        else:
            MH = MH[idx]
            sigma = sigma[idx]
            print(mu_prim, sigma_prim)

    print("MH = {:.2f}+-{:.2f}, {:.0f}".format(mu_prim, sigma_prim, np.sum(idx)))


def plot_tokens(name, weights):
    with open(name, "r") as f:
        params = json.loads(f.read())
    model = SEDTransformer(
        params["bands"],
        params["d_model"],
        params["heads"],
        params["hidden"],
        params["dropout"],
        params["layers"],
    )
    model.load_state_dict(torch.load(weights))
    data = model.embedding.weight.detach().numpy()
    transformation = PCA(2)  # TSNE(perplexity=2)
    generate = PhotometryGenerator(Bands_def_all_short)
    print(data.shape)
    new_data = transformation.fit_transform(data)
    fig, ax = plt.subplots()
    ax.scatter(new_data[:, 0], new_data[:, 1])
    plt.savefig("PCA.png")


def RMSE(X, Y):
    assert len(X) == len(Y)
    out = (X - Y) ** 2
    out = out[np.logical_not(np.isnan(out))]
    return np.sqrt(np.mean(out))


def MAD(X, Y):
    assert len(X) == len(Y)
    out = np.abs(X - Y)
    out = out[np.logical_not(np.isnan(out))]
    return np.mean(out)


if __name__ == "__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--name", type=str, help="Name of the model", required=True)
    parser.add_argument(
        "--how_many", type=int, default=64, help="How many samples to take"
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Scale of the errors")
    parser.add_argument("--eta", type=float, default=0.01, help="Scale of the errors")
    parser.add_argument("--low", type=float, default=0.01, help="Scale of the errors")
    parser.add_argument("--cuda", type=int, default=1.0, help="Use CUDA for inference")

    args = parser.parse_args()
    to_dump = {}
    # optimize(["APOGEE_disc"],name_model = args.name,directory = "examples/",to_dump = to_dump)
    evaluate_dataset(
        args.name,
        "APOGEE_disc",
        "examples/",
        to_dump,
        scale=args.scale,
        eta=args.eta,
        low=args.low,
        IS=False,
        how_many=args.how_many,
        CUDA=args.cuda,
    )
    evaluate_dataset(
        args.name,
        "APOGEE_halo",
        "examples/",
        to_dump,
        scale=args.scale,
        eta=args.eta,
        low=args.low,
        IS=False,
        how_many=args.how_many,
        CUDA=args.cuda,
    )

    to_dump["eta"] = args.eta
    to_dump["low"] = args.low
    to_dump["scale"] = args.scale

    with open("{}_metrics.json".format(args.name), "w") as file:
        json.dump(to_dump, file)

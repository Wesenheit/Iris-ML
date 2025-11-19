import argparse
import json

import arviz as av
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from models import SED_NDE, SEDTransformer
from utils import (
    DataBundle,
)

example_dir = "./examples/"


@torch.no_grad()
def test_true_coverage(
    name_model,
    name,
    directory,
    B=256,
    scale=1,
    eta=0,
    low=0,
    how_many=64,
    CUDA=1,
):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

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
    confidence_interval = np.linspace(0.01, 0.99, 20)
    coverage = np.zeros([len(confidence_interval), 3])
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
        samples = samples.cpu().numpy()
        samples[:, :, 0] *= 1000
        T_true = data.file["T_true"].values[id]
        T_err = data.file["T_err"].values[id]
        logg_true = data.file["logg_true"].values[id]
        logg_err = data.file["logg_err"].values[id]
        MH_true = data.file["[M/H]_true"].values[id]
        MH_err = data.file["[M/H]_err"].values[id]
        num_per_batch = len(id)
        for idx, conf in enumerate(confidence_interval):
            T_sample = np.random.randn(samples.shape[1], num_per_batch) * T_err + T_true
            MH_sample = (
                np.random.randn(samples.shape[1], num_per_batch) * MH_err + MH_true
            )
            logg_sample = (
                np.random.randn(samples.shape[1], num_per_batch) * logg_err + logg_true
            )
            hdi_T = av.hdi(samples[:, :, 0].T, hdi_prob=conf)
            hdi_logg = av.hdi(samples[:, :, 1].T, hdi_prob=conf)
            hdi_MH = av.hdi(samples[:, :, 2].T, hdi_prob=conf)
            coverage[idx, 0] += (
                np.sum(np.logical_and(T_sample > hdi_T[:, 0], T_sample < hdi_T[:, 1]))
                / how_many
            )
            coverage[idx, 1] += (
                np.sum(
                    np.logical_and(
                        logg_sample > hdi_logg[:, 0], logg_sample < hdi_logg[:, 1]
                    )
                )
                / how_many
            )
            coverage[idx, 2] += (
                np.sum(
                    np.logical_and(MH_sample > hdi_MH[:, 0], MH_sample < hdi_MH[:, 1])
                )
                / how_many
            )
    return confidence_interval, coverage / len(data.file)


if __name__ == "__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(
        description="Test true Bayesian coverage for the model"
    )
    parser.add_argument("--name", type=str, help="Name of the model", required=True)
    parser.add_argument(
        "--how_many", type=int, default=2048, help="How many samples to take"
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Scale of the errors")
    parser.add_argument("--eta", type=float, default=0.01, help="Scale of the errors")
    parser.add_argument("--low", type=float, default=0.01, help="Scale of the errors")
    parser.add_argument("--cuda", type=int, default=1.0, help="Use CUDA for inference")

    args = parser.parse_args()
    X, coverage_disc = test_true_coverage(
        args.name,
        "APOGEE_disc",
        "examples/",
        scale=args.scale,
        eta=args.eta,
        low=args.low,
        how_many=args.how_many,
        CUDA=args.cuda,
    )
    X, coverage_halo = test_true_coverage(
        args.name,
        "APOGEE_halo",
        "examples/",
        scale=args.scale,
        eta=args.eta,
        low=args.low,
        how_many=args.how_many,
        CUDA=args.cuda,
    )
    fig, ax = plt.subplots()
    ax.plot(X, X, color="black")
    ax.plot(X, coverage_disc[:, 0], label="T, disc", color="red", linestyle="--")
    ax.plot(X, coverage_disc[:, 1], label="logg, disc", color="blue", linestyle="--")
    ax.plot(X, coverage_disc[:, 2], label="[M/H], disc", color="green", linestyle="--")

    ax.plot(X, coverage_halo[:, 0], label="T, halo", color="red", linestyle="-")
    ax.plot(X, coverage_halo[:, 1], label="logg, halo", color="blue", linestyle="-")
    ax.plot(X, coverage_halo[:, 2], label="[M/H], halo", color="green", linestyle="-")

    ax.legend()
    ax.grid()
    ax.set_xlabel("Credibility interval")
    ax.set_ylabel("Empirical coverage")
    plt.savefig("true_bayesian_coverage_{}.pdf".format(args.name), dpi=300)

import torch
import neptune
from argparse import ArgumentParser
import numpy as np
from models import SEDTransformer
from tqdm import tqdm
from torch import nn as nn
from utils import CosineWarmupScheduler, BandDataset, str2bool
from torch.utils.data import DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt

torch.backends.cudnn.enabled = True
import json
import time
import os

torch.set_float32_matmul_precision("medium")


def train(args, log=True):
    data_dir = args.dir
    if log:
        run = neptune.init_run(project="wesenheit/Iris-ML")

    with open(os.path.join(data_dir, "train.json")) as f:
        data = json.loads(f.read())

    with open(os.path.join(data_dir, "test.json")) as f:
        data_test = json.loads(f.read())
    data_set = BandDataset(os.path.join(data_dir, "train"), data["N"])
    data_set_test = BandDataset(os.path.join(data_dir, "test"), data_test["N"])
    model = SEDTransformer(
        data["bands"],
        args.d_model,
        args.heads,
        args.hidden_dim,
        args.dropout,
        args.layers,
    )
    param_dict = {}
    param_dict["layers"] = args.layers
    param_dict["d_model"] = args.d_model
    param_dict["bands"] = data["bands"]
    param_dict["hidden"] = args.hidden_dim
    param_dict["heads"] = args.heads
    param_dict["dropout"] = args.dropout
    param_dict["name"] = args.name
    if log:
        run["params"] = param_dict

    with open("params_{}.json".format(args.name), "w") as f:
        f.write(json.dumps(param_dict))
    optim = torch.optim.Adam(model.parameters(), args.learning_rate)
    data_loader = DataLoader(data_set, args.batch_size, True)
    data_loader_test = DataLoader(data_set_test, args.batch_size, False)
    device = "cuda" if args.cuda else "cpu"
    how_many_at_time = data["how_many"]
    low = 4
    if args.autocast:
        print("enabled AMP!")
        scaler = torch.amp.GradScaler(device)
    if args.compile:
        print("compiling!")
        train_model = torch.compile(model, backend="inductor")
    else:
        train_model = model
    if args.load:
        print("loading weights!")
        model.load_state_dict(torch.load("SED_transformer.tc"))
    shed = CosineWarmupScheduler(optim, args.num_epoche / 10, args.num_epoche)
    loss_train_arr = []
    loss_test_arr_T = []
    loss_test_arr_EBV = []
    means = np.zeros(data["bands"])
    squares = np.zeros(data["bands"])
    for X, Y, bands in tqdm(data_loader, "estimating mean", disable=args.disable):
        X = X - torch.mean(X, axis=-1, keepdim=True)
        for j in range(X.shape[0]):
            bands_temp = bands[j]
            values = np.sum(X[j].numpy(), axis=0)
            sq_val = np.sum(X[j].numpy() ** 2, axis=0)
            for i, k in enumerate(bands_temp):
                means[k] += values[i] / (len(data_set) * data["B"])
                squares[k] += sq_val[i] / (len(data_set) * data["B"])
    stds = np.sqrt(squares - means**2)
    print(means, stds)
    means = torch.from_numpy(means).float()
    stds = torch.from_numpy(stds).float()
    model.means = means
    model.stds = stds
    model.to(device)
    means_Y = np.array(data["means"], dtype=np.float32)
    stds_Y = np.array(data["stds"], dtype=np.float32)
    means_Y = torch.from_numpy(means_Y).to(device)
    stds_Y = torch.from_numpy(stds_Y).to(device)
    for i in range(args.num_epoche):
        epoche_loss = 0
        train_model.train()
        for X, Y, bands in tqdm(data_loader, "train", disable=args.disable):
            X = X.to(device).float()
            Y = Y.to(device).float()
            bands = bands.to(device).float()
            for j in range(X.shape[0]):
                idx = np.random.permutation(how_many_at_time)
                idx = np.stack(
                    [np.random.permutation(how_many_at_time) for _ in range(data["B"])]
                )
                num = np.random.randint(low, how_many_at_time)
                idx = torch.from_numpy(idx[:, :num]).to(device)
                optim.zero_grad()
                X_temp = torch.gather(X[j], 1, idx).permute(1, 0)
                Y_temp = Y[j]
                bands_temp = torch.stack([bands[j][i] for i in idx]).int().T
                if args.autocast:
                    with torch.autocast(device, dtype=torch.float16):
                        val = train_model(bands_temp, X_temp)
                        loss = F.mse_loss(
                            val / stds_Y, Y_temp / stds_Y, reduction="sum"
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    val = train_model(bands_temp, X_temp)
                    loss = F.mse_loss(val / stds_Y, Y_temp / stds_Y, reduction="sum")
                    loss.backward()
                    optim.step()
                epoche_loss += loss.item() / (len(data_set) * data["B"])

        train_model.eval()
        epoche_loss_test_T = 0
        epoche_loss_test_EBV = 0
        epoche_loss_test_metal = 0
        for X, Y, bands in tqdm(data_loader_test, "evaluating", disable=args.disable):
            X = X.to(device).float()
            Y = Y.to(device).float()
            bands = bands.to(device).float()
            for j in range(X.shape[0]):
                idx = np.random.permutation(how_many_at_time)
                idx = np.stack(
                    [np.random.permutation(how_many_at_time) for _ in range(data["B"])]
                )
                num = np.random.randint(low, how_many_at_time)
                idx = torch.from_numpy(idx[:, :num]).to(device)
                optim.zero_grad()
                X_temp = torch.gather(X[j], 1, idx).permute(1, 0)
                Y_temp = Y[j]
                bands_temp = torch.stack([bands[j][i] for i in idx]).int().T
                with torch.no_grad():
                    if args.autocast:
                        with torch.autocast(device, dtype=torch.float16):
                            val = train_model(bands_temp, X_temp)
                    else:
                        val = train_model(bands_temp, X_temp)
                    loss_T = F.mse_loss(val[:, 0], Y_temp[:, 0], reduction="sum")
                    loss_EBV = F.mse_loss(val[:, -1], Y_temp[:, -1], reduction="sum")
                    loss_metal = F.mse_loss(val[:, -2], Y_temp[:, -2], reduction="sum")
                    epoche_loss_test_T += loss_T.item() / (
                        len(data_set_test) * data_test["B"]
                    )
                    epoche_loss_test_EBV += loss_EBV.item() / (
                        len(data_set_test) * data_test["B"]
                    )
                    epoche_loss_test_metal += loss_metal.item() / (
                        len(data_set_test) * data_test["B"]
                    )
        shed.step()
        print(
            "Num epoche: {:.0f}, loss: {:.3f}, test loss temperature: {:.3f}, test loss EBV: {:.3f}, test loss metal: {:.3f}".format(
                i,
                np.sqrt(epoche_loss),
                np.sqrt(epoche_loss_test_T),
                np.sqrt(epoche_loss_test_EBV),
                np.sqrt(epoche_loss_test_metal),
            )
        )
        loss_test_arr_T.append(epoche_loss_test_T)
        loss_test_arr_EBV.append(epoche_loss_test_EBV)
        loss_train_arr.append(epoche_loss)
        if log:
            run["train/accuracy"].append(np.sqrt(epoche_loss))
            run["test/accuracy_T"].append(np.sqrt(epoche_loss_test_T))
            run["test/accuracy_EBV"].append(np.sqrt(epoche_loss_test_EBV))
            run["test/accuracy_MH"].append(np.sqrt(epoche_loss_test_metal))
        if i % 30 == 0 and i > 0:
            print("Saving model")
            torch.save(model.state_dict(), "SED_transformer_{}.tc".format(args.name))
    torch.save(model.state_dict(), "SED_transformer_{}.tc".format(args.name))
    if log:
        run.stop()
    loss_test_arr_T = np.array(loss_test_arr_T)
    loss_test_arr_EBV = np.array(loss_test_arr_EBV)
    loss_train_arr = np.array(loss_train_arr)


if __name__ == "__main__":
    parser = ArgumentParser(description="SED transformer training")
    parser.add_argument("-b", "--batch-size", default=10, type=int, help="Batch size")
    parser.add_argument(
        "-n", "--num-epoche", default=500, type=int, help="number of epochs to train"
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=3e-5,
        type=float,
        help="Learning rate, use with cosine annealing",
    )
    parser.add_argument(
        "-c", "--cuda", type=str2bool, nargs="?", const=True, default=True
    )
    parser.add_argument(
        "-H", "--heads", default=8, type=int, help="number of heads in transformer"
    )
    parser.add_argument("-l", "--layers", default=8, type=int, help="number of layers")
    parser.add_argument(
        "-d",
        "--d-model",
        default=256,
        type=int,
        help="dimensionality of the model, equal to the effective token dimension",
    )
    parser.add_argument(
        "-hid",
        "--hidden-dim",
        default=3 * 256,
        type=int,
        help="hidden number of dimensions in the feed-forward",
    )
    parser.add_argument("-D", "--dropout", default=0.05, type=float)
    parser.add_argument(
        "-com",
        "--compile",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="compile model with torch.compile",
    )
    parser.add_argument(
        "-a",
        "--autocast",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="use automatic mixed precision (AMP)",
    )
    parser.add_argument(
        "-load",
        "--load",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Load weights at the beginning (for restarting)",
    )
    parser.add_argument(
        "-na", "--name", type=str, required=True, help="name of the model"
    )
    parser.add_argument(
        "-dir", "--dir", type=str, required=True, help="directory to save"
    )
    parser.add_argument(
        "-dis", "--disable", type=int, default=True, help="disable tqdm"
    )
    print(parser.parse_args())
    train(parser.parse_args(), True)

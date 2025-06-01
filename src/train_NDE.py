import torch
import neptune
from argparse import ArgumentParser
import argparse
import numpy as np
from models import SED_NDE
from tqdm import tqdm
from torch import nn as nn
from utils import CosineWarmupScheduler,BandDataset,str2bool
from torch.utils.data import DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt
torch.backends.cudnn.enabled = True
import json
import time
import os
torch.set_float32_matmul_precision('medium')

def train(args,log = True,save = 5):
    data_dir = args.dir
    if log:
        run = neptune.init_run(project='wesenheit/Iris-ML')

    print("name: ",args.name)
    print("name load: ",args.name_load)
    with open(os.path.join(data_dir,"train.json")) as f:
        data = json.loads(f.read())

    with open(os.path.join(data_dir,"test.json")) as f:
        data_test = json.loads(f.read())
    data_set = BandDataset(os.path.join(data_dir,"train"),data["N"],noise = True)
    data_set_test = BandDataset(os.path.join(data_dir,"test"),data_test["N"],noise = True)
    
    with open("params_{}.json".format(args.name_load),"r") as f:
        params_transformer = json.loads(f.read())

    device = "cuda" if args.cuda else "cpu"
    params_MAF = vars(args)
    means = np.array(data["means"],dtype = np.float32)
    stds = np.array(data["stds"],dtype = np.float32)
    means[3] = 0
    stds[3] = 1
    model = SED_NDE(params_MAF,params_transformer)
    model.MAF.net[0].set_means(torch.from_numpy(means),torch.from_numpy(stds))
    weights = torch.load("SED_transformer_{}.tc".format(args.name_load),weights_only=True)
    model.Transformer.load_state_dict(weights)
    model.eval()
    if log:
        run["params"] = params_MAF
    optim = torch.optim.Adam(model.parameters(),args.learning_rate)
    data_loader = DataLoader(data_set,args.batch_size,True)
    data_loader_test = DataLoader(data_set_test,args.batch_size,False)
    if args.load:
        print("loading")
        model.load_state_dict(torch.load("SED_NDE_{}.tc".format(args.name)))
    model = model.to(device)
    how_many_at_time = data["how_many"]
    low = 6
    if args.compile:
        print("compiling!")
        train_model = torch.compile(model,backend="inductor")
    else:
        train_model = model
    shed = torch.optim.lr_scheduler.CosineAnnealingLR(optim,args.num_epoche*len(data_loader))
    loss_train_arr = []
    loss_test_arr = []
    if args.autocast:
        print("AMP enabled")
        scaler = torch.amp.GradScaler(device)
    final = {}
    final["MAF"] = params_MAF
    final["TF"] = params_transformer
    with open("SED_NDE_{}.json".format(args.name),"w") as f:
        f.write(json.dumps(final))
    for n_ep in range(args.num_epoche):
        epoche_loss = 0
        train_model.train()
        for X,errors,Y,bands in tqdm(data_loader,"train",disable = args.disable):
            X = X.to(device).float()
            Y = Y.to(device).float()
            bands = bands.to(device)
            errors = errors.to(device).float()
            permutation = np.random.permutation(X.shape[0])
            for j in permutation:
                idx = np.stack([np.random.permutation(how_many_at_time) for _ in range(data["B"])])
                num = np.random.randint(low,how_many_at_time)
                idx = torch.from_numpy(idx[:,:num]).to(device)
                optim.zero_grad()
                X_temp = torch.gather(X[j],1,idx).permute(1,0)
                errors_temp = torch.gather(errors[j],1,idx).permute(1,0)
                Y_temp = Y[j]
                bands_temp = torch.stack([bands[j][i] for i in idx]).int().T
                with torch.autocast(device_type = "cuda",dtype = torch.float16,enabled = args.autocast):
                    val = -train_model(Y_temp,bands_temp,X_temp,errors_temp)
                    loss = torch.mean(val)
                if torch.isnan(loss):
                    print("NAN!")
                    model.load_state_dict(torch.load("SED_NDE_{}.tc".format(args.name)))
                else:
                    if args.autocast:
                        scaler.scale(loss).backward()
                        scaler.step(optim)
                        scaler.update()
                    else:
                        loss.backward()
                        optim.step()
                    epoche_loss += loss.item()/(len(data_loader)*X.shape[0])
            shed.step()
        epoche_loss_test = 0
        train_model.eval()
        for X,errors,Y,bands in tqdm(data_loader_test,"evaluating",disable = args.disable):
            X = X.to(device).float()
            Y = Y.to(device).float()
            bands = bands.to(device)
            errors = errors.to(device).float()
            for j in range(X.shape[0]):
                num = np.random.randint(low,how_many_at_time)
                idx = np.stack([np.random.permutation(how_many_at_time) for _ in range(data["B"])])
                num = np.random.randint(low,how_many_at_time)
                idx = torch.from_numpy(idx[:,:num]).to(device)
                optim.zero_grad()
                X_temp = torch.gather(X[j],1,idx).permute(1,0)
                errors_temp = torch.gather(errors[j],1,idx).permute(1,0)
                Y_temp = Y[j]
                bands_temp = torch.stack([bands[j][i] for i in idx]).int().T
                with torch.no_grad():
                    with torch.autocast(device_type = "cuda",dtype = torch.float16,enabled = args.autocast):
                        loss = torch.mean(-train_model(Y_temp,bands_temp,X_temp,errors_temp))
                    epoche_loss_test += loss.item()/(len(data_loader_test)*X.shape[0])
        if n_ep%save == 0 and n_ep>0:
            MSE = np.zeros(5)
            for X,errors,Y,bands in tqdm(data_loader_test,"sampling",disable = args.disable):
                X = X.to(device).float()
                Y = Y.to(device).float()
                bands = bands.to(device)
                errors = errors.to(device).float()
                for j in range(X.shape[0]):
                    X_temp = X[j].permute(1,0)
                    Y_temp = Y[j]
                    errors_temp = errors[j].permute(1,0)
                    bands_temp = bands[j].reshape(-1,1).repeat(1,data_test["B"]).int()
                    with torch.autocast(device_type = "cuda",dtype = torch.float16,enabled = args.autocast):
                        samples = model.sample(128,bands_temp,X_temp,errors_temp).cpu().numpy()
                    predicts = np.median(samples,axis = 1)
                    MSE += np.mean((Y_temp.cpu().numpy()-predicts)**2,axis = 0)/X.shape[0]
            MSE /= len(data_loader_test)
            MSE = np.sqrt(MSE)
            T_err = MSE[0]
            EBV_err = MSE[-1]
            print("Sampling errors: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(*MSE))
        print("Num epoche: {:.0f}, loss: {:.4f}, test loss: {:.4f}".format(n_ep,epoche_loss,epoche_loss_test))
        loss_test_arr.append(epoche_loss_test)
        loss_train_arr.append(epoche_loss)
        if log:
            run["train/neglink"].append(epoche_loss)
            run["test/neglink"].append(epoche_loss_test)
            if n_ep%save==0 and n_ep > 0:
                run["test/T_sample"].append(T_err)
                run["test/EBV_sample"].append(EBV_err)
        if n_ep < save or (n_ep%save == 0 and n_ep>0):
            print("Saving model")
            torch.save(model.state_dict(),"SED_NDE_{}.tc".format(args.name))
            torch.save(optim.state_dict(),"NDE_optim.tc")
            torch.save(shed.state_dict(),"NDE_shed.tc")
    torch.save(model.state_dict(),"SED_NDE_{}.tc".format(args.name))
    if log:
        run.stop()


if __name__=="__main__":
    parser = ArgumentParser(description = "SED Neural Density Estimation")
    parser.add_argument("-b","--batch-size",default = 20,type = int,help = "number of bands to load at once")
    parser.add_argument("-n","--num-epoche",default = 500,type = int,help = "number of training epoches")
    parser.add_argument("-lr","--learning-rate",default = 3e-5,type = float)
    parser.add_argument("-c","--cuda",default = True,type = bool, help = "use CUDA")
    parser.add_argument("-com","--compile",default = True,type = bool, help = "compile")
    parser.add_argument("-X","--X-dim",default = 5,type = int)
    parser.add_argument("-N","--N",default = 10,type = int)
    parser.add_argument("-H","--hidden-dims",default = [256,256])
    parser.add_argument("-cond","--cond-dim",default = 256,type = int)
    parser.add_argument("-comp","--n-comp",default = 10,type = int)
    parser.add_argument("-l","--load",default = False,type = bool)
    parser.add_argument("-nl","--name-load",type = str,required = True, help = "what preprocessor to load")
    parser.add_argument("-na","--name",required = True,type = str, help = "name of the model")
    parser.add_argument("-a","--autocast",type=str2bool, nargs='?', const=True, default = False)
    parser.add_argument("-dir","--dir",type = str,required = True, help = "directory with the data")
    parser.add_argument("-dis","--disable",type = int,default = True,help = "disable tqmd")
    train(parser.parse_args(),True)

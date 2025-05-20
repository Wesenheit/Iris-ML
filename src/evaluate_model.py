import json
import sys, os
import numpy as np
import torch
from time import sleep
import pandas as pd
from models import SEDTransformer,SED_NDE
from utils import PhotometryGenerator,Bands_def_all_short,DataBundle,is_set,M5,retry
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
from Iris import Star,Galactic
from tqdm import tqdm
from astropy.io import fits
from sklearn.decomposition import PCA
import corner
import emcee
from astropy.table import unique
import multiprocessing as mp
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy
from chainconsumer import Chain, Truth,ChainConsumer, PlotConfig,ChainConfig
from chainconsumer.plotting import plot_contour, plot_truths
import threading
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import pystellibs

example_dir = "./examples/"

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


@torch.no_grad()
def test_NDE(name,ra,dec,scale = 1,eta = 0.0,IS = False,MCMC = False):
    with open("SED_NDE_{}.json".format(name),"r") as f:
        final = json.loads(f.read())
    model = SED_NDE(final["MAF"],final["TF"])
    model.load_state_dict(torch.load("SED_NDE_{}.tc".format(name)))
    dev = "cuda"
    model.to(dev)
    model.eval()
    print(ra,dec)
    obj = Star("test", ra = ra, dec = dec, catalog = Galactic,d = 3, d_err = 1)
    obj.lib_stell = pystellibs.Bosz(2000,1,interpolator = "ndlinear")
    obj.dis_norm = 5
    obj.get_SDSS()
    #obj.get_all()
    obj.get_photo("II/379/smssdr4")
    #obj.get_photo("V/154/sdss16")
    obj.get_photo("I/355/gaiadr3")
    #obj.get_photo("II/336/apass9")
    obj.get_photo("II/246/out")
    obj.get_photo("II/281/2mass6x")
    obj.get_photo("II/335/galex_ais")
    obj.get_photo("II/312/ais")
    obj.get_photo("II/312/mis")
    obj.get_photo("II/349/ps1")
    obj.get_photo("II/328/allwise")
    obj.get_photo("II/311/wise")
    #obj.delete("WISE_RSR_W2")
    #id = obj.err > 0.2
    #obj.filters = obj.filters[np.logical_not(id)]
    #obj.ampl = obj.ampl[np.logical_not(id)]
    #obj.err = obj.err[np.logical_not(id)]
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    catalogs = Vizier(columns = ["RAJ2000","DEJ2000",
                                 "Teff","e_Teff","f_Teff",
                                 "AkT",
                                 "[M/H]","e_[M/H]","f_[M/H]",
                                 "logg","e_logg","f_logg",
                                 "AFlag"]).query_region(coord,radius = 0.4*u.arcsec,catalog = "III/284/allstars")
    true = np.zeros(5)-99
    try:
        catalogs = catalogs[0]
        print(catalogs)
        true[0] = catalogs["Teff"]
        true[1] = catalogs["logg"]
        true[2] = catalogs["[M/H]"]
    except:
        pass
    #obj.list_filters()
    arr = np.array(Bands_def_all_short)
    mags = obj.ampl
    errors = obj.err
    names = obj.filters
    id_to_use = []
    bands = []
    remove = []
    for i,name in enumerate(names):
        if name in arr:
            bands.append(np.nonzero(name==arr)[0])
            id_to_use.append(i)
            print(name,mags[i],errors[i])
        else:
            remove.append(name)
    for name in remove:
        obj.delete(name)
    obj.prepare_data(None)
    bands = np.stack(bands).squeeze(1)
    id_to_use = np.array(id_to_use)
    mags = mags[id_to_use]
    names = names[id_to_use]
    errors = errors[id_to_use]
    mags = torch.from_numpy(mags).to(dev).float().unsqueeze(1)
    bands = torch.from_numpy(bands).int().to(dev).unsqueeze(1)
    errors = torch.sqrt(torch.from_numpy(errors).float().to(dev).unsqueeze(1)**2*scale**2+eta**2)
    samples = model.sample(2048*16,bands,mags,errors).squeeze(0)
    samples_numpy = samples.clone().cpu().numpy()
    samples_numpy[:,0] *= 1000
    data_org = pd.DataFrame(columns = ["T","logg","[M/H]","AV","RV"],data = samples_numpy)
    c = ChainConsumer()
    chain1 = Chain(samples = data_org,name = "NPE")
    c.add_chain(chain1)
    generator = PhotometryGenerator(Bands_def_all_short)
    idx_IS = bands.cpu().numpy().T
    mags_IS = mags.cpu().numpy().T
    errors_IS = errors.cpu().numpy().T
    if IS:
        print("Performing IS")
        log_probs_pred = 5 * model(samples,bands.repeat_interleave(samples.shape[0],1),
                                   mags.repeat_interleave(samples.shape[0],1),
                                   errors.repeat_interleave(samples.shape[0],1)).cpu().numpy()
        for_IS = samples_numpy.copy()
        for_IS[:,0] /= 1000
        log_probs = generator.log_prob(for_IS.reshape(1,-1,5),idx_IS,mags_IS,errors_IS).squeeze(0)
        weights = scipy.special.softmax(log_probs - log_probs_pred)
        idx = np.random.choice(samples_numpy.shape[0], p = weights,size = samples_numpy.shape[0])
        samples_SIR = samples_numpy[idx,:]
        data_SIR = pd.DataFrame(columns = ["T","logg","[M/H]","AV","RV"],data = samples_SIR)
        chain2 = Chain(samples = data_SIR,name = "NPE + IS",kde = 1.0)
        c.add_chain(chain2)
        print("SIR")
        print(np.mean(samples_SIR,axis = 0))
        print(np.std(samples_SIR,axis = 0))
    if MCMC:
        print("Performing MCMC")
        start = np.array([5,2,-0.1,0.2,3.1]).reshape(5,-1)
        start = start + np.random.randn(5,32)*0.01*start
        start = start.T
        sampler = emcee.EnsembleSampler(
                32, 5, lambda x: generator.log_prob(x.reshape(1,-1,5),idx_IS,mags_IS,errors_IS).squeeze(0),vectorize = True
            )
        sampler.run_mcmc(start, 10000, progress=True);
        samples_MCMC = sampler.get_chain(discard = 4000,flat=True)
        print(samples_MCMC)
        #range_z = (np.min(obj.lib_stell.Z),np.max(obj.lib_stell.Z))
        #samples_MCMC, chi2, pred = obj.run_chain_full(5000,5000,32,logg_range = (0,5.5),Z_range = range_z,RV_range=(1.5,4.5),start = start) 
        #samples_MCMC[:,2] = np.log10(samples_MCMC[:,2])
        samples_MCMC[:,0] *= 1000#10**(samples_MCMC[:,0])
        data_MCMC = pd.DataFrame(columns = ["T","logg","[M/H]","AV","RV"],data = samples_MCMC)
        chain_MCMC = Chain(samples = data_MCMC,name = "MCMC")
        c.add_chain(chain_MCMC)


    #truth = Truth(location = {"T":true[0],"logg":true[1],"[M/H]":true[2]},line_style=":")
    c.set_override(ChainConfig(shade_alpha=0.1))
    fig = c.plotter.plot()
    axs = fig.axes
    corner.overplot_lines(fig, true, color="C1")
    print("VALUES")
    print(np.mean(samples_numpy,axis = 0))
    print(np.std(samples_numpy,axis = 0))
    plt.savefig("test_NDE.pdf")


def check_apogee(number:int, iner:bool, para:bool = False, num_p:int = 4,batch:int = 2000,name_server:str = "vizier.cds.unistra.fr"):
    if iner:
        fil = {'GLAT':'<20 & >-20','logg':'>0','TEFF':'< 10000','PROGRAMNAME':'!= yso',
               'GAIAEDR3_PHOT_G_MEAN_MAG':'>12'}
    else:
        fil = {'GLAT':'<-20 | >20','logg':'>0','TEFF':'< 10000','PROGRAMNAME':'!= yso & != Drout_18b',
               'GAIAEDR3_PHOT_G_MEAN_MAG':'>12'}
    catalogs = Vizier(row_limit = number,column_filters = fil, vizier_server = name_server,
                      columns = ["RAJ2000","DEJ2000",
                                 "Teff","e_Teff","f_Teff",
                                 "Ak","ALPHA_M",
                                 "[M/H]","e_[M/H]","f_[M/H]",
                                 "logg","e_logg","f_logg",
                                 "AFlag","GaiaEDR3","plx","Gmag"]).get_catalogs("III/286/catalog")[0]
    print(catalogs.columns) 
    name = "APOGEE_"
    if iner:
        name += "disc"
    else:
        name += "halo"
    print("name: ",name)
    if para:
        print("using: {} proc".format(num_p))

    bits = [7,23,19,27,3,40,41]
    idx = np.ones([len(catalogs)]).astype(bool)
    print("len before quality cuts: ",len(catalogs))
    for i in range(len(catalogs)):
        for bit in bits:
            if is_set(bit,catalogs["AFlag"][i]):
                idx[i] = False
    catalogs = catalogs[idx] #remove those object that do not have good flags

    catalogs = unique(catalogs,"GaiaEDR3") #remove duplicates from the same GaiaDR3 index
    print("len after quality cuts: ",len(catalogs))
    sdss = {"V/154/sdss16":[["psfMag_u","psfMagErr_u","psfMag_g","psfMagErr_g",
                            "psfMag_r","psfMagErr_r","psfMag_i","psfMagErr_i",
                            "psfMag_z","psfMagErr_z",],["SDSS_u","SDSS_g","SDSS_r","SDSS_i","SDSS_z"],0.3,[True,True,True,True,True]]}
    cat_to_use = Galactic | sdss
    
    data = DataBundle(name,cat_to_use,leng = len(catalogs),bands = Bands_def_all_short)
    if iner:
        data.dis_norm = 0.5
    else:
        data.dis_norm = 1
    print("norm: ",data.dis_norm)
    for i in tqdm(range(len(catalogs))):
        data.new_row(i,
                     catalogs["GaiaEDR3"][i],
                     catalogs["RAJ2000"][i],
                     catalogs["DEJ2000"][i],
                     catalogs["plx"][i],
                     catalogs["Gmag"][i],
                     0,
                     [catalogs["Teff"][i],
                     catalogs["e_Teff"][i],
                     catalogs["[a/M]"][i],
                     catalogs["Ak"][i],
                     catalogs["logg"][i],
                     catalogs["e_logg"][i],
                     catalogs["[M/H]"][i],
                     catalogs["e_[M/H]"][i]])

    names = ["I/355/gaiadr3",
             "II/246/out",
             "II/335/galex_ais",
             "II/312/mis",
             "II/349/ps1",
             "II/379/smssdr4",
             "V/154/sdss16",
             #"II/336/apass9",
             #"II/311/wise",
             ]
    if iner:
        names += ["II/328/allwise",]
    else:
        names += ["II/365/catwise",]
    print(names)
    names_remove = ["J/A+A/675/A195/mast",
                    "I/358/varisum",
                    "B/vsx/vsx",
                    "J/A+A/674/A22/catalog",
                    "J/A+A/648/A44/tabled1",
                    "J/MNRAS/433/3398/table2",
                    "J/MNRAS/503/3975/catalog",
                    "J/ApJS/259/11/table4",
                    "J/ApJS/209/27/xmystix",
                    "J/ApJS/209/31",
                    "J/ApJS/247/66/table8",
                    "I/361/lenscand",
                    "I/356/galcand",
                    ]
    tasks = []
    remove = []
    for num in range(0,len(data.file),batch):
        for name in names_remove:
            remove.append((name,num))
    if para:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {executor.submit(retry(data.check_if_exist), tup[0], tup[1],batch,vizier_server = name_server): tup for tup in remove}

        data.remove() 

        for num in range(0,len(data.file),batch):
            for name in names:
                tasks.append((name,num))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {executor.submit(retry(data.get_photo), tup[0], tup[1],batch, vizier_server = name_server): tup for tup in tasks}
    else:
        for task in remove:
            retry(data.check_if_exist)(task[0],task[1],batch, vizier_server = name_server)

        data.remove()

        for num in range(0,len(data.file),batch):
            for name in names:
                tasks.append((name,num))

        for task in tasks:
            retry(data.get_photo)(task[0],task[1],batch, vizier_server = name_server)
    data.save("examples/")
    print(data.file)

def check_lamost(number=100,para  = False,num_p = 10,AFGK = True):
    if AFGK:
        cat = "V/156/dr7slrs"
        metal = "[Fe/H]"
        metal_err = "e_[Fe/H]"
        flat = "__Fe_H_"
        flat_err = "e__Fe_H_"
    else:
        cat = "V/156/dr7mslrs"
        metal = "Z"
        metal_err = "e_Z"
        flat = metal
        flat_err=  metal_err
    fil = {"snru": ">20",
           "snrg":">20",
           "snrr":">20",
           "snri":">20",
           "snrz":">20",
           "Gmag":">12",
           }
    catalogs = Vizier(row_limit = number,column_filters = fil,
                      columns = ["GaiaDR2",
                                 "Gmag",
                                "RAJ2000","DEJ2000",
                                 "Teff","e_Teff",
                                 metal,metal_err,
                                 "logg","e_logg"]).get_catalogs(cat)[0]
    if AFGK:
        label = "_AFGK"
    else:
        label = "_M"
    name = "LAMOST" + label
    sdss = {"V/154/sdss16":[["psfMag_u","psfMagErr_u","psfMag_g","psfMagErr_g",
                            "psfMag_r","psfMagErr_r","psfMag_i","psfMagErr_i",
                            "psfMag_z","psfMagErr_z",],["SDSS_u","SDSS_g","SDSS_r","SDSS_i","SDSS_z"],0.3,[True,True,True,True,True]]}
    cat_to_use = Galactic | sdss
    
    data = DataBundle(name,cat_to_use,leng = len(catalogs),bands = Bands_def_all_short)
    data.dis_norm = 0.5
    print("norm: ",data.dis_norm)
    for i in tqdm(range(len(catalogs))):
        data.new_row(i,
                     catalogs["GaiaDR2"][i],
                     catalogs["RAJ2000"][i],
                     catalogs["DEJ2000"][i],
                     0,
                     catalogs["Gmag"][i],
                     0,
                     [catalogs["Teff"][i],
                     catalogs["e_Teff"][i],
                     0,
                     0,
                     catalogs["logg"][i],
                     catalogs["e_logg"][i],
                     catalogs[flat][i],
                     catalogs[flat_err][i]])

    names = ["I/355/gaiadr3",
             "II/246/out",
             "II/335/galex_ais",
             "II/312/mis",
             "II/349/ps1",
             "II/379/smssdr4",
             "II/336/apass9",
             "V/154/sdss16",
             #"II/328/allwise",
             #"II/311/wise",
             ]
    names_remove = ["J/A+A/675/A195/mast",
                    "I/358/varisum",
                    "B/vsx/vsx",
                    "J/A+A/674/A22/catalog",
                    "J/A+A/648/A44/tabled1",
                    "J/MNRAS/433/3398/table2",
                    "J/MNRAS/503/3975/catalog",
                    "J/ApJS/259/11/table4",
                    "J/ApJS/209/27/xmystix",
                    "J/ApJS/209/31",
                    "J/ApJS/247/66/table8",
                    "I/361/lenscand",
                    "I/356/galcand",
                    ]
    
    tasks = []
    remove = []
    batch = 2000
    for num in range(0,len(data.file),batch):
        for name in names_remove:
            remove.append((name,num))
    if para:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {executor.submit(retry(data.check_if_exist), tup[0], tup[1]): tup for tup in remove}

        data.remove() 

        for num in range(0,len(data.file),batch):
            for name in names:
                tasks.append((name,num))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {executor.submit(retry(data.get_photo), tup[0], tup[1]): tup for tup in tasks}
    else:
        for task in remove:
            retry(data.check_if_exist)(task[0],task[1])

        data.remove()

        for num in range(0,len(data.file),batch):
            for name in names:
                tasks.append((name,num))

        for task in tasks:
            data.get_photo(task[0],task[1])
    data.save("examples/")

def check_cluster(name_cluster,data_csv,para = False,num_p = 50):
    data_csv = pd.read_csv(data_csv,delim_whitespace=True)
    ra_list = data_csv["ra"].values
    dec_list = data_csv["dec"].values
    id = data_csv["memberprob"].values > 0.90
    ra_list = ra_list[id]
    dec_list = dec_list[id]
    data_csv = data_csv.iloc[id]
    print("number of stars: ",len(ra_list))
    del Galactic["II/319/las9"]
    catalog = Galactic | M5
    data = DataBundle(name_cluster,catalog,leng = len(ra_list),bands = Bands_def_all_short)
    data.dis_norm = 0.5
    print("norm: ",data.dis_norm)
    for i in tqdm(range(len(ra_list))):
        data.new_row(i,
                     data_csv["source_id"].values[i],
                     data_csv["ra"].values[i],
                     data_csv["dec"].values[i],
                     data_csv["plx"].values[i],
                     data_csv["g_mag"].values[i],
                     0,
                     [-99.,
                     -99.,
                     -99.,
                     -99.,
                     -99.,
                     -99.,
                     -99.,
                     -99.])

    names = ["I/355/gaiadr3",
             #"II/246/out",
             "II/335/galex_ais",
             #"II/312/ais",
             "II/312/mis",
             "II/349/ps1",
             "II/365/catwise",
             "II/379/smssdr4",
             "II/336/apass9",
             "V/154/sdss16",
             "J/A+A/632/A56/catalog",
             "II/319/las9"
             #"II/311/wise",
             ]
    names_remove = ["J/A+A/675/A195/mast",
                    "I/358/varisum",
                    "B/vsx/vsx",
                    "J/A+A/674/A22/catalog",
                    "J/A+A/648/A44/tabled1",
                    "J/MNRAS/433/3398/table2",
                    "J/MNRAS/503/3975/catalog",
                    "J/ApJS/259/11/table4",
                    "J/ApJS/209/27/xmystix",
                    "J/ApJS/209/31",
                    ]
    tasks = []
    remove = []
    batch = 2000
    for num in range(0,len(data.file),batch):
        for name in names_remove:
            remove.append((name,num))
    if para:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {executor.submit(data.check_if_exist, tup[0], tup[1]): tup for tup in remove}

        data.remove() 

        for num in range(0,len(data.file),batch):
            for name in names:
                tasks.append((name,num))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {executor.submit(data.get_photo, tup[0], tup[1]): tup for tup in tasks}
    else:
        for task in remove:
            data.check_if_exist(task[0],task[1])

        data.remove()

        for num in range(0,len(data.file),batch):
            for name in names:
                tasks.append((name,num))

        for task in tasks:
            data.get_photo(task[0],task[1])
    data.save("examples/")
    print(data.file)

def RMSE(X,Y):
    assert len(X)==len(Y)
    out = (X-Y)**2
    out = out[np.logical_not(np.isnan(out))]
    return np.sqrt(np.mean(out))

def MAD(X,Y):
    assert len(X)==len(Y)
    out = np.abs(X-Y)
    out = out[np.logical_not(np.isnan(out))]
    return np.mean(out)


@torch.no_grad()
def evaluate_dataset(name_model,name,directory,B = 256,scale = 1,eta = 0,low = 0,how_many = 64,IS = False):
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    print("name: ",name)
    print("weights: ",name_model)
    data = DataBundle(name,B = B)
    data.load(directory)
    print("len: ",len(data.file))
    with open("SED_NDE_{}.json".format(name_model),"r") as f:
        params = json.loads(f.read())
    model = SED_NDE(params["MAF"],params["TF"])
    model.load_state_dict(torch.load("SED_NDE_{}.tc".format(name_model),weights_only = True))
    dev = "cuda"
    model.to(dev)
    model.eval()

    def helper_function(out,index,*args):
        if len(args[0]) == 0:
            return
        output = generator.log_prob(*args)
        out[index] = output

    if IS:
        generator = PhotometryGenerator(Bands_def_all_short)

    for bands,mags,errors,id in tqdm(data,"sampling"):
        bands = bands.T.int().to(dev)
        mags = mags.T.float().to(dev)
        errors_new = torch.clip(errors,low,torch.inf)
        errors_new = torch.sqrt(torch.clip(errors_new.T.float().to(dev),0,torch.inf)**2*scale**2 + eta**2)
        with torch.autocast("cuda",torch.float16,enabled = params["MAF"]["autocast"]):
            samples,log_probs  = model.sample(how_many,bands,mags,errors_new,return_log_prob = True)
            log_probs = log_probs.cpu().numpy()
        if IS:
            cores = IS
            band_size = bands.shape[0] #number of unique bands
            samples_batch = np.array_split(samples.cpu().numpy(),cores)
            bands_batch = np.array_split(bands.T.cpu().numpy(),cores)
            mags_batch = np.array_split(mags.T.cpu().numpy(),cores)
            errors_new_batch = np.array_split(errors_new.T.cpu().numpy(),cores)

            processes = []
            man = mp.Manager()
            shared = man.list([None] * cores)  # Shared list to store ordered results
            for i in range(cores):
                p = mp.Process(target = helper_function,args = (shared,i,samples_batch[i],bands_batch[i],mags_batch[i],errors_new_batch[i]))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            shared = [x for x in shared if x is not None]
            log_prob_prim = np.concatenate(shared,axis = 0) 

            weights = scipy.special.softmax(log_prob_prim-log_probs,axis = 1)
            samples = samples.cpu().numpy()
            samples[:,:,0] *= 1000
            best = np.sum(weights[:,:,np.newaxis]* samples,axis = 1)#/np.sum(weights,axis = 1)
        else:
            samples = samples.cpu().numpy()
            samples[:,:,0] *= 1000
            best = np.mean(samples,axis = 1)
        log_probs = np.mean(log_probs,axis = -1)
        data.upload(id,samples,best,log_probs)
    data.save(directory,save_hp = False)
    RMSE_temp = RMSE(data.file["T_true"].values,data.file["T_50th"].values)
    RMSE_temp_best = RMSE(data.file["T_true"].values,data.file["T_best"].values)
    RMSE_logg = RMSE(data.file["logg_true"].values,data.file["logg_50th"].values)
    RMSE_metal = RMSE(data.file["[M/H]_true"].values,data.file["[M/H]_50th"].values)
    RMSE_logg_best = RMSE(data.file["logg_true"].values,data.file["logg_best"].values)
    RMSE_metal_best = RMSE(data.file["[M/H]_true"].values,data.file["[M/H]_best"].values)
    print("temp err: ",RMSE_temp,RMSE_temp_best)
    print("metal err: ",RMSE_metal,RMSE_metal_best)
    print("logg err: ",RMSE_logg,RMSE_logg_best)
    MAD_temp = MAD(data.file["T_true"].values,data.file["T_50th"].values)
    MAD_temp_best = MAD(data.file["T_true"].values,data.file["T_best"].values)
    MAD_logg = MAD(data.file["logg_true"].values,data.file["logg_50th"].values)
    MAD_metal = MAD(data.file["[M/H]_true"].values,data.file["[M/H]_50th"].values)
    MAD_logg_best = MAD(data.file["logg_true"].values,data.file["logg_best"].values)
    MAD_metal_best = MAD(data.file["[M/H]_true"].values,data.file["[M/H]_best"].values)
    print("MAD temp: ",MAD_temp,MAD_temp_best)
    print("MAD metal: ",MAD_metal,MAD_metal_best)
    print("MAD logg: ",MAD_logg,MAD_logg_best)
    RMSE_arr = np.array([RMSE_temp,RMSE_metal,RMSE_logg])
    return RMSE_arr


def optimize(names,**kwargs):
    norm = np.array([300,1,1])
    def fun(X):
        print(X)
        eta,scale,low = X
        value = 0
        for name in names:
            value += np.sum(evaluate_dataset(name = name,eta = eta,low = low,scale = scale,**kwargs)/norm)
        return value
    start = np.array([0.005,1.3,0.005]) 
    out = scipy.optimize.minimize(fun,x0 = start,method = "Nelder-Mead")
    print(out.x)

@torch.no_grad()
def process_cluster(name_model,name,directory,B = 256,scale = 1,eta = 0,low = 0,how_many = 64,IS = False):
    print("name: ",name)
    print("weights: ",name_model)
    data = DataBundle(name,B = B)
    data.load(directory)
    print("len: ",len(data.file))
    with open("SED_NDE_{}.json".format(name_model),"r") as f:
        params = json.loads(f.read())
    model = SED_NDE(params["MAF"],params["TF"])
    model.load_state_dict(torch.load("SED_NDE_{}.tc".format(name_model),weights_only = True))
    dev = "cuda"
    model.to(dev)
    model.eval()
    if IS:
        generator = PhotometryGenerator(Bands_def_all_short)

    for bands,mags,errors,id in tqdm(data,"sampling"):
        bands = bands.T.int().to(dev)
        mags = mags.T.float().to(dev)
        #print(mags.shape,bands.shape)
        errors_new = torch.clip(errors,low,torch.inf)
        errors_new = torch.sqrt(torch.clip(errors_new.T.float().to(dev),0,torch.inf)**2*scale**2 + eta**2)
        with torch.autocast("cuda",torch.float16,enabled = params["MAF"]["autocast"]):
            samples,log_probs  = model.sample(how_many,bands,mags,errors_new,return_log_prob = True)
            log_probs = log_probs.cpu().numpy()
        if IS:
            log_prob_prim = 5* generator.log_prob(samples.cpu().numpy(),bands.T.cpu().numpy(),mags.T.cpu().numpy(),errors_new.T.cpu().numpy())
            weights = scipy.special.softmax(log_prob_prim-log_probs,axis = 1)
            #weights = np.exp(log_prob_prim - log_probs)
            samples = samples.cpu().numpy()
            samples[:,:,0] *= 1000
            best = np.sum(weights[:,:,np.newaxis]* samples,axis = 1)#/np.sum(weights,axis = 1)
        else:
            samples = samples.cpu().numpy()
            samples[:,:,0] *= 1000
            best = np.mean(samples,axis = 1)
        log_probs = np.mean(log_probs,axis = -1)
        data.upload(id,samples,best,log_probs)
    data.save(directory,save_hp = False)
    idx = np.logical_and(data.file["Entropy"].values > -np.inf,data.file["leng"].values > 10)
    MH = data.file["[M/H]_50th"].values[idx]
    sigma = (-data.file["[M/H]_16th"].values[idx] + data.file["[M/H]_84th"].values[idx])/2
    th = 2
    while True:
        sigma_prim = 1/np.sqrt(np.sum(1/sigma**2))
        mu_prim = np.sum(MH/sigma**2)*sigma_prim**2
        idx = (MH - mu_prim) < th * np.sqrt(sigma_prim**2 + sigma**2)
        if np.sum(idx) == len(idx):
            break
        else:
            MH = MH[idx]
            sigma = sigma[idx]
            print(mu_prim,sigma_prim)

    print("MH = {:.2f}+-{:.2f}, {:.0f}".format(mu_prim,sigma_prim,np.sum(idx)))


def plot_tokens(name,weights):
    with open(name,"r") as f:
        params = json.loads(f.read())
    model = SEDTransformer(params["bands"],params["d_model"],params["heads"],params["hidden"],params["dropout"],params["layers"])
    model.load_state_dict(torch.load(weights))
    data = model.embedding.weight.detach().numpy()
    transformation = PCA(2)#TSNE(perplexity=2)
    generate = PhotometryGenerator(Bands_def_all_short)
    print(data.shape)
    new_data = transformation.fit_transform(data)
    fig,ax=plt.subplots()
    ax.scatter(new_data[:,0],new_data[:,1])
    plt.savefig("PCA.png")

if __name__=="__main__":
    torch.manual_seed(42)
    n = 400000
    batch = 5000
    scale = 1.
    eta = 0.01
    low = 0.01
    para = False
    name = "medium_BOSZ_Z_mix"
    IS = 32
    name_server = "vizier.cds.unistra.fr"
    how_many = 16
    check_apogee(n,iner = True,para = para, name_server = name_server, batch = batch)
    check_apogee(n,iner = False,para = para, name_server = name_server, batch = batch)
    #check_lamost(n,para = para, AFGK = True)
    #check_lamost(n,para = para, AFGK = False)
    #check_cluster("M5","examples/NGC_5904_M_5.txt",para = para)

    #process_cluster(name,"M5","examples/",scale = scale,eta = eta,low = low,IS = IS,how_many = how_many)
    #check_cluster("M5","examples/NGC_5904_M_5.txt",60)
    #evaluate_dataset(name,"APOGEE_disc","examples/",scale = scale,eta = eta,low = low,IS = IS,how_many = how_many)
    #evaluate_dataset(name,"APOGEE_halo","examples/",scale = scale,eta = eta,low = low,IS = IS,how_many = how_many)

    #test_NDE(name,244.196463, 22.013308,scale = scale,eta = eta,IS = IS,MCMC = True)

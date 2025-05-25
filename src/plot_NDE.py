import json
import numpy as np
import torch
import pandas as pd
from models import SEDTransformer,SED_NDE
from utils import PhotometryGenerator,Bands_def_all_short
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
from Iris import Star,Galactic
from sklearn.decomposition import PCA
import corner
import emcee
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy
from chainconsumer import Chain, Truth,ChainConsumer, PlotConfig,ChainConfig
from chainconsumer.plotting import plot_contour, plot_truths
import pystellibs
import argparse

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
def test_NDE(name,ra,dec,scale = 1,eta = 0.0,IS = False,MCMC = False,cuda = True):
    with open("SED_NDE_{}.json".format(name),"r") as f:
        final = json.loads(f.read())
    model = SED_NDE(final["MAF"],final["TF"])
    model.load_state_dict(torch.load("SED_NDE_{}.tc".format(name)))
    dev = "cuda" if cuda else "cpu"
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

if __name__ == "__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description='Plot the NDE')
    parser.add_argument('--name', type=str, help='Name of the model',required=True)
    parser.add_argument("--how_many",type = int,default = 64,help = "How many samples to take")
    parser.add_argument("--scale",type = float,default = 1.0,help = "Scale of the errors")
    parser.add_argument("--eta",type = float,default = 0.0,help = "error in quadrature")
    parser.add_argument("--low",type = float,default = 0.0,help = "lower bound of the error")
    parser.add_argument("--cuda",type = int,default = 1,help = "Use cuda")
    parser.add_argument("--IS",type = int,default = 1,help = "Use importance sampling")
    parser.add_argument("--MCMC",type = int,default = 0,help = "Use MCMC")
    args = parser.parse_args()
    test_NDE(args.name,70.401283,25.035074,scale = args.scale,eta = args.eta,MCMC = args.MCMC,IS = args.IS,cuda = args.cuda)

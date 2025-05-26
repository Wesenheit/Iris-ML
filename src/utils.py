import pyphot
from tqdm import tqdm
from astroquery.vizier import Vizier
import astropy.coordinates as coords
from astropy import units as u
from astropy.table import QTable
from pyphot import (unit, Filter)
from pathlib import Path
import numpy as np
from pystellibs import BaSeL,BTSettl,Kurucz,Phoenix,Bosz
import extinction
from torch.utils.data import Dataset,DataLoader
import torch
from scipy.interpolate import CloughTocher2DInterpolator
from torch import optim as optim
from Iris import Star
import pandas as pd
import h5py
import scipy
from scipy.stats import uniform, norm
from concurrent.futures import ProcessPoolExecutor
import os

def retry(fun,maxi = 10):
    def inner(*args,**kwargs):
        c = 0
        while c < maxi:
            try:
                out = fun(*args,**kwargs)
                return out
            except Exception as e:
                print(f"ERROR! {e}")
                c += 1
        print("Maximum number")
        out = None
        return out
    return inner

def is_set(bit,mask):
    return (mask & 2**bit) != 0

def check_bit(mask,bits):
    for bit in bits:
        bit_n = 1 << bit
        if not (mask & bit_n) == 0:
            return False
    return True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

M5 = {
    "V/154/sdss16":[["umag","e_umag","gmag","e_gmag","rmag","e_rmag","imag","e_imag","zmag","e_zmag"],
                    ["SDSS_u","SDSS_g","SDSS_r","SDSS_i","SDSS_z"],0.5,[True,True,True,True,True]],

    "J/A+A/632/A56/catalog":[["Jmag","e_Jmag","Hmag","e_Hmag","Ksmag","e_Ksmag"],["2MASS_J","2MASS_H","2MASS_Ks"],0.5,[True,True,True]],
    "II/319/las9":[["Jmag1","e_Jmag1","Hmag","e_Hmag","Kmag","e_Kmag"],["2MASS_J","2MASS_H","2MASS_Ks"],0.5,[True,True,True]],
    }

Bands_def_all_short = [
    "GAIA_DR3_G",
    "GAIA_DR3_BP",
    "GAIA_DR3_RP",
    "GALEX_NUV",
#    "GALEX_FUV",
    "PS1_g",
    "PS1_i",
    "PS1_z",
    "PS1_r",
    "PS1_y",
    "SDSS_u",
    "SDSS_g",
    "SDSS_r",
    "SDSS_i",
    "SDSS_z",
    "SkyMapper_u",
    "SkyMapper_v",
    "SkyMapper_r",
    "SkyMapper_i",
    "SkyMapper_g",
    "SkyMapper_z",
    "2MASS_H",
    "2MASS_J",
    "2MASS_Ks",
#    "H_UKIDSS",
#    "J_UKIDSS",
#    "Y_UKIDSS",
#    "K_UKIDSS",
#    "Z_UKIDSS",
#    "SPITZER_IRAC_36",
#    "SPITZER_IRAC_45",
    "WISE_RSR_W1",
    "WISE_RSR_W2",
    ]

kpc=3.08567758*10**21

class PhotometryGenerator(Star):
    def __init__(self,bands) -> None:
        super().__init__("test",0,0)
        self.gaia = self.OTHER["GAIA_DR3_G"]
        self.bands = bands
        self.filters = bands

        self.p_core = 0.0
        self.p_pop2 = 0.25
        self.AV_core_mean = 5
        self.AV_core_std = 3
        self.AV_exp = 1.5
        self.RV_mean = 3.13
        self.RV_std = 0.18
        self.lib_stell = Bosz(2000,1,interpolator = "ndlinear")
        self.ext = extinction.fitzpatrick99
        for i in self.filters:
            try:
                f = self.lib_phot[i]
            except:
                f = self.OTHER[i]
            if i in self.AB:
                self.zerop.append(f.AB_zero_flux.magnitude)
            else:
                self.zerop.append(f.Vega_zero_flux.magnitude)
        self.fil_obj=[]
        for i in range(len(self.filters)):
            try:
                self.fil_obj.append(self.lib_phot[self.filters[i]])
            except:
                self.fil_obj.append(self.OTHER[self.filters[i]])
        self.zerop = np.array(self.zerop)

        if "basel" in str(type(self.lib_stell)):
            FEH_range = np.linspace(-3.5,1,10)
            target_logZ = np.unique(self.lib_stell.logZ)
            self.sol = np.polyfit(FEH_range,target_logZ,deg = 1)
        elif "phoenix" or "bosz" in str(type(self.lib_stell)):
            self.sol = 1
        elif "kurucz" in str(type(self.lib_stell)):
            self.sol = 0.02
        else:
            self.sol = 0.013399998792087208

    def to_Z(self,FEH):
        if "basel" in str(type(self.lib_stell)):
            logZ = self.sol[0]*FEH + self.sol[1]
            return 10**logZ
        else:
            return 10**FEH * self.sol

    def __len__(self):
        return len(self.bands)
    
    def get_boundaries(self,g,n = 1000):
        logt_arr = np.linspace(0,5.5,n)
        g_arr = np.ones_like(logt_arr)*g
        arr = np.stack((logt_arr,g_arr))
        is_in = self.lib_stell.points_inside(arr.T)
        t_min = np.min(logt_arr[is_in])
        t_max = np.max(logt_arr[is_in])
        return t_min,t_max
    
    def generate_data(self,T,g,Z,AV,RV,id):
        spectrum = self.lib_stell.generate_stellar_spectrum(T,g,0,Z)
        spectrum = np.clip(spectrum,0,np.inf)
        idx = np.isnan(spectrum)
        mag = np.zeros([len(id),])
        if idx.any():
            print(self.lib_stell.wavelength.magnitude[idx])
            print(T,g,Z)
            return mag
            #raise ValueError
        ext = self.ext(self.lib_stell.wavelength.magnitude,a_v = AV,r_v = RV)
        spectrum = 10**(-0.4*ext)*spectrum
        wave = self.lib_stell.wavelength.magnitude*unit['AA']
        spectrum = np.array(spectrum)*unit['flam']
        for j,i in enumerate(id):
            flux = self.fil_obj[i].get_flux(wave,spectrum)/(4*np.pi*(10*kpc)**2)
            try:
                flux = flux.value
            except:
                pass
            if flux <= 0:
                print(self.bands[i],T,g,Z,AV,RV)
                raise ValueError
            mag[j] = -2.5*np.log10(flux/self.zerop[i])
        return mag



    def log_prob(self,samples,id,mag,err):
        """
        Values in format Batch, Different Samples, dim
        """

        T = samples[:,:,0].flatten()
        logg = samples[:,:,1].flatten()
        FEH = samples[:,:,2].flatten()
        AV = samples[:,:,3].flatten()
        RV = samples[:,:,4].flatten()


        #different priors
        log_prob_AV = - self.p_core*(self.AV_core_mean-AV)**2/(2*self.AV_core_std**2) - (1-self.p_core)*(np.log(self.AV_exp) + self.AV_exp*(AV-0.03))
        log_prob_RV = - (self.RV_mean-RV)**2/(2*self.RV_std**2)
        log_prob_MEH = - self.p_core * FEH**2/(2*0.2**2) - (1-self.p_core) * (self.p_pop2 * (-1.49-FEH)**2/(2*0.4**2) + (1-self.p_pop2)*(-0.1-FEH)**2/(2*0.2**2))
        log_prob_logg = 0
         
        assert samples.shape[0]==id.shape[0]
        unique = np.unique(id)
        AV = np.clip(AV,0,np.inf)
        dictionary = {}

        dictionary["logT"] = np.log10(1000*T)#np.clip(np.log10(T*1000),self.lib_stell.log_T_min,self.lib_stell.log_T_max)
        dictionary["logg"] = logg
        dictionary["logL"] = np.zeros_like(logg)
        dictionary["Z"] = self.to_Z(FEH)
        tab = QTable(dictionary)
        stell = self.lib_stell.generate_individual_spectra(tab)[1].magnitude.T
        ext = np.zeros([stell.shape[0],AV.shape[0]])
        for i in range(AV.shape[0]):
            ext[:,i] = self.ext(self.lib_stell.wavelength.magnitude,AV[i],RV[i])
        stell = np.power(10,-0.4 * ext) * stell
        stell = stell.T
        if samples.ndim < 3:
            pred = np.zeros([samples.shape[0],len(self.filters)])
        else:
            pred = np.zeros([T.shape[0],len(self.filters)])
        
        for i in range(len(unique)):
            val = (self.fil_obj[unique[i]].get_flux(self.lib_stell.wavelength.magnitude*unit['AA'],stell*unit['flam'],axis = 1))/(4*np.pi*(10*kpc)**2)
            try:
                pred[:,i] = val.value / self.zerop[unique[i]]
            except AttributeError:
                pred[:,i] = val / self.zerop[unique[i]]
        
        pred = -2.5 * np.log10(pred)
        pred_temp = pred.reshape(samples.shape[0],samples.shape[1],-1)
        sel_pred = np.zeros([samples.shape[0],samples.shape[1],mag.shape[1]])
        for i in range(id.shape[0]):
            for j in range(len(id[i])):
                idx = np.nonzero((id[i,j] == unique))[0].item()
                sel_pred[i,:,j] = pred_temp[i,:,idx]

        mag_broad = mag[:,np.newaxis,:].repeat(samples.shape[1],axis = 1) #mag with the shape boradcasted to the size of samples
        #sel_pred = np.where(np.isnan(sel_pred),mag_broad,sel_pred)
        if (np.isnan(mag_broad - sel_pred).all()):
            raise ValueError
        factor = np.nanmean(mag_broad - sel_pred,axis = -1,keepdims = True)
        return_val = -np.sum((sel_pred + factor - mag_broad)**2 / (2 * err[:,np.newaxis,:].repeat(samples.shape[1],axis = 1)**2),axis = -1)
        #print(return_val[0,1:100]) 
        return_val = np.where(np.isnan(return_val),-np.inf,return_val) + (log_prob_AV + log_prob_RV + log_prob_MEH + log_prob_logg).reshape(return_val.shape)
        return return_val

class BandDataset(Dataset):

    def __init__(self,data_dir,N,noise = False) -> None:
        super().__init__()
        self.dir = data_dir
        self.len = N
        self.noise = noise
        self.prob = 0.0
        self.prob_gaia = 0.0
        self.scale = 0.25
        self.th = 0.00
        self.master = {}
        for index in range(N):
            mags = np.load(os.path.join(self.dir,"batch_{}_X.npy".format(index)))
            parameters = np.load(os.path.join(self.dir,"batch_{}_Y.npy".format(index)))
            what = np.load(os.path.join(self.dir,"batch_{}_bands.npy".format(index)))
            self.master[index] = (mags,parameters,what)
    def __len__(self):
        return self.len 

    def __getitem__(self, index):
        mags,parameters,what = self.master[index]
        mags_torch,parameters_torch,what_torch = torch.from_numpy(mags),torch.from_numpy(parameters),torch.from_numpy(what)
        if self.noise:
            how_many = mags.flatten().shape[0]
            scale = scipy.stats.truncnorm(loc = 0.01,scale = 0.04, a = 0.000,b = 0.2).rvs(how_many).reshape(mags.shape)
            errors = torch.from_numpy(scale)
            fluxes = 10**(0.4*(22-mags_torch))
            fluxes_err = fluxes * 0.4*np.log(10) * errors
            mags_torch = 22- 2.5*torch.log10(fluxes + torch.randn(fluxes.shape)*fluxes_err)
            return mags_torch,errors,parameters_torch,what_torch
        else:
            return mags_torch,parameters_torch,what_torch
    

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class DataBundle():
    def __init__(self,name,catalog = None,leng = 0,bands = [0,0,0],B = 128):
        self.name = name
        self.main = {}
        self.B = B
        dic = {}
        dic["DR3_id"] = np.zeros([leng,],dtype = np.int64)
        dic["RA"] = np.zeros([leng,],dtype = np.float64)
        dic["DEC"] = np.zeros([leng,],dtype = np.float64)
        dic["plx"] = np.zeros([leng,],dtype = np.float64)
        dic["Mag_G"] = np.zeros([leng,],dtype = np.float64)
        dic["leng"] = np.zeros([leng,],dtype = np.int64)
        dic["Entropy"] = np.zeros([leng,],dtype = np.float64)
        dic["T_best"] = np.zeros([leng,],dtype = np.float64)
        dic["logg_best"] = np.zeros([leng,],dtype = np.float64)
        dic["[M/H]_best"] = np.zeros([leng,],dtype = np.float64)
        dic["AV_best"] = np.zeros([leng,],dtype = np.float64)
        dic["RV_best"] = np.zeros([leng,],dtype = np.float64)
        dic["T_16th"] = np.zeros([leng,],dtype = np.float64)
        dic["T_50th"] = np.zeros([leng,],dtype = np.float64)
        dic["T_84th"] = np.zeros([leng,],dtype = np.float64) 
        dic["logg_16th"] = np.zeros([leng,],dtype = np.float64)
        dic["logg_50th"] = np.zeros([leng,],dtype = np.float64)
        dic["logg_84th"] = np.zeros([leng,],dtype = np.float64)
        dic["[M/H]_16th"] = np.zeros([leng,],dtype = np.float64)
        dic["[M/H]_50th"] = np.zeros([leng,],dtype = np.float64)
        dic["[M/H]_84th"] = np.zeros([leng,],dtype = np.float64)
        dic["AV_16th"] = np.zeros([leng,],dtype = np.float64)
        dic["AV_50th"] = np.zeros([leng,],dtype = np.float64)
        dic["AV_84th"] = np.zeros([leng,],dtype = np.float64)
        dic["RV_16th"] = np.zeros([leng,],dtype = np.float64)
        dic["RV_50th"] = np.zeros([leng,],dtype = np.float64)
        dic["RV_84th"] = np.zeros([leng,],dtype = np.float64)

        dic["T_true"] = np.zeros([leng,],dtype = np.float64)
        dic["T_err"] = np.zeros([leng,],dtype = np.float64)
        dic["[a/H]"] = np.zeros([leng,],dtype = np.float64)
        dic["Ak"] = np.zeros([leng,],dtype = np.float64)
        dic["logg_true"] = np.zeros([leng,],dtype = np.float64)
        dic["logg_err"] = np.zeros([leng,],dtype = np.float64)
        dic["[M/H]_true"] = np.zeros([leng,],dtype = np.float64)
        dic["[M/H]_err"] = np.zeros([leng,],dtype = np.float64)
        self.file = pd.DataFrame(data = dic).astype(np.float64)
        self.file["DR3_id"]  = self.file["DR3_id"].astype(np.int64)
        self.data = np.zeros([leng,len(bands),2]) - 1
        self.bands = np.array(bands)
        self.catalogs = catalog
        self.keep = np.ones([leng]).astype(bool)
        self.dis_norm = 1
        self.con ={
        "J_UKIDSS":"2MASS_J",
        "H_UKIDSS":"2MASS_H",
        "K_UKIDSS":"2MASS_Ks",
        "J_2MASS":"2MASS_J",
        "H_2MASS":"2MASS_H",
        "K_2MASS":"2MASS_Ks",
        "J":"GROUND_BESSELL_J",
        "H":"GROUND_BESSELL_H",
        "K":"GROUND_BESSELL_K",
        "3.6":"SPITZER_IRAC_36",
        "4.5":"SPITZER_IRAC_45",
        "5.8":"SPITZER_IRAC_58",
        "8.0":"SPITZER_IRAC_80",
        "I":"GROUND_COUSINS_I",
        "V":"GROUND_JOHNSON_V",
        "g_SM":"SkyMapper_g",
        "r_SM":"SkyMapper_r",
        "i_SM":"SkyMapper_i",
        "z_SM":"SkyMapper_z",
        "u_SM":"SkyMapper_u",
        "v_SM":"SkyMapper_v",
        "B":"GROUND_JOHNSON_B",
        "U":"GROUND_JOHNSON_U",
        "G_Gaia":"GaiaDR2_G",
        "RP_Gaia":"GaiaDR2_RP",
        "BP_Gaia":"GaiaDR2_BP",
        "W1":"WISE_RSR_W1",
        "W2":"WISE_RSR_W2",
        "W3":"WISE_RSR_W3",
        "W4":"WISE_RSR_W4",
        "R":"GROUND_COUSINS_R",
        "g_PS1":"PS1_g",
        "i_PS1":"PS1_i",
        "z_PS1":"PS1_z",
        "r_PS1":"PS1_r",
        "y_PS1":"PS1_y",
        "T_TESS":"TESS",
        "FUV":"GALEX_FUV",
        "NUV":"GALEX_NUV"
        }
        self.unique_remove = {}
        self.unique_photo = {}
    def save(self,directionary,save_hp = True,th = 6):
        if save_hp:
            for j in tqdm(range(len(self.file))):
                for i in range(self.data.shape[1]):
                    if (self.data[j,i,0] > 0 and self.data[j,i,1] <= 0) or self.data[j,i,1] > 0.1:
                        self.data[j,i,0] = -1
                #leng = np.int64(np.sum(self.data[j,:,0]>0))
                self.file.loc[j,"leng"] = np.int64(np.sum(self.data[j,:,0]>0))
            self.file["leng"] = self.file["leng"].astype(np.int64)
            idx = self.file["leng"].values > th
            self.file = self.file.iloc[idx,:].reset_index(drop = True)
            self.data = self.data[idx]
            #self.file["Mag_G"] = self.data[:,0,0]
            #self.file["Mag_G_err"] = self.data[:,0,1]
            print("len: ",len(self.data))
        self.file.to_csv(directionary + self.name + ".csv",index = False)
        if not save_hp:
            return
        np.save(directionary + self.name + ".npy",self.data)

    def convert(self,name):
        out = (self.bands == name)
        if len(np.nonzero(out)[0]) > 0:
            return np.nonzero(out)[0].item()
        else:
            return -1

    def get_photo(self,name,num,batch,**kwargs):
        idx = np.logical_and(np.arange(len(self.file)) >= num, np.arange(0,len(self.file)) < (num + batch))
        if self.catalogs == None:
            print("No catalog found!")
            return
        if name not in self.unique_photo:
                self.unique_photo[name] = np.ones([len(self.file),]).astype(bool)
        if "ps1" in name:
            fil = {}
            columns = self.catalogs[name][0] + ["gMeanPSFMagNpt","rMeanPSFMagNpt","iMeanPSFMagNpt","zMeanPSFMagNpt","yMeanPSFMagNpt"]
        elif "sdss" in name:
            columns = self.catalogs[name][0] + ["flags_u","flags_g","flags_r","flags_i","flags_z"]
            fil = {"mode":"=1","class":"=6"}
        elif "smss" in name:
            columns = self.catalogs[name][0] + ["u_mmvar","g_mmvar","r_mmvar","i_mmvar","z_mmvar"]
            fil = {}
        else:
            columns = self.catalogs[name][0]
            fil = {}
        v = Vizier(row_limit = -1,columns = columns,column_filters = fil,timeout = 86400,**kwargs)
        count = 0
        while count < 10:
            try:
                result = v.query_region(coords.SkyCoord(ra = self.file["RA"].values[idx], dec = self.file["DEC"].values[idx],
                                        unit = (u.deg, u.deg),
                                        frame = 'icrs'),
                                        catalog = name,
                                        radius = 2*u.arcsec*self.dis_norm)
                                        #radius = self.catalogs[name][2]*u.arcsec*self.dis_norm)
            

            except:
                count += 1
                print("connection error! " +count)
            else:
                break
        if len(result)>0:
            file = result[0]
            temporary_d = {}
            offset = np.min(np.nonzero(idx)[0])
            for j in range(len(file)):
                id = file[j][0]-1 + offset
                arr = np.array(file[j][1:])
                idx_p = np.logical_and(arr > 1,np.logical_not(np.isnan(arr)))
                if np.sum(idx_p) == 0:
                    mean = 100
                else:
                    mean = np.mean(arr[idx_p])
                if id not in temporary_d:
                    temporary_d[id] = (mean,j)
                else:
                    temporary_d[id] = (-1,j)
                    """
                    if temporary_d[id][0] is None:
                        print("Blending!")
                        pass
                    if np.abs(temporary_d[id][0] - mean) < 2 :
                        temporary_d[id] = (None,None)
                    elif temporary_d[id][0] > mean:
                        temporary_d[id] = (mean,j)
                    """
            for j in range(len(file)):      
                for i in range(len(self.catalogs[name][1])):
                    if str(file[j][2*i+1]) == "--":
                        pass
#                        print("Something missing in {}!".format(name))
                    else:
                        err = 0 if str(file[j][2*i+2]) == "--" else float(file[j][2*i+2])
                        ampl = float(file[j][2*i+1])
                        name_test = (self.catalogs[name][1])[i]
                        name_new = self.con[name_test] if name_test in self.con else name_test
                        id_band = self.convert(name_new)
                        id = file[j][0] - 1 + offset
                        if "ps1" in name:
                            if file[j][11 + i] > 3:
                                #print(file[j][10 + i])
                                add_flag = True
                            else:
                                add_flag = False
                        elif "sdss" in name:
                            if i == 0:
                                ampl -= 0.04
                            elif i == 4:
                                ampl += 0.02
                            #if check_bit(file[j][11 + i],[5,19,15,8,12,]):
                            #if check_bit(file[j][11 + i],[5,19,15,40]) and is_set(28,file[j][11+i]):
                            #https://skyserver.sdss.org/dr16/en/help/docs/realquery.aspx#cleanStars
                            if check_bit(file[j][11 + i],[2,7,5,19,47,18,40,46,44,12]) and is_set(file[j][11+i],28):
                                add_flag = True
                            else:
                                add_flag = False
                        elif "smss" in name:
                            if file[j][11+i] < 0.1:
                                add_flag = True
                            else:
                                add_flag = False
                        else:
                            add_flag = True
                        if id_band >= 0 and temporary_d[id][1] == j and add_flag:
                            if self.data[id,id_band,0] < 0:
                                self.data[id,id_band,0] = ampl
                                self.data[id,id_band,1] = err
                        if temporary_d[id][1] == j and temporary_d[id][0] < 0:
                            for num_band in range(self.data.shape[1]):
                                self.data[id,num_band,0] = -1
                                    
        else:
            print("ERROR!",name,num)
        print(name,num/len(self.file))
        #if lock is not None:
        #    lock.release()

    def check_if_exist(self,name,num,batch,**kwargs):
        idx = np.logical_and(np.arange(len(self.file)) >= num, np.arange(0,len(self.file)) < (num + batch))
        if self.catalogs == None:
            print("No catalog found!")
            return
        v = Vizier(row_limit = -1,timeout = 86400,**kwargs)
        if name not in self.unique_remove:
            self.unique_remove[name] = np.ones([len(self.file),]).astype(bool)
        result = v.query_region(coords.SkyCoord(ra = self.file["RA"].values[idx], dec = self.file["DEC"].values[idx],
                                        unit = (u.deg, u.deg),
                                        frame = 'icrs'),
                                        catalog = name,
                                        radius = 2*u.arcsec*self.dis_norm)
        try:
        #if True:
            file = result[0]
            offset = np.min(np.nonzero(idx)[0])
            for j in range(len(file)):
                id = file[j][0] -1 + offset
                self.unique_remove[name][id] = False
        except:
            print("ERROR!",name,num)
        print(name,num/len(self.file))

    def load(self,dire,remove = None):
        self.file = pd.read_csv(dire + self.name + ".csv")
        self.data = np.load(dire + self.name + ".npy")
        ####
        #remove = [23,24]
        if remove is not None:
            for r in remove:
                print("remove: ",r)
                for i in range(len(self.data)):
                    if self.data[i,r,1] > 0:
                        self.file.iloc[i,5] = self.file.iloc[i,5] - 1
                        self.data[i,r,:] = -1.
        ####

    def __iter__(self):
        self.max = np.max(self.file.iloc[:,5])
        #print("maximal leng: ",self.max)
        self.id = []
        self.iterator_list = [ [] for _ in range(self.max+1)]
        permutation = np.random.permutation(len(self.file))
        #permutation = np.arange(len(self.file))
        for i in permutation:
            id = self.file.iloc[i,5]
            if len(self.iterator_list[id]) == self.B:
                idx = np.array(self.iterator_list[id].copy())
                self.iterator_list[id] = []
                self.id.append(idx)
            self.iterator_list[id].append(i)
        for element in self.iterator_list:
            if len(element) > 0:
                idx = np.array(element)
                self.id.append(idx)
        self.count = 0
        return self

    def __next__(self):
        if self.count == len(self.id):
            raise StopIteration
        idx = self.id[self.count]
        bands = []
        mags = []
        errors = []
        #idx = np.sort(idx)[np.random.permutation(len(idx))]
        for id in idx:
            id_temp = self.data[id,:,0] > 0
            b = np.nonzero(id_temp)[0]
            m = self.data[id,:,0][id_temp]
            e = self.data[id,:,1][id_temp]
            #print(b,m,e)
            bands.append(b)
            mags.append(m)
            errors.append(e)
        try:
            bands = np.stack(bands)
            mags = np.stack(mags)
            errors = np.stack(errors)
        except:
            maxi = 0
            mini = 100
            for element in bands:
                maxi = max(len(element),maxi)
                mini = min(len(element),mini)
            print(maxi,mini)
            
        self.count += 1 
        return torch.from_numpy(bands),torch.from_numpy(mags),torch.from_numpy(errors),idx

    def new_row(self,idx,id,ra,dec,mag,mag_err,length,params):
        row = [id,ra,dec,mag,mag_err,length,-99,-99,-99,-99,-99,-99]
        for _ in range(5):
            row += [-99,-99,-99]
        row += [*params]
        row_prim = list(map(lambda x: np.nan if str(x) == "--" else x,row))
        #for j in range(len(self.file.columns)):
        #    self.file.iloc[idx,j] = np.nan if str(row[j]) == "--" else row[j]
        self.file.loc[idx] = row_prim

    def upload(self,id,samples,best,log_prob):
        try:
            best = best.numpy()
            samples = samples.numpy()
            log_prob = log_prob.numpy()
        except:
            pass
        offset = 7
        low = np.quantile(samples,0.16,axis = 1)
        med = np.quantile(samples,0.5,axis = 1)
        upp = np.quantile(samples,0.84,axis = 1)
        for i,num in enumerate(id):
            self.file.iloc[num,offset-1] = log_prob[i]
            self.file.iloc[num,[offset,offset+1,offset+2,offset+3,offset+4]] = best[i]
            for j in range(5):
                self.file.iloc[num,[offset + 5 + j*3, offset +5 + j*3 + 1, offset + 5 + j*3 + 2]] = np.array([low[i,j],med[i,j],upp[i,j]])
            
    def remove(self):
        for name in self.unique_remove:
            self.keep = np.logical_and(self.keep,self.unique_remove[name])
        print("mean remaining: ",np.mean(self.keep))
        print("len remaining: ",np.sum(self.keep))
        self.file = self.file.loc[self.keep,:].reset_index(drop = True)
        self.data = self.data[self.keep]

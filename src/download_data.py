import numpy as np
import pandas as pd
from astroquery.vizier import Vizier
import concurrent.futures
from utils import (
    PhotometryGenerator,
    Bands_def_all_short,
    DataBundle,
    is_set,
    M5,
    retry,
)
from tqdm import tqdm
from astropy.table import unique
from Iris import Star, Galactic
import argparse


def check_apogee(
    number: int,
    iner: bool,
    para: bool = False,
    num_p: int = 4,
    batch: int = 2000,
    name_server: str = "vizier.cds.unistra.fr",
):
    if iner:
        fil = {
            "GLAT": "<20 & >-20",
            "logg": ">0",
            "TEFF": "< 10000",
            "PROGRAMNAME": "!= yso",
            "GAIAEDR3_PHOT_G_MEAN_MAG": ">12",
        }
    else:
        fil = {
            "GLAT": "<-20 | >20",
            "logg": ">0",
            "TEFF": "< 10000",
            "PROGRAMNAME": "!= yso & != Drout_18b",
            "GAIAEDR3_PHOT_G_MEAN_MAG": ">12",
        }
    catalogs = Vizier(
        row_limit=number,
        column_filters=fil,
        vizier_server=name_server,
        columns=[
            "RAJ2000",
            "DEJ2000",
            "Teff",
            "e_Teff",
            "f_Teff",
            "Ak",
            "ALPHA_M",
            "[M/H]",
            "e_[M/H]",
            "f_[M/H]",
            "logg",
            "e_logg",
            "f_logg",
            "AFlag",
            "GaiaEDR3",
            "plx",
            "Gmag",
        ],
    ).get_catalogs("III/286/catalog")[0]
    print(catalogs.columns)
    name = "APOGEE_"
    if iner:
        name += "disc"
    else:
        name += "halo"
    print("name: ", name)
    if para:
        print("using: {} proc".format(num_p))

    bits = [7, 23, 19, 27, 3, 40, 41]
    idx = np.ones([len(catalogs)]).astype(bool)
    print("len before quality cuts: ", len(catalogs))
    for i in range(len(catalogs)):
        for bit in bits:
            if is_set(bit, catalogs["AFlag"][i]):
                idx[i] = False
    catalogs = catalogs[idx]  # remove those object that do not have good flags

    catalogs = unique(
        catalogs, "GaiaEDR3"
    )  # remove duplicates from the same GaiaDR3 index
    print("len after quality cuts: ", len(catalogs))
    sdss = {
        "V/154/sdss16": [
            [
                "psfMag_u",
                "psfMagErr_u",
                "psfMag_g",
                "psfMagErr_g",
                "psfMag_r",
                "psfMagErr_r",
                "psfMag_i",
                "psfMagErr_i",
                "psfMag_z",
                "psfMagErr_z",
            ],
            ["SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"],
            0.3,
            [True, True, True, True, True],
        ]
    }
    cat_to_use = Galactic | sdss

    data = DataBundle(name, cat_to_use, leng=len(catalogs), bands=Bands_def_all_short)
    if iner:
        data.dis_norm = 0.5
    else:
        data.dis_norm = 1
    print("norm: ", data.dis_norm)
    for i in tqdm(range(len(catalogs))):
        data.new_row(
            i,
            catalogs["GaiaEDR3"][i],
            catalogs["RAJ2000"][i],
            catalogs["DEJ2000"][i],
            catalogs["plx"][i],
            catalogs["Gmag"][i],
            0,
            [
                catalogs["Teff"][i],
                catalogs["e_Teff"][i],
                catalogs["[a/M]"][i],
                catalogs["Ak"][i],
                catalogs["logg"][i],
                catalogs["e_logg"][i],
                catalogs["[M/H]"][i],
                catalogs["e_[M/H]"][i],
            ],
        )

    names = [
        "I/355/gaiadr3",
        "II/246/out",
        "II/335/galex_ais",
        "II/312/mis",
        "II/349/ps1",
        "II/379/smssdr4",
        "V/154/sdss16",
        # "II/336/apass9",
        # "II/311/wise",
    ]
    if iner:
        names += [
            "II/328/allwise",
        ]
    else:
        names += [
            "II/365/catwise",
        ]
    print(names)
    names_remove = [
        "J/A+A/675/A195/mast",
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
    for num in range(0, len(data.file), batch):
        for name in names_remove:
            remove.append((name, num))
    if para:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {
                executor.submit(
                    retry(data.check_if_exist),
                    tup[0],
                    tup[1],
                    batch,
                    vizier_server=name_server,
                ): tup
                for tup in remove
            }

        data.remove()

        for num in range(0, len(data.file), batch):
            for name in names:
                tasks.append((name, num))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {
                executor.submit(
                    retry(data.get_photo),
                    tup[0],
                    tup[1],
                    batch,
                    vizier_server=name_server,
                ): tup
                for tup in tasks
            }
    else:
        for task in remove:
            retry(data.check_if_exist)(
                task[0], task[1], batch, vizier_server=name_server
            )

        data.remove()

        for num in range(0, len(data.file), batch):
            for name in names:
                tasks.append((name, num))

        for task in tasks:
            retry(data.get_photo)(task[0], task[1], batch, vizier_server=name_server)
    data.save("examples/")
    print(data.file)


def check_lamost(number=100, para=False, num_p=10, AFGK=True):
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
        flat_err = metal_err
    fil = {
        "snru": ">20",
        "snrg": ">20",
        "snrr": ">20",
        "snri": ">20",
        "snrz": ">20",
        "Gmag": ">12",
    }
    catalogs = Vizier(
        row_limit=number,
        column_filters=fil,
        columns=[
            "GaiaDR2",
            "Gmag",
            "RAJ2000",
            "DEJ2000",
            "Teff",
            "e_Teff",
            metal,
            metal_err,
            "logg",
            "e_logg",
        ],
    ).get_catalogs(cat)[0]
    if AFGK:
        label = "_AFGK"
    else:
        label = "_M"
    name = "LAMOST" + label
    sdss = {
        "V/154/sdss16": [
            [
                "psfMag_u",
                "psfMagErr_u",
                "psfMag_g",
                "psfMagErr_g",
                "psfMag_r",
                "psfMagErr_r",
                "psfMag_i",
                "psfMagErr_i",
                "psfMag_z",
                "psfMagErr_z",
            ],
            ["SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"],
            0.3,
            [True, True, True, True, True],
        ]
    }
    cat_to_use = Galactic | sdss

    data = DataBundle(name, cat_to_use, leng=len(catalogs), bands=Bands_def_all_short)
    data.dis_norm = 0.5
    print("norm: ", data.dis_norm)
    for i in tqdm(range(len(catalogs))):
        data.new_row(
            i,
            catalogs["GaiaDR2"][i],
            catalogs["RAJ2000"][i],
            catalogs["DEJ2000"][i],
            0,
            catalogs["Gmag"][i],
            0,
            [
                catalogs["Teff"][i],
                catalogs["e_Teff"][i],
                0,
                0,
                catalogs["logg"][i],
                catalogs["e_logg"][i],
                catalogs[flat][i],
                catalogs[flat_err][i],
            ],
        )

    names = [
        "I/355/gaiadr3",
        "II/246/out",
        "II/335/galex_ais",
        "II/312/mis",
        "II/349/ps1",
        "II/379/smssdr4",
        "II/336/apass9",
        "V/154/sdss16",
        # "II/328/allwise",
        # "II/311/wise",
    ]
    names_remove = [
        "J/A+A/675/A195/mast",
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
    for num in range(0, len(data.file), batch):
        for name in names_remove:
            remove.append((name, num))
    if para:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {
                executor.submit(retry(data.check_if_exist), tup[0], tup[1]): tup
                for tup in remove
            }

        data.remove()

        for num in range(0, len(data.file), batch):
            for name in names:
                tasks.append((name, num))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {
                executor.submit(retry(data.get_photo), tup[0], tup[1]): tup
                for tup in tasks
            }
    else:
        for task in remove:
            retry(data.check_if_exist)(task[0], task[1])

        data.remove()

        for num in range(0, len(data.file), batch):
            for name in names:
                tasks.append((name, num))

        for task in tasks:
            data.get_photo(task[0], task[1])
    data.save("examples/")


def check_cluster(name_cluster, data_csv, para=False, num_p=50):
    data_csv = pd.read_csv(data_csv, delim_whitespace=True)
    ra_list = data_csv["ra"].values
    dec_list = data_csv["dec"].values
    id = data_csv["memberprob"].values > 0.90
    ra_list = ra_list[id]
    dec_list = dec_list[id]
    data_csv = data_csv.iloc[id]
    print("number of stars: ", len(ra_list))
    del Galactic["II/319/las9"]
    catalog = Galactic | M5
    data = DataBundle(
        name_cluster, catalog, leng=len(ra_list), bands=Bands_def_all_short
    )
    data.dis_norm = 0.5
    print("norm: ", data.dis_norm)
    for i in tqdm(range(len(ra_list))):
        data.new_row(
            i,
            data_csv["source_id"].values[i],
            data_csv["ra"].values[i],
            data_csv["dec"].values[i],
            data_csv["plx"].values[i],
            data_csv["g_mag"].values[i],
            0,
            [-99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0, -99.0],
        )

    names = [
        "I/355/gaiadr3",
        # "II/246/out",
        "II/335/galex_ais",
        # "II/312/ais",
        "II/312/mis",
        "II/349/ps1",
        "II/365/catwise",
        "II/379/smssdr4",
        "II/336/apass9",
        "V/154/sdss16",
        "J/A+A/632/A56/catalog",
        "II/319/las9",
        # "II/311/wise",
    ]
    names_remove = [
        "J/A+A/675/A195/mast",
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
    for num in range(0, len(data.file), batch):
        for name in names_remove:
            remove.append((name, num))
    if para:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {
                executor.submit(data.check_if_exist, tup[0], tup[1]): tup
                for tup in remove
            }

        data.remove()

        for num in range(0, len(data.file), batch):
            for name in names:
                tasks.append((name, num))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_p) as executor:
            out = {
                executor.submit(data.get_photo, tup[0], tup[1]): tup for tup in tasks
            }
    else:
        for task in remove:
            data.check_if_exist(task[0], task[1])

        data.remove()

        for num in range(0, len(data.file), batch):
            for name in names:
                tasks.append((name, num))

        for task in tasks:
            data.get_photo(task[0], task[1])
    data.save("examples/")
    print(data.file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from VizieR")
    parser.add_argument(
        "-n", "--number", type=int, default=400000, help="Number of stars to download"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=20000, help="Batch size for downloading"
    )
    parser.add_argument(
        "-p", "--para", type=int, help="Use parallel processing", default=1
    )
    parser.add_argument(
        "-np",
        "--num_p",
        type=int,
        default=16,
        help="Number of processes to use for parallel processing",
    )
    args = parser.parse_args()
    name_server = "vizier.cds.unistra.fr"
    if args.para:
        print("parallel processing")
    check_apogee(
        args.number,
        iner=True,
        para=args.para,
        name_server=name_server,
        batch=args.batch,
        num_p=args.num_p,
    )
    check_apogee(
        args.number,
        iner=False,
        para=args.para,
        name_server=name_server,
        batch=args.batch,
        num_p=args.num_p,
    )


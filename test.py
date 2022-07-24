#!/usr/bin/env python

import numpy as np
from Helpers import *
import pickle
import os
from PIL import Image

if __name__ == '__main__':
    for z in [200]:
        dim = 20 * 14 + 6 * 14
        # a = 97.85
        a = 200
        # z = 90
        lmbd = 1.051903
        # ProgressBar = False
        n = 0
        meanafterWP = np.array([])
        cmp = np.array([])
        qr_diff = np.array([])
        thresholds = np.array([])
        MultiCore = False
        BeamDist = 900
        n_max = 500
        # SIM = False

        # make folder
        # add_text = input("Any special string for saving?: ")
        add_text = "DivBeam"+str(BeamDist)+"mm"
        dir_path_base = "Data/Project2_a_{a}_{add_text}" \
            .format(a=a, dim=dim, z=z, add_text=add_text)
        dir_path = dir_path_base + "/Project2_dim_{dim}_z_{z}_{add_text}" \
            .format(a=a, dim=dim, z=z, add_text=add_text)
        try:
            os.mkdir(dir_path_base)
        except:
            print('Directory already exists...')

        try:
            os.mkdir(dir_path)
        except:
            print('Directory already exists...')
        print("Opened: " + dir_path)
        ## Create data
        if 1:
            print("Making fresh start....")

            # Beam = np.ones((dim, dim))

            # def BeamPhase(X, Y, k, z, a):
            #   return k * (np.sqrt(z ** 2 + (np.sqrt((X - (a / 2)) ** 2 + (Y - (a / 2)) ** 2)) ** 2) - z)

            X = np.linspace(-a/2, a/2, dim)
            Y = np.linspace(-a/2, a/2, dim)
            X, Y = np.meshgrid(X, Y)

            # twoD_Gaussian((x, y), 1, (a - 1) / 2, (a - 1) / 2, 10,10)

            # xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset
            Beam = np.sqrt(gaussian_2d((X, Y), 1, 0, 0, 10, 10, 0, 0)).reshape((dim, dim))
            # data from 500mm from source (20210609)

            # M_phase = BeamPhase(X, Y, 2 * np.pi / lmbd, BeamDist, a - 1)
            # Beam = Beam * np.exp(1j * M_phase)

            ## Subtract Phaseplate size 120x120

            # frm1 = int((a - 120) / 2)
            # to1 = int(a - (a - 120) / 2)
            # Mask = np.zeros((a, a))
            # Mask = np.ones((dim, dim))
            # Mask[frm1:to1, frm1:to1] = 1
            img = Image.open('Targets/circle.png').resize(size=(dim//2, dim//2))
            Mask0 = np.array(img, dtype=float) / 255  # convert to numpy array and scale values
            Mask0 = abs(Mask0[:, :, 1] - 1)
            Mask0[Mask0 < 0.5] = 0  # binarize
            Mask0[Mask0 > 0.5] = 1
            Mask = np.zeros((dim, dim))
            Mask[dim//2-dim//4:dim//2+dim//4,dim//2-dim//4:dim//2+dim//4] = Mask0
            Beam = Beam * Mask

            AfterWP = np.zeros((dim, dim))
            AfterWPP = np.zeros((dim, dim))

            ##  Target

            wish = np.zeros((dim, dim))
            ##simple geometrics
            wish[4*14:-3*14:14, dim//2] = 1
            wish[4 * 14:-3 * 14:14, dim // 2-14] = 1
            wish[4 * 14:-3 * 14:14, dim // 2+14] = 1

            Img = wish
            ImgTest = wish * 0

            print('Making Beam symmetrical...')
            Beam[0:int(len(Beam) / 2), int(len(Beam) / 2):] = np.fliplr(
                Beam[0:int(len(Beam) / 2), 0:int(len(Beam) / 2)])
            Beam[int(len(Beam) / 2):, int(len(Beam) / 2):] = np.flip(Beam[0:int(len(Beam) / 2), 0:int(len(Beam) / 2)])
            Beam[int(len(Beam) / 2):, 0:int(len(Beam) / 2)] = np.flipud(
                Beam[0:int(len(Beam) / 2), 0:int(len(Beam) / 2)])

            data = {
                "Beam": Beam,
                # "AfterWP": AfterWP,
                "AfterWPP": AfterWPP,
                "wish": wish,
                "n": n,
                "Mask": Mask,
                "lmbd": lmbd,
                "BeamDist": BeamDist,
                "z": z,
                "a": a,
                "dim": dim,
                "ImgTest": ImgTest,
                # "Img": Img,
                "meanafterWP": meanafterWP,
                "cmp": cmp,
                "qr_diff": 0,
                "qr_size": None,
                "qr_target_ratio": None,
                "thresholds": thresholds,
                "add_text": add_text
            }
            data_base = {key: data[key] for key in
                         ["Beam", "wish", "Mask", "lmbd", "BeamDist", "z", "a",
                          "dim", "qr_size", "qr_target_ratio", "add_text"]}
            f_name = dir_path + \
                     "/Project1_dim_{dim}_z_{z}_{add_text}_BASE.pgz".format(dim=dim,
                                                                            z=z, add_text=add_text)
            with gzip.GzipFile(f_name, 'wb') as f:
                cpickle.dump(data_base, f)
            # data = [Beam, AfterWP, AfterWPP, wish, n, Mask, lmbd, BeamDist, z, a,
            #        dim, ImgTest, Img, meanafterWP, cmp, qr_diff, qr_size, qr_target_ratio]
        else:
            if 0:  # OLD SAVING STYLE
                PIK = dir_path + "/Project1_dim_{dim}_z_{z}_n_429.dat".format(dim=dim, z=z)
                print("Loading old data from: " + PIK)
                with open(PIK, "rb") as f:
                    data = cpickle.load(f)

                data_base = {key: data[key] for key in
                             ["Beam", "wish", "Mask", "lmbd", "BeamDist", "z", "a",
                              "dim", "qr_size", "qr_target_ratio", "add_text"]}
                f_name = dir_path \
                         + "/Project1_dim_{dim}_z_{z}_{add_text}_BASE.pgz".format(
                    dim=dim, z=z, add_text=add_text)
                with gzip.GzipFile(f_name, 'wb') as f:
                    cpickle.dump(data_base, f)

                f_name = dir_path + \
                         "/Project1_dim_{dim}_z_{z}_n_429.pgz".format(dim=dim, z=z)
                data_int = {key: data[key] for key in
                            ["AfterWPP", "n", "ImgTest", "meanafterWP", "cmp", "qr_diff", "thresholds"]}
                with gzip.GzipFile(f_name, 'wb') as f:
                    cpickle.dump(data_int, f)
            else:
                PIK = dir_path + \
                      "/Project1_dim_{dim}_z_{z}_{add_text}_BASE.pgz".format(dim=dim,
                                                                             z=z, add_text=add_text)
                # with open(PIK, "rb") as f:
                with gzip.open(PIK, "rb") as f:
                    database = cpickle.load(f)

                PIK = dir_path + "/Project1_dim_{dim}_z_{z}_n_206.pgz".format(dim=dim, z=z)
                print("Loading old data from: " + PIK)
                # with open(PIK, "rb") as f:
                with gzip.open(PIK, "rb") as f:
                    dataint = cpickle.load(f)

                data = {**database, **dataint}
            # archive = zipfile.ZipFile('images.zip', 'r')
            # imgfile = archive.open('PIK')
        # # load from dim=250
        # PIK = "Data/Final/Project1_dim_500_BeamDist_75_" + "/Project1_dim_500_BeamDist_75_n_150.dat"
        # print("Loading old data with n=150: " + PIK)
        # with open(PIK, "rb") as f:
        #     data0 = pickle.load(f)
        # # Beam, AfterWP, AfterWPP, wish, n, Mask, lmbd, BeamDist, z, a, dim, ImgTest, Img, meanafterWP, cmp = data
        # #   0     1         2       3    4   5      6     7       8  9   10   11       12    13          14
        # Img0 = data0[12]
        # AfterWPP0 = data0[2]
        # ImgTest0 = data0[11]
        # Img = rebin(Img0, (dim, dim))
        # AfterWPP = rebin(AfterWPP0, (dim, dim))
        # ImgTest = rebin(ImgTest0, (dim, dim))
        # del data0
        #
        # data = [Beam, AfterWP, AfterWPP, wish, n, Mask, lmbd, BeamDist, z, a, dim, ImgTest, Img, meanafterWP, cmp]

        #fig, axs = main_fig(data)   

        #Cntn = input("Write 1 to continue? ")
        Cntn = "1"

        dta = AlgorithmInput(
            dim=dim,
            z=z,
            a=a,
            lmbd=lmbd,
            Beam=Beam,
            ImgTest=ImgTest,
            wish=wish,
            AfterWPP=AfterWPP,
            Mask=Mask,
            label=add_text,
        )

        #figure = AlgorithmFigure(dta,dir_path)
        main = Algorithm(dta, dir_path, MultiCore=MultiCore, QR_detector_switch=False, numba=True,
                         n_max=n_max)
        main.check_symmetries()
        main.run_algorithm()

#        plt.close("all")
        # with open(PIK, "wb") as f:
        #   pickle.dump(data, f)
        # print("Saved")

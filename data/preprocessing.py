import sys

import numpy as np
import pandas as pd
from PIL import Image
import scipy
import scipy.signal


def load_set_from_CSV(exp_set):
    folder = '../fra_exp_csv'

    # prepare dictionaries to reference data
    exp_derivs = {}
    exp_num_chem = {}

    # start toolbar
    sys.stdout.write("Loading experimental data \n")
    sys.stdout.flush()
    loaded = 0

    for count, chem in enumerate(exp_set):
        nums = exp_set[chem]

        #loads image then normalizes data
        datas = []
        for b, num in enumerate(nums):
            data = pd.read_csv(folder + "/%06d.csv" %
                               (num))  # load data from file
            data_spectra = (data.values[:, 1:] -
                            np.min(data.values[:, 1:])).transpose()  # pylint: disable=no-member
            datas.append(data_spectra)

            # update the bar
            sys.stdout.write('\r')
            bar = ((b + 1) * 100) / len(nums)
            # the exact output you're looking for:
            sys.stdout.write(
                str(chem) + ": [%-20s] %d%%" %
                ('=' * int(bar / 5 - 1) + str(int(bar % 10)), bar))
            sys.stdout.flush()
            loaded += 1

        phase_derivs = []
        for da in datas:
            # performing manually: phase_derivs = [da.getPhaseDerivative('spectral',1,smoothing=True,normalize=False) for da in datas]
            # Step 1: get phase derivative data at each time step
            ft_data = np.fft.fft(da)
            # Step 2: Calculates the phase
            R = np.real(ft_data[:, 1])
            I = np.imag(ft_data[:, 1])
            phi = I / (R**2 + I**2)**0.5
            # Step 3: Calculate phase derivative and perform smoothing with default parameters
            phi_deriv = np.diff(phi)
            phi_deriv = scipy.signal.savgol_filter(phi_deriv,
                                                   window_length=31,
                                                   polyorder=2)

            phase_derivs.append(phi_deriv)

        #phase_derivs = [deriv for deriv in phase_derivs]

        #saves the derivs in the dictionary
        for n, num in enumerate(nums):
            if np.size(phase_derivs[n]) < 600:
                phase_derivs[n] = np.append(phase_derivs[n], [0])
            exp_derivs[num] = phase_derivs[n]
            exp_num_chem[num] = chem

        # update the bar
        sys.stdout.write("   " + str(count + 1) + "/" + str(len(exp_set)) +
                         " complete\n")
        sys.stdout.flush()

    exp_set_size = loaded
    print("Length of experimental set loaded: " + str(exp_set_size))

    return exp_derivs, exp_num_chem, exp_set_size


def load_images_from_CSV(exp_set):
    folder = '../fra_exp_csv'

    # prepare dictionaries to reference data
    exp_images = {}
    exp_data = {}
    exp_num_chem = {}

    # prepare arrays to story data
    train_data = []
    train_im = []
    labels = []
    exp_num = []

    # start toolbar
    sys.stdout.write("Loading experimental data \n")
    sys.stdout.flush()
    loaded = 0

    for count, chem in enumerate(exp_set):
        nums = exp_set[chem]

        #loads image then normalizes data
        for b, num in enumerate(nums):
            data = pd.read_csv(folder + "/%06d.csv" %
                               (num))  # load data from file
            data_im = (data.values[:, 1:] -
                       np.min(data.values[:, 1:])).transpose()  # pylint: disable=no-member
            data_im = data_im * 255 / np.max(data_im)

            img = Image.fromarray(data_im)
            img = img.convert('RGB')
            img_resized = img.resize(
                (299, 299))  # necessary for Inception module

            exp_images[num] = data_im  # keep unconverted data for reference
            exp_num_chem[num] = chem  # keep reference of chemical
            exp_data[num] = np.array(img_resized) * 255 / np.max(img_resized)
            exp_num.append(num)
            train_im.append(img_resized)
            train_data.append(np.array(img_resized) * 255 / np.max(img_resized))
            labels.append(chem)  # create labels

            # update the bar
            sys.stdout.write('\r')
            bar = ((b + 1) * 100) / len(nums)
            # the exact output you're looking for:
            sys.stdout.write(chem + ": [%-20s] %d%%" %
                             ('=' * int(bar / 5 - 1) + str(int(bar % 10)), bar))
            sys.stdout.flush()
            loaded += 1

        # update the bar
        sys.stdout.write("   " + str(count + 1) + "/" + str(len(exp_set)) +
                         " complete\n")
        sys.stdout.flush()

    train_data = np.array(train_data)

    exp_set_size = loaded
    print("Length of experimental set loaded: " + str(exp_set_size))

    return train_data, exp_num_chem, exp_set_size

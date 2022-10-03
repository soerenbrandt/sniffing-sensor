"""
Creadted Mar 8 2019 by Soeren Brandt

This file contains all the functions used in my python scripts for HANDLING OF DATA
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
from PIL import Image

DATA_PATH = Path(__file__).parent.joinpath("fra_exp_csv")
SNIFFING_PATH = Path(__file__).parent.joinpath("sniffing_data")


class DataLoader:
    def __init__(
        self, dataset: dict, transforms: list = None, folder: str = DATA_PATH
    ):
        self.folder = Path(folder)
        self.data = dataset
        self.transforms = transforms or []

    def load(self):
        exp_derivs = {}
        exp_num_chem = {}

        loaded = 0
        print("Loading experimental data")

        for set_number, (chem, exp_ids) in enumerate(self.data.items()):
            loadbar = self.loadbar(
                f"{chem} ({set_number+1}/{len(self.data)})", len(exp_ids)
            )
            for b, experiment in enumerate(exp_ids):
                loadbar.update(b)

                try:
                    file_path = self.folder.joinpath(f"{experiment}.csv")
                    if not file_path.exists():
                        file_path = self.folder.joinpath(
                            f"{experiment:06d}.csv"
                        )
                    data = pd.read_csv(file_path)
                except FileNotFoundError as e:
                    logging.warning(str(e))
                    continue

                for t in self.transforms:
                    data = t.transform(data)

                exp_derivs[experiment] = data
                exp_num_chem[experiment] = chem
                loaded += 1
            loadbar.close()
        print("Length of experimental set loaded: " + str(loaded))

        return exp_derivs, exp_num_chem

    class loadbar:
        def __init__(self, name, length):
            self.name = name
            self.length = length
            self.bar = lambda x: f"{name}: [%-20s] %d%%" % (
                "=" * int(x / 5 - 1) + str(int(x % 10)),
                x,
            )

        def update(self, num: int):
            percent_filled = ((num + 1) * 100) / self.length
            print(self.bar(percent_filled), end="\r")

        def close(self):
            print(self.bar(100), "complete")


class Transform(ABC):
    def __call__(self, data):
        return self.transform(data)

    @staticmethod
    @abstractmethod
    def transform(data) -> np.array:
        pass


class PhaseTransform(Transform):
    @staticmethod
    def transform(data) -> np.array:
        data_spectra = (
            data.values[:, 1:] - np.min(data.values[:, 1:])
        ).transpose()

        # performing manually: phase_derivs = [da.getPhaseDerivative('spectral',1,smoothing=True,normalize=False) for da in datas]
        # Step 1: get phase derivative data at each time step
        ft_data = np.fft.fft(data_spectra)
        # Step 2: Calculates the phase
        R = np.real(ft_data[:, 1])
        I = np.imag(ft_data[:, 1])
        phi = I / (R**2 + I**2) ** 0.5

        # smooth
        phi[0] = phi[1]
        phi = scipy.signal.savgol_filter(phi, window_length=31, polyorder=3)

        return phi


class DerivTransform(Transform):
    @staticmethod
    def transform(data) -> np.array:
        phi = PhaseTransform.transform(data)

        # Step 3: Calculate phase derivative and perform smoothing with default parameters
        phi_deriv = np.diff(phi)
        phi_deriv = scipy.signal.savgol_filter(
            phi_deriv, window_length=31, polyorder=2
        )

        if np.size(phi_deriv) < 600:
            phi_deriv = np.append(phi_deriv, [0])

        return phi_deriv


class ImageTransform(Transform):
    size = (299, 299)

    def __init__(self, size=(299, 299)):
        """Convert raw data to image format.

        Args:
            size (tuple[int, int]): Image dimensions. Defaults to (299, 299) for Inception modeul.
        """
        super().__init__()
        self.size = size

    def transform(self, data) -> np.array:
        data_im = (data.values[:, 1:] - np.min(data.values[:, 1:])).transpose()
        data_im = data_im * 255 / np.max(data_im)

        img = Image.fromarray(data_im)
        img = img.convert("RGB")
        img_resized = img.resize(self.size)

        return np.array(img_resized)


class NormalizeTransform(Transform):
    @staticmethod
    def transform(data) -> np.array:
        return data / np.sqrt(np.sum(data**2))


class NormalizePhaseTransform(Transform):
    @staticmethod
    def transform(data) -> np.array:
        return (data - np.mean(data)) / np.std(data)


class SleepTransform(Transform):
    @staticmethod
    def transform(data):
        del data
        time.sleep(0.5)


# ------------------------------------------------------------#
# Old functions to load data using fra_expt (requires python 2.7)
# ------------------------------------------------------------#
# import our modules
# import fra_expt

# import time
# import sys
# import warnings
# from collections import OrderedDict

# def load_labeled_set(exp_set):  # originally load_exps
#     #sets up dictionaries to hold the derivs for different compounds and corresponding number/chemical
#     exp_derivs = {}
#     exp_num_name = {}

#     # create report if chem labels don't match
#     report = ""

#     # start toolbar
#     sys.stdout.write("Loading experimental data \n")
#     sys.stdout.flush()
#     loaded = 0

#     for count, chem in enumerate(exp_set):
#         nums = exp_set[chem]

#         t = time.time()  # start timer
#         #loads experiments then smoothes and normalizes data
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             exps = []  #[fra_expt.get_expt(n) for n in nums]
#             for b, n in enumerate(nums):
#                 exp = fra_expt.get_expt(n)
#                 exps.append(exp)

#                 # check if chemical is actually labeled correctly
#                 if sum([
#                         exp.sample_composition[string] == 1
#                         for n, string in enumerate(exp.sample_composition.keys(
#                         )) if string.lower() == chem.lower()
#                 ]) != 1:
#                     report = report + "\nExp. " + str(
#                         n
#                     ) + " does not match. Labeled " + chem + " but lists " + str(
#                         [
#                             key + ": " + str(value)
#                             for key, value in exp.sample_composition.items()
#                         ])

#                 # update the bar
#                 sys.stdout.write('\r')
#                 bar = ((b + 1) * 100) / len(nums)
#                 # the exact output you're looking for:
#                 sys.stdout.write(
#                     str(chem) + ": [%-20s] %d%%" %
#                     ('=' * (bar / 5 - 1) + str(bar % 10), bar))
#                 sys.stdout.flush()
#                 loaded += 1

#         t1 = int(time.time() - t)  # time loading
#         sys.stdout.write(" " + str(t1 / 60) + ":" + "%02.d" % (t1 % 60) +
#                          "min loading")
#         sys.stdout.flush()

#         datas = [exp.main_spec_data.set_times(600) for exp in exps]
#         datas = [da.lower_data_freq(1) for da in datas
#                  ]  #necessary because some experiments sampled at 4 Hz or 1 Hz
#         #    freqs = [da.get_data_frequency() for da in datas]
#         #    print freqs

#         #calculate the phase derivative, UNNORMALIZED TO SHOW BAD LINEAR PCA WEAKNESSES
#         phase_derivs = [
#             da.getPhaseDerivative('spectral',
#                                   1,
#                                   smoothing=True,
#                                   normalize=False) for da in datas
#         ]
#         phase_derivs = [deriv for deriv in phase_derivs]

#         #saves the derivs in the dictionary
#         for n, num in enumerate(nums):
#             if np.size(phase_derivs[n]) < 600:
#                 phase_derivs[n] = np.append(phase_derivs[n], [0])
#             exp_derivs[num] = phase_derivs[n]
#             exp_num_name[num] = chem

#         t2 = int(time.time() - t) - t1  # time derivatives
#         sys.stdout.write("  " + str(t2 / 60) + ":" + "%02.d" % (t2 % 60) +
#                          "min derivatives")
#         sys.stdout.flush()

#         # update the bar
#         sys.stdout.write("   " + str(count + 1) + "/" + str(len(exp_set)) +
#                          " complete\n")
#         sys.stdout.flush()

#     exp_set_size = len(np.concatenate(exp_set.values(), axis=None))
#     print("Length of experimental set loaded: " + str(exp_set_size))

#     return (exp_derivs, exp_num_name, exp_set_size)

# def load_unlabeled_set(exp_set):
#     # setup dictionary to hold datasets
#     dataset = OrderedDict()

#     # start toolbar
#     sys.stdout.write("Loading experimental data \n")
#     sys.stdout.flush()
#     loaded = 0

#     for name, nums in exp_set.items:
#         #sets up dictionaries to hold the derivs for different compounds and corresponding number/chemical
#         exp_derivs = {}
#         exp_concentrations = {}

#         for num in nums:  #enumerate(nums):
#             t = time.time()  # start timer
#             #loads experiment then smoothes and normalizes data
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 exp = fra_expt.get_expt(num)

#             data = exp.main_spec_data.set_times(
#                 600)  # might have to set to 500 for regression
#             data = data.lower_data_freq(1)

#             #calculate the phase derivative
#             phase_deriv = data.getPhaseDerivative('spectral',
#                                                   1,
#                                                   smoothing=True,
#                                                   normalize=False)
#             exp_derivs[num] = phase_deriv / np.sqrt(np.sum(phase_deriv**
#                                                            2))  # normalize

#             #get concentration
#             exp_concs[num] = exp.sample_composition

#             # update the bar
#             sys.stdout.write('\r')
#             bar = ((count + 1) * 100) / len(exp_set)
#             # the exact output you're looking for:
#             sys.stdout.write("[%-20s] %d%%" %
#                              ('=' * (bar / 5 - 1) + str(bar % 10), bar))
#             sys.stdout.flush()
#             loaded += 1

#         # clean up concentrations dictionary
#         keys = set(
#             [key for composition in exp_concentrations for key in composition])
#         exp_concs = {}
#         for num in nums:
#             exp_concs[num] = {}
#             for chem in keys:
#                 try:
#                     exp_concs[num][chem.capitalize()] = A[num][chem]
#                 except:
#                     exp_concs[num][chem.capitalize()] = 0

#         # add dictionaries to dataset
#         dataset[name] = [exp_derivs, exp_concs]

#         # update the bar
#         sys.stdout.write("   " + " complete\n")
#         sys.stdout.flush()

#     exp_set_size = count + 1
#     print(
#         str(exp_set_size) + " experiments loaded in " + len(exp_set) +
#         " datasets.")

#     return (exp_derivs, exp_concentrations, exp_set_size)

# def load_unlabeled_list(exp_set, chem):  # previously load_list
#     #sets up dictionaries to hold the derivs for different compounds and corresponding number/chemical
#     exp_derivs = {}
#     exp_concentrations = {}

#     # start toolbar
#     sys.stdout.write("Loading experimental data \n")
#     sys.stdout.flush()
#     loaded = 0

#     for count, num in enumerate(exp_set):
#         t = time.time()  # start timer
#         #loads experiment then smoothes and normalizes data
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             exp = fra_expt.get_expt(num)

#         data = exp.main_spec_data.set_times(
#             600)  # might have to set to 500 for regression
#         data = data.lower_data_freq(1)

#         #calculate the phase derivative
#         phase_deriv = data.getPhaseDerivative('spectral',
#                                               1,
#                                               smoothing=True,
#                                               normalize=False)
#         exp_derivs[num] = phase_deriv / np.sqrt(np.sum(phase_deriv**
#                                                        2))  # normalize

#         #get concentration
#         exp_concentrations[num] = exp.sample_composition[chem]

#         # update the bar
#         sys.stdout.write('\r')
#         bar = ((count + 1) * 100) / len(exp_set)
#         # the exact output you're looking for:
#         sys.stdout.write("[%-20s] %d%%" % ('=' *
#                                            (bar / 5 - 1) + str(bar % 10), bar))
#         sys.stdout.flush()
#         loaded += 1

#     # update the bar
#     sys.stdout.write("   " + " complete\n")
#     sys.stdout.flush()

#     exp_set_size = count + 1
#     print("Length of experimental set loaded: " + str(exp_set_size))

#     return (exp_derivs, exp_concentrations, exp_set_size)

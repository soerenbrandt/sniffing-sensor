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
from scipy.interpolate import interp1d

DATA_PATH = Path(__file__).parent.joinpath("fra_exp_csv")
SNIFFING_PATH = Path(__file__).parent.joinpath("sniffing_data")


class DataLoader:
    def __init__(
        self,
        dataset: dict,
        transforms: list = None,
        truncate_data_after: int = 600,
        folder: str = DATA_PATH,
    ):
        self.folder = Path(folder)
        self.data = dataset
        self.transforms = transforms or []
        self.truncate_data_after = truncate_data_after
        self._metadata = {}

    def load(self):
        exp_derivs, exp_num_chem = {}, {}
        self._metadata["wavelengths"] = {}

        loaded = 0
        print("Loading experimental data")

        for set_number, (chem, exp_ids) in enumerate(self.data.items()):
            loadbar = self.loadbar(
                f"({set_number+1}/{len(self.data)}) {chem} ({len(exp_ids)} exp.)",
                len(exp_ids),
            )
            for b, experiment in enumerate(exp_ids):
                loadbar.update(b)

                try:
                    file_path = self.folder.joinpath(f"{experiment}.csv")
                    if not file_path.exists():
                        file_path = self.folder.joinpath(
                            f"{experiment:06d}.csv"
                        )

                    data = (
                        pd.read_csv(file_path)
                        .drop(columns="Unnamed: 0", errors="ignore")
                        .iloc[:, : self.truncate_data_after + 1]
                    )
                    wavelengths = data.values[:, 0]
                except FileNotFoundError as e:
                    logging.warning(str(e))
                    continue

                for t in self.transforms:
                    data = t.transform(data)

                exp_derivs[experiment] = data
                exp_num_chem[experiment] = chem
                self._metadata["wavelengths"][experiment] = wavelengths
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
        wavelengths = (data.values[:, :1]).transpose()[0]

        # Step 1: get phase derivative data at each time step
        ft_data = np.fft.fft(data_spectra)
        # Step 2: Calculates the phase
        R = np.real(ft_data[:, 1])
        I = np.imag(ft_data[:, 1])
        phi = I / (R**2 + I**2) ** 0.5

        # smooth
        phi = scipy.signal.savgol_filter(phi, window_length=31, polyorder=3)

        # scale
        # In the case of static analysis where phi only ever increases, this is
        #  equivalent to initial_peak=wavelengths[np.argmax(data_spectra[0, :])]
        initial_peak = wavelengths[np.argmax(data_spectra, axis=1).min()]
        # In the case of static analysis where phi only ever increases, this is
        #  equivalent to final_peak=wavelengths[np.argmax(data_spectra[-1, :])]
        final_peak = wavelengths[np.argmax(data_spectra, axis=1).max()]
        scale = (
            lambda x: (x - phi.min())
            / (phi.max() - phi.min())
            * (final_peak - initial_peak)
            + initial_peak
        )

        return scale(phi)


class DerivTransform(Transform):
    @staticmethod
    def transform(data) -> np.array:
        phi = PhaseTransform.transform(data)

        # Step 3: Calculate phase derivative and perform smoothing with default parameters
        phi_deriv = np.diff(phi)

        # smooth
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


class RiverTransform(Transform):
    @staticmethod
    def transform(data) -> np.array:
        data_im = (data.values[:, 1:] - np.min(data.values[:, 1:])).transpose()
        data_im = data_im * 255 / np.max(data_im)

        img = Image.fromarray(data_im)

        return np.array(img)


class StaticShiftTransform(Transform):
    def __init__(self, normalize=False):
        """Convert raw data to image format.

        Args:
            size (tuple[int, int]): Image dimensions. Defaults to (299, 299) for Inception modeul.
        """
        super().__init__()
        self.normalize = normalize

    def transform(self, data) -> np.array:
        data_spectra = (
            data.values[:, 1:] - np.min(data.values[:, 1:])
        ).transpose()
        wavelengths = (data.values[:, :1]).transpose()[0]

        # Step 1: get shift between first and last spectrum
        reflectance_shift = data_spectra[0] - data_spectra[-1]
        if self.normalize:
            reflectance_shift /= max(data_spectra[-1])
        # Step 2: Smooth data #moved latee: and normalize the spectrum
        reflectance_shift = scipy.signal.savgol_filter(
            reflectance_shift, window_length=31, polyorder=2
        )
        # Reflectance_shift = Reflectance_shift/np.sqrt(np.sum(Reflectance_shift**2)) #moved to later
        # Step 3: Interpolate spectrum to 600-dimensional array
        reflectance_shift_interp = interp1d(wavelengths, reflectance_shift)

        return reflectance_shift_interp(np.linspace(401, 799, 600))


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

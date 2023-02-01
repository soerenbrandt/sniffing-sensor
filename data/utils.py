"""

"""

import random
import sqlite3
from collections import OrderedDict

import numpy as np
import pandas as pd

from .experiment_parameters import ANALYTES_DB


def train_test_split(dataset, test_size=1 / 2.5, seed=0):
    # set seed of random function
    random.seed(seed)

    # create training set and test set
    exp_set_train = OrderedDict()
    exp_set_test = OrderedDict()

    for chem, values in dataset.items():
        # determine test set length and sample random experiments
        test_len = int(len(values) * test_size)
        test_ind = random.sample(range(0, len(values)), test_len)
        # print(chem + ": " + str(test_ind))

        # get test and training sets
        exp_set_test[chem] = [values[ind] for ind in test_ind]
        exp_set_train[chem] = [
            values[ind]
            for ind in list(set(range(0, len(values))) - set(test_ind))
        ]

    return (exp_set_train, exp_set_test)


def sort_by_set(exp_derivs, *sets):
    """Ex: train_derivs, train_lbl, test_derivs, test_lbl = sort_by_set(exp_derivs, exp_num_name, train_set, test_set)"""
    # create empty dictionaries
    sorted_dataset = []
    for dataset in sets:
        derivs = []
        labels = []

        for chem, set_numbers in dataset.items():
            for num in set_numbers:
                derivs.append(
                    exp_derivs[num] / np.sqrt(np.sum(exp_derivs[num] ** 2))
                )
                labels.append(chem)

        sorted_dataset.append(derivs)
        sorted_dataset.append(labels)

    return sorted_dataset


def create_label_for(dataset, by):
    """by = Concentration, Vapor Pressure, Boiling Point, Flash Point, Viscosity"""
    conn = sqlite3.connect(ANALYTES_DB)

    label = []

    for chem, values in dataset.items():
        values_df = pd.read_sql_query(
            'SELECT "Experiment Number", "Analyte ID" FROM ExperimentData WHERE "Experiment Number" IN ('
            + ", ".join(map(str, values))
            + ")",
            conn,
        )

        for n in range(0, len(values)):
            if by == "Concentration":
                label_df = pd.read_sql_query(
                    'SELECT "Analyte ID", "Component 1", "Concentration 1", "Component 2", "Concentration 2" FROM AnalyteData WHERE "Analyte ID" IS '
                    + str(values_df["Analyte ID"][n]),
                    conn,
                )
                if label_df["Component 1"][0] == chem.lower():
                    label.append(float(label_df["Concentration 1"][0]))
                else:
                    label.append(float(label_df["Concentration 2"][0]))
            else:
                label_df = pd.read_sql_query(
                    'SELECT "Analyte ID", "'
                    + by
                    + '" FROM AnalyteData WHERE "Analyte ID" IS '
                    + str(values_df["Analyte ID"][0]),
                    conn,
                )

                if by in ["Boiling Point", "Flash Point"]:
                    label.append(float(label_df[by][0]))  # +273.15)
                else:
                    label.append(float(label_df[by][0]))

    return label


def data_dimensionality(data):
    """Initializes PCA, then fits to the sample"""
    from sklearn.decomposition import PCA

    pca = PCA()  # n_components=np.min([exp_set_size, 67]))
    pca.fit_transform(data)
    print(
        "Dimensionality: "
        + str(np.argmax(np.cumsum(pca.explained_variance_ratio_) > 0.95))
    )  # report total variance captured by PCA
    print(
        "Dimensions at 1%: "
        + str(np.argmax(pca.explained_variance_ratio_ < 0.01))
    )  # report total variance captured by PCA
    print(
        "Captured variance: "
        + str((pca.explained_variance_ratio_[:10].round(2)))
    )  # report total variance captured by PCA

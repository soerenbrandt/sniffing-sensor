"""
Creadted Mar 8 2019 by Soeren Brandt

This file contains all the DATASETS used in machine learning
"""

from abc import ABC
from collections import OrderedDict
import os
from typing import Union, List

import numpy as np
import pandas as pd
import sqlite3

from .experiment_parameters import ExperimentFilter, DataFilter

COLOR_DICT = OrderedDict([
    ('Pentane', (69. / 255, 127. / 255, 181. / 255)),  # blue
    ('Hexane', (238. / 255, 124. / 255, 48. / 255)),  # orange
    ('Heptane', (96. / 255, 183. / 255, 96. / 255)),
    ('Octane', (192. / 255, 46. / 255, 38. / 255)),  # red
    ('Nonane', (174. / 255, 140. / 255, 205. / 255)),
    ('Decane', (168. / 255, 128. / 255, 119. / 255)),
    ('Ethanol', (233. / 255, 152. / 255, 209. / 255)),
    ('Water', (158. / 255, 158. / 255, 158. / 255)),
    ('Acetone', (204. / 255, 205. / 255, 89. / 255)),
    ('Toluene', (80. / 255, 206. / 255, 218. / 255)),
    ('Acetonitrile', (100. / 255, 180. / 255, 180. / 255))
])


class Dataset(OrderedDict):
    def __init__(self,
                 experiments: dict,
                 filters: Union[ExperimentFilter, DataFilter, List] = None,
                 **kwargs):
        super().__init__()
        for chem, experiment_ids in experiments.items():
            self.__setitem__(chem, experiment_ids)

        if filters is not None:
            self.filter(filters)

    def filter(self, filters: Union[ExperimentFilter, DataFilter, List]):
        if not hasattr(filter, '__iter__'): filters = [filters]
        for filter_ in filters:
            for chem, experiment_ids in self.items():
                self.__setitem__(chem, filter_(experiment_ids))


#------------------------------------------------------------#
# Specially curated datasets
#------------------------------------------------------------#


class PureCompoundsInTallCuvettes(Dataset):
    def __init__(self):
        """Experiments of pure compounds in tall cuvettes (8cm, 56-58mL)
        where Injection_Time = 15s, Injection_Rate = 6.0mL/min, and
        Injection_Volume = 1.3mL"""
        exp_set = OrderedDict([
            ('Pentane', [148, 174, 202, 203, 239, 240, 241, 434, 435]),
            ('Hexane',
             [175, 184, 185, 186, 187, 199, 204, 205, 424,
              425]),  # 149 new cuvette (measurement error?), 282 short cuvette?
            #('Cyclohexane', [155, 180]),
            ('Heptane', [150, 206,
                         207]),  # 176 new cuvette (measurement error?)
            ('Octane', [151, 177, 208, 209, 242, 243, 244, 283]),
            ('Nonane', [152, 178, 188, 189, 267]),
            ('Decane', [153, 179, 190, 191, 197, 268,
                        284]),  # 198 new decane (measurement error?)
            #('Dodecane', [192]),
            #('Methanol', [156, 172, 193, 196]),
            ('Ethanol', [157, 171, 324, 330, 440, 441,
                         442]),  # 325 measurement error?
            #('Isopropanol', [158, 173, 200, 201]),
            ('Water', [169, 170, 312, 313, 443, 444,
                       445]),  # 330 short cuvette?, 342 sucrose?
            ('Acetone', [159, 160, 326, 327, 436, 437, 438, 439]),
            ('Toluene', [154, 181, 446, 447, 448, 449, 450]),
            ('Acetonitrile', [163, 164, 452, 453, 454]),
            #('1-Butanol', [167, 168]),
            #('DCM', [161, 162])
        ])
        super().__init__(exp_set)


class OriginalSVCSet(Dataset):
    def __init__(self):
        """Experimental set used for fig2 by Tim and/or Sean"""
        exp_set = OrderedDict([
            ('Pentane', [239, 240, 241, 434, 435]),  #203,248
            ('Hexane', [204, 592, 593, 594, 571,
                        570]),  #130,185,138,139,184,185,141,199
            ('Heptane', [150, 206, 207, 576]),  #176,575,690
            ('Octane', [177, 208, 209, 243, 244]),  #242,155
            ('Nonane', [267, 188, 189]),  #,687,688,689,583,584,585,152
            ('Decane', [153, 179, 190, 191, 197, 280, 281]),
            #()'Methanol', [193,196,415]), #156,172,
            ('Ethanol', [157, 171, 324, 440, 441, 442]),  #325
            #()'Isopropanol', [158,173,200,201]),
            ('Water', [169, 170, 312, 313, 443, 444, 445]),
            ('Acetone', [327, 436, 437, 439]),  #438,160,326,
            ('Toluene', [154, 181, 446, 447, 448, 450]),  #449,
            ('Acetonitrile', [163, 453, 455, 456]),  #454,164,452
            ('1-Butanol', [167, 168, 677, 678, 680, 681]),
            ('DCM', [682, 683, 684, 685])  #,161,162,686
        ])
        super().__init__(exp_set)


class AllPureCompounds(Dataset):
    def __init__(self):
        """Experimental set including all experiments with pure compounds"""
        exp_set = OrderedDict([
            ('Pentane', [
                8, 9, 42, 43, 44, 45, 49, 50, 72, 73, 74, 75, 90, 91, 103, 104,
                120, 123, 125, 148, 174, 202, 203, 239, 240, 241, 261, 262, 306,
                309, 346, 352, 360, 361, 376, 380, 434, 435, 480, 489, 522, 523,
                524, 525, 526, 527, 562, 563, 564, 565, 566, 567, 622, 623
            ]),
            ('Hexane', [
                0, 1, 14, 15, 16, 37, 38, 39, 40, 41, 46, 47, 48, 53, 54, 55,
                56, 57, 58, 76, 77, 78, 79, 80, 92, 93, 105, 106, 114, 115, 116,
                117, 121, 122, 124, 126, 127, 128, 129, 130, 131, 132, 133, 134,
                135, 136, 137, 149, 175, 184, 185, 186, 187, 194, 199, 204, 205,
                282, 285, 286, 295, 296, 297, 300, 303, 347, 357, 377, 419, 424,
                425, 487, 490, 561, 568, 569, 570, 571, 592, 593, 594
            ]),
            ('Cyclohexane', [155, 180]),
            ('Heptane', [
                107, 108, 150, 176, 206, 207, 307, 348, 478, 481, 486, 488, 560,
                572, 573, 574, 575, 576, 690
            ]),
            ('Octane', [
                109, 151, 177, 208, 209, 242, 243, 244, 283, 287, 288, 293, 294,
                298, 301, 305, 343, 344, 345, 349, 356, 362, 363, 378, 385, 418,
                483, 485, 491, 492, 493, 497, 559, 577, 578, 579, 580
            ]),
            ('Nonane', [
                152, 178, 188, 189, 267, 308, 350, 358, 359, 479, 482, 494, 528,
                529, 532, 533, 535, 558, 581, 582, 583, 584, 585, 687, 688, 689
            ]),
            ('Decane', [
                119, 153, 179, 190, 191, 195, 197, 198, 268, 280, 281, 284, 289,
                290, 291, 292, 299, 302, 304, 351, 379, 417, 484, 495, 496, 530,
                531, 534, 536, 557, 586, 587, 588, 589, 624, 625, 626, 627, 628
            ]),
            ('Dodecane', [192]),
            ('Methanol', [113, 156, 172, 193, 196]),
            ('Ethanol', [
                111, 157, 171, 324, 325, 330, 420, 440, 441, 442, 759, 760, 761,
                762, 763, 764, 765, 766, 807, 808, 809, 810, 811, 837, 842, 843,
                847, 848, 849, 853, 854, 855, 862, 863, 864, 865, 866, 867, 908,
                909, 931, 932, 933, 934, 965, 966, 967, 986, 987, 988, 989, 990,
                991
            ]),  #325
            ('Isopropanol', [158, 173, 200, 201]),
            ('Water',
             np.concatenate(
                 ([112, 169, 170, 312, 313, 339, 342, 421, 443, 444, 445
                   ], np.arange(691, 710), np.arange(715, 750),
                  [801, 802, 803, 804, 805, 806], np.arange(825, 833), [
                      835, 836, 838, 839, 840, 841, 844, 845, 846, 850, 851,
                      852, 856, 857, 858, 859, 860, 861
                  ], np.arange(868, 879), np.arange(
                      911, 931), [992, 993, 994, 995, 996]),
                 axis=None)
             ),  #342 has sucrose, check 443 444 445, 710-714 crash on exp.main_spec_data.set_times(600)
            ('Acetone', [110, 159, 160, 326, 327, 436, 437, 438, 439]),
            ('Toluene', [154, 181, 423, 446, 447, 448, 449, 450]),
            ('Acetonitrile', [163, 164, 452, 453, 454, 455, 456]),
            ('1-Butanol', [167, 168, 677, 678, 680, 681]),
            ('DCM', [161, 162, 422, 682, 683, 684, 685, 686])
        ])
        super().__init__(exp_set)


#------------------------------------------------------------#
# Datasets pulled directly from database
#------------------------------------------------------------#


def _exclude_from_dataset():
    exp_nums = [
        200,
        201,
        202,  # all experiments with Procedure 19
        424,
        425,  # all experiments with Sensor_ID 18 and 19 (plasmonic, non-plasmonic Bragg stacks)
        177,
        190,
        221,  # issues at start
        176,
        198,
        203,
        882,
        956,
        958,
        340,  # complete outliers
        951,
        949,
        950,
        959,
        960,
        940,
        919,
        952,
        480,  # incredibly noisy
        #246, 248, 239, 148 # inconsistent
        #272, 274 # not sure
        339,  #probably wrong cuvette?
        522,
        523,
        524,
        525,
        526,
        527,  # wrong cuvette, 522-524 listed as 6cm, 525-527 listed as 1cm. Eq times match (good data except for mislabeled
        342,
        541,
        545,
        546,
        547,
        548,
        549,
        550  # not mole fraction measurements (also, gradient samples)
    ]

    return exp_nums


def _all_compounds(existing_CSV=True, **kwargs):
    """ all_data that fit kwargs:

    kwargs include:
        in_ : short_Cuvettes, medium_Cuvettes, tall_Cuvettes, other_Cuvettes
        with_ : SM30_Sensors, TM40_Sensors, hybrid_Sensors, func_Sensors

    Example: exp_set = all_data(in_ = tall_Cuvettes(), with_ = SM30_Sensors(), and_ = injection_Rate(6.0))
    """
    # connect to experiment libraries
    conn = sqlite3.connect("Library copy")
    conn2 = sqlite3.connect("Analytes.db")

    # create label dictionary
    label_dict = {}
    analytes_df = pd.read_sql_query('SELECT * FROM AnalyteData', conn2)

    for analyteID in analytes_df["Analyte ID"]:
        # create label
        n = 1
        new_label = []
        while n <= 4 and analytes_df.loc[analytes_df["Analyte ID"] == analyteID][
                "Concentration 4"].values[0] is not None and analytes_df.loc[
                    analytes_df["Analyte ID"] ==
                    analyteID]["Concentration 4"].values[0] > 0:
            new_label.append(
                str(analytes_df.loc[analytes_df["Analyte ID"] == analyteID][
                    "Component " + str(n)].values[0].capitalize()))
            n = n + 1
        label_dict[analyteID] = '-'.join(new_label)

    # update chem_list with IDs
    exp_df = pd.read_sql_query(
        "SELECT * FROM experiments WHERE Experiment_ID NOT IN (" +
        ', '.join(map(str, _exclude_from_dataset())) + ")", conn)
    #exp_set = OrderedDict([('data',list(exp_df.Experiment_ID))])

    if existing_CSV == True:
        # make sure files are in CSV directory
        filenames = os.listdir('../fra_exp_csv')
        existing_IDs = [
            int(os.path.splitext(name)[0]) for name in filenames[0:-1]
        ]
        if bool(set(list(
                exp_df.Experiment_ID)).difference(existing_IDs)) == True:
            print(
                str(set(list(exp_df.Experiment_ID)).difference(existing_IDs)) +
                " are not in CSV database")

        exp_IDs = list(
            set(existing_IDs).intersection(list(exp_df.Experiment_ID)))
        #exp_set = OrderedDict([('data',list(set(existing_IDs).intersection(list(exp_df.Experiment_ID))))])
    else:
        exp_IDs = list(exp_df.Experiment_ID)
        #exp_set = OrderedDict([('data',list(exp_df.Experiment_ID))])

    # create exp_set
    exp_set = OrderedDict()
    for key in sorted(label_dict.values()):
        exp_set[key] = []
    for ID in exp_IDs:
        ID_df = pd.read_sql_query(
            'SELECT "Analyte ID" FROM ExperimentData WHERE "Experiment Number" IS '
            + str(ID), conn2)
        if ID_df["Analyte ID"][0] != None:
            exp_set[label_dict[ID_df["Analyte ID"][0]]].append(ID)

    # apply kwargs subset procedures
    for subset in kwargs.values():
        exp_set = subset.select(exp_set)

    return exp_set


def _pure_compounds(chem_list=None, **kwargs):
    """
    pure_compounds imports all experiment IDs from Library copy that fit the parameters in kwargs:

    kwargs include:
        in_ : short_Cuvettes, medium_Cuvettes, tall_Cuvettes, other_Cuvettes
        with_ : SM30_Sensors, TM40_Sensors, hybrid_Sensors, func_Sensors

    Example: exp_set = pure_compounds('Pentane', in_ = tall_Cuvettes(), with_ = SM30_Sensors(), and_ = injection_Rate(6.0))
    """
    # connect to experiment library
    conn = sqlite3.connect("Library copy")

    # update chem_list with IDs
    if chem_list == None:  # get all chemicals in dataset
        df = pd.read_sql_query("SELECT * FROM chemicals", conn)
        chem_list = [str(name).capitalize() for name in df.Name]
    elif not type(chem_list) == list:
        chem_list = [chem_list]

    # retrieve dataset and create exp_set
    exp_set = OrderedDict()
    for chem in chem_list:
        # get chemical ID from library
        id_df = pd.read_sql_query(
            "SELECT * FROM chemicals WHERE Name LIKE '%" + chem + "%'", conn)
        chem_ID = list(id_df.Chemical_ID)
        # get experiment IDs from library
        exp_df = pd.read_sql_query(
            "SELECT * FROM experiments WHERE Experiment_ID IN (SELECT Experiment_ID FROM concentrations WHERE Molar_Fraction = 1.0 AND Chemical_ID = "
            + str(chem_ID[0]) + ") AND Experiment_ID NOT IN (" +
            ', '.join(map(str, _exclude_from_dataset())) + ")", conn)
        exp_set[chem] = list(exp_df.Experiment_ID)

    # apply kwargs subset procedures
    for subset in kwargs.values():
        exp_set = subset.select(exp_set)

    return exp_set


def _binary_mixtures(chem_list=None, **kwargs):
    """
    binary_mixtures imports all experiment IDs from Library copy that contain the chemicals in chem_list and fit the parameters in kwargs:

    kwargs include:
        in_ : short_Cuvettes, medium_Cuvettes, tall_Cuvettes, other_Cuvettes
        with_ : SM30_Sensors, TM40_Sensors, hybrid_Sensors, func_Sensors

    Example: exp_set = binary_mixtures(['Pentane', 'Hexane'], in_ = tall_Cuvettes(), with_ = SM30_Sensors(), and_ = injection_Rate(6.0))
    """
    # connect to experiment library
    conn = sqlite3.connect("Library copy")
    conn2 = sqlite3.connect("Analytes.db")

    # check that chem list contains mixture
    if len(chem_list) != 2:  # abort
        print("The chem_list should contain two compounds.")
        return False

    # collect list of analyte IDs
    #query analyte IDs for the chem1 and chem2
    pure_df = pd.read_sql_query(
        'SELECT "Analyte ID", "Component 1", "Concentration 1", "Component 2", "Concentration 2" FROM AnalyteData WHERE ("Component 1" LIKE "'
        + chem_list[0] +
        '" AND "Component 2" IS NULL) OR ("Component 1" LIKE "' + chem_list[1] +
        '" AND "Component 2" IS NULL)', conn2)
    mixtures_df = pd.read_sql_query(
        'SELECT "Analyte ID", "Component 1", "Concentration 1", "Component 2", "Concentration 2" FROM AnalyteData WHERE ("Component 1" LIKE "'
        + chem_list[0] + '" AND "Component 2" LIKE "' + chem_list[1] +
        '") OR ("Component 1" LIKE "' + chem_list[1] +
        '" AND "Component 2" LIKE "' + chem_list[0] + '")', conn2)

    #query experiment IDs for analytes
    exp_df = pd.read_sql_query(
        'SELECT "Experiment Number", "Analyte ID" FROM ExperimentData WHERE "Analyte ID" IN ('
        + ', '.join(map(str, pure_df["Analyte ID"])) + ', ' +
        ', '.join(map(str, mixtures_df["Analyte ID"])) +
        ') AND "Experiment Number" NOT IN (' +
        ', '.join(map(str, _exclude_from_dataset())) + ')', conn2)

    # collect list of experiment IDs
    #query experimental parameters for all experiments in exp_set
    #exp_df2 = pd.read_sql_query("SELECT Experiment_ID, Name, Description, Sensor_ID, Chamber_ID, Relative_Humidity_Percent, Procedure, Experimenter_ID, Injection_Time_s, Injection_Rate_mL_per_min, Injection_Volume_mL FROM experiments WHERE Experiment_ID IN (" + ', '.join(map(str, exp_df["Experiment Number"])) + ") AND Experiment_ID NOT IN (" + ', '.join(map(str, _exclude_from_dataset())) + ")", conn)

    # sort concentrations in dataset
    list(pure_df["Analyte ID"]) + list(mixtures_df["Analyte ID"])
    pure_df.append(mixtures_df)
    mixtures_sorted_df = mixtures_df.sort_values(by=['Concentration 1'],
                                                 inplace=False)

    zero_df = pure_df[pure_df['Component 1'] == chem_list[1].lower()]
    one_df = pure_df[pure_df['Component 1'] == chem_list[0].lower()]

    sorted_df = zero_df.append(mixtures_sorted_df).append(one_df)
    sorted_df = sorted_df.reset_index()

    # make exp_set
    from collections import OrderedDict

    exp_set = OrderedDict()

    for n, analyte in enumerate(list(sorted_df["Analyte ID"])):
        if chem_list[0].lower() == sorted_df["Component 1"][n]:
            concentration = sorted_df["Concentration 1"][n]
        elif chem_list[0].lower() == sorted_df["Component 2"][n]:
            concentration = sorted_df["Concentration 2"][n]
        else:
            concentration = 0

        exps_at_concentration = exp_df[exp_df['Analyte ID'] == analyte]
        try:
            exp_set[float(concentration)] + list(
                exps_at_concentration["Experiment Number"])
        except:
            exp_set[float(concentration)] = list(
                exps_at_concentration["Experiment Number"])

    # apply kwargs subset procedures
    for subset in kwargs.values():
        exp_set = subset.select(exp_set)

    # remove empty sets
    for key, values in exp_set.items():
        if not values:
            exp_set.pop(key)

    return OrderedDict(sorted(exp_set.items()))


class AllCompounds(Dataset):
    def __init__(self):
        exp_set = _all_compounds()
        super().__init__(exp_set)


class PureCompounds(Dataset):
    def __init__(self):
        exp_set = _pure_compounds()
        super().__init__(exp_set)


class BinaryMixtures(Dataset):
    def __init__(self):
        exp_set = _binary_mixtures()
        super().__init__(exp_set)


#------------------------------------------------------------#
# Experimental set including all experiments before Mar 20 2019
#------------------------------------------------------------#


class AllExperimentsMar20_2019(Dataset):
    def __init__(self):
        exp_set = list(
            np.concatenate((np.arange(0, 26), np.arange(
                37, 679), np.arange(680, 795), np.arange(
                    796, 870), 795, np.arange(870, 968), np.arange(986, 997)),
                           axis=None))  # np.arange(26,37),
        super().__init__(exp_set)


#------------------------------------------------------------#
# Gradient experimental sets including all experiments before Mar 20 2019
#------------------------------------------------------------#


class AllGradients(Dataset):
    def __init__(self):
        exp_set = OrderedDict([
            ('C5C6 1',
             np.concatenate([
                 np.arange(0, 26),
                 np.arange(46, 55),
                 np.arange(55, 105),
                 np.arange(55, 105)
             ],
                            axis=None)),  # np.arange(37,46) pure compounds
            ('incomplete', [])
        ])
        super().__init__(exp_set)


class COOPGradients(Dataset):
    def __init__(self):
        """
        C5C6       : Pentane-Hexane, tall cuvettes, SM30 sensor, 0.4mL injected at 4.0s, at 6.0mL/min
        C5C8 1     : Pentane-Octane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s at 6.0mL/min
        C5C8 2     : Pentane-Octane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s at 6.0mL/min
        C5C8 3     : Pentane-Octane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s at 6.0mL/min
        EtDIW      : Ethanol-Water on 13F, tall cuvette, TM40 and SM30 sensor, 0.85mL injected, at 0.15s at 6.0mL/min
        EtDIW low  : Ethanol-Water, tall cuvette, SM30 sensor, 1.3mL injected, at 15s at 6.0mL/min
        EtDIW high : Ethanol-Water, tall cuvette, SM30 sensor, 1.3mL injected, at 15s at 6.0mL/min
        EtDIW 13F  : Ethanol-Water on 13F, tall cuvette, TM40 and SM30 sensor, 0.85mL injected, at 0.15s at 6.0mL/min
        EtDIW Whitesides: Ethanol-Water on gradient 13F, tall cuvette, TM40 sensor, 0.85mL injected, at 0.15s at 6.0mL/min
        """
        exp_set = OrderedDict([
            ('C5C6', [592, 593, 594] + [595, 597] + [598, 599, 600] +
             [602, 603] + [605, 606] + [607, 608, 609] + [610, 611, 629] +
             [612, 613, 614] + [615, 616, 617] + [618, 619, 620] +
             [623]),  # 601, 604, 621, 622 # 629-2.0s injection
            ('C5C8 1', [239, 275] + range(241, 246) + range(247, 248) +
             [208, 209, 283, 174, 203] + [
                 216, 217, 273, 218, 219, 220, 230, 272, 274, 279, 233, 236,
                 434, 435
             ] + range(386, 401)),
            ('C5C8 2', [242, 243, 244, 208, 209, 283] + [386, 387, 388] +
             [245, 247] + [220, 230] + [389, 390, 391] + [236] +
             [218, 219, 392, 393, 394] + [272] + [233] + [274, 395, 396, 397] +
             [216, 217] + [274] + [279, 398, 399, 400] + [275] +
             [239, 241, 174, 203, 434, 435]),  # 221
            ('C5C8 3',
             range(241, 246) + range(247, 248) + [208, 209, 283, 174, 203] + [
                 216, 217, 273, 218, 219, 220, 230, 272, 274, 279, 233, 236,
                 434, 435
             ] + range(386, 401)),
            ('EtDIW', [877, 878, 879] + [880, 881, 882] + [883, 884, 885] +
             [886, 887, 910] + [888, 889, 890, 916, 917, 918] +
             [891, 892, 893, 919, 920] + [894, 895, 896, 921] +
             [897, 898, 899] + [900, 901, 902, 903] + [904, 905, 906] +
             [907, 908, 909]),  # might be on 13F too
            ('EtDIW low', [170, 444, 445] + [476, 477] + [474] +
             [466, 467, 469] + [468, 471] + [457, 458, 459] + [460, 461, 462] +
             [463, 464, 465] + [401, 402]),  #475,472,473,470
            ('EtDIW high',
             [170, 444, 445] + [401, 402] + [315] + [403, 404] + [316, 317] +
             [405, 406] + [318, 319] + [320, 321] + [407, 408] + [322, 323] +
             [324, 325, 440, 441, 442]),  #314, added later: 322, 325, 441
            ('EtDIW 13F', [878, 879] + [881] + [884] + [917] + [891, 920] +
             [921] + [898, 899] + [900, 903] + [904, 905] + [907, 908]),
            (
                'EtDIW Whitesides', [935] + [939] + [941, 943] +
                [944, 945, 946] + [947, 948] + [949, 950, 951] + [954] + [955] +
                [958] + [961, 963] + [964, 965, 966]
            )  #936,937, 938,939, 942,943, 949, 952,953,954, 956,957, 959,960, 926,963
        ])
        super().__init__(exp_set)


class AllPentaneHexane(Dataset):
    def __init__(self):
        """
        C5C6 1: Pentane-Hexane, tall cuvettes, SM30 sensor, 0.4mL injected at 4.0s, at 6.0mL/min
        C5C6 2: Pentane-Hexane, short cuvettes, Ida sensor, 1.15mL injected at 15.0s, at 6.0mL/min
        C5C6 3: Pentane-Hexane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C5C6 4: Pentane-Hexane, short cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        """
        exp_set = OrderedDict([
            ('C5C6 1', [592, 593, 594] + [595, 597] + [598, 599, 600] +
             [602, 603] + [605, 606] + [607, 608, 609] + [610, 611, 629] +
             [612, 613, 614] + [615, 616, 617] + [618, 619, 620] + [623]
             ),  # 606, 621, 622 # 629-2.0s injection # range(592,623) + [629])
            ('C5C6 2', range(46, 55) + range(55, 105)),
            ('C5C6 3', range(211, 216) + [232, 234]),  # 233
            ('C5C6 4', [259, 260])
        ])
        super().__init__(exp_set)


class AllPentaneHeptane(Dataset):
    def __init__(self):
        """
        C5C7 1: Pentane-Heptane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C5C7 2: Pentane-Heptane, short cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        """
        exp_set = OrderedDict([('C5C7 1', range(222, 226) + [
            148, 150, 174, 176, 202, 203, 206, 207, 239, 240, 241, 434, 435,
            560, 562, 563, 564, 565
        ]), ('C5C7 2', range(263, 265) + [261, 262])])
        super().__init__(exp_set)


class AllPentaneOctane(Dataset):
    def __init__(self):
        """
        C5C8 1: Pentane-Octane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C5C8 2: Pentane-Octane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C5C8 3: Pentane-Octane, medium cuvettes, TM40 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C5C8 4: Pentane-Octane, short cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C5C8 5: Pentane-Octane, medium cuvettes, TM40 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C5C8 6: Pentane-Octane, tall cuvettes, Hybrid sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C5C8 7: Pentane-Octane, tall cuvettes, Sm30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        """
        exp_set = OrderedDict([
            ('C5C8 1', [239, 275] + range(241, 246) + range(247, 248) +
             [208, 209, 283, 174, 203] +
             [216, 217, 273, 218, 219, 220, 230, 272, 274, 233, 236, 434, 435] +
             range(386, 401)),  #221, 246, 248,
            ('C5C8 2', range(239, 248) + range(248, 251) + range(251, 257)),
            ('C5C8 3', range(352, 357)),  #352 tall cuvette or typo?
            ('C5C8 4', [257, 258, 279]),
            ('C5C8 5', range(360, 376)),
            ('C5C8 6', range(380, 386)),
            ('C5C8 7', range(386, 401))
        ])
        super().__init__(exp_set)


class AllPentaneNonane(Dataset):
    def __init__(self):
        """
        C5C9 1: Pentane-Nonane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C5C9 2: Pentane-Nonane, short cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        """
        exp_set = OrderedDict([('C5C9 1', [269, 270]), ('C5C9 2', [276, 278])])
        super().__init__(exp_set)


class AllPentaneDecane(Dataset):
    def __init__(self):
        """
        C5C10 1: Pentane-Decane, short cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        """
        exp_set = OrderedDict([('C5C10 1', [265, 266])])
        super().__init__(exp_set)


class AllPentaneDodecane(Dataset):
    def __init__(self):
        """
        C5C12 1: Pentane-Dodecane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C5C12 2: Pentane-Dodecane, short cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        """
        exp_set = OrderedDict([('C5C12 1', [271]), ('C5C12 2', [277])])
        super().__init__(exp_set)


class AllHexaneOctane(Dataset):
    def __init__(self):
        """
        C6C8 1: Hexane-Octane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        C6C8 2: Hexane-Octane, short cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        """
        exp_set = OrderedDict([
            ('C6C8 1', range(226, 230) + [231, 235]),
            ('C6C8 2', [231, 235]),
        ])
        super().__init__(exp_set)


class AllHexaneAcetone(Dataset):
    def __init__(self):
        """
        C6Ac 1: Pentane-Decane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        """
        exp_set = OrderedDict([
            ('C6Ac 1', range(326, 330) + range(331, 339)),
        ])
        super().__init__(exp_set)


class AllEthanolWater(Dataset):
    def __init__(self):
        """
        EtDIW 1: Ethanol-Water, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        EtDIW 2: Ethanol-Water, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min # shitty sensor ID 20
        EtDIW 3: Ethanol-Water, tall cuvettes, SM30 sensor, 0.85mL injected at 10.0s, at 6.0mL/min
        EtDIW 4: Ethanol-Water on 13F, tall cuvettes, SM30 sensor, 0.85mL injected at 0.15s, at 6.0mL/min
        EtDIW 5: Ethanol-Water on 13F, tall cuvettes, TM40 sensor, 0.85mL injected at 0.15s, at 6.0mL/min
        EtDIW 4: Ethanol-Water on gradient, tall cuvettes, TM40 sensor, 0.85mL injected at 0.15s, at 6.0mL/min
        """
        exp_set = OrderedDict([
            ('EtDIW 1', range(314, 326) + [340] + range(401, 409)),
            ('EtDIW 2', range(457, 478)),
            ('EtDIW 3', range(759, 801)),
            (
                'EtDIW 4', range(877, 888)
            ),  # compare to EtDIW 5, because they follow seemlessly, might be mistake
            ('EtDIW 5', range(888, 911) + range(916, 925)),
            ('EtDIW 6', range(935, 968))
        ])
        super().__init__(exp_set)


class EthanolWaterGradients(Dataset):
    def __init__(self, surface='glass'):
        """
        surface: glass, 13F, gradient
        glass: tall cuvettes, SM30 sensor, injected 1.3mL at 15s at 6mL/min
        13F: tall cuvettes, SM30(734-887) or TM40 (888-924), injected 0.85mL at 10s (734-841) or 0.15s (877-924) at 6mL/min
        gradient: tall cuvette, TM40, injected 0.85mL, at 0.15s at 6mL/min
        """
        datasets = {
            'glass':
            OrderedDict([
                (0., range(312, 314) + [169, 170] + [691] + range(443, 446)),
                (0.000001, range(475, 478)),
                (0.00001, range(472, 475)),
                (0.0001, [466, 467, 469]),
                (0.001, [468, 470, 471]),
                (0.01, range(457, 460)),
                (0.02, range(460, 463)),
                (0.05, range(463, 466)),
                (0.2, [314, 315]),
                (0.25, [401, 402]),
                (0.4, [316, 317, 340]),
                (0.5, [426, 427, 403, 404]),
                (0.6, [318, 319]),
                (0.75, [405, 406]),
                (0.8, [320, 321]),
                (0.89, [322, 323]),
                (0.95, [407, 408]),
                (1.0, [324, 325] + [440, 441] + [171, 330]),
            ]),
            '13F':
            OrderedDict([
                (0., range(734, 737) + [746] + range(838, 842) +
                 range(877 - 880) + range(910, 916)),
                (0.1, range(880, 883)),
                (0.2, range(883, 886)),
                (0.25, range(790, 793)),
                (0.3, range(886, 888) + [910]),
                (0.4, range(888, 891) + range(916, 919)),
                (0.5, range(767, 770) + range(891, 894) + [919, 920]),
                (0.6, range(894, 897) + [921]),
                (0.7, range(780, 783) + range(897, 900)),
                (0.75, range(796, 798)),
                (0.8, range(900, 904) + range(922, 925)),
                (0.9, range(904, 907)),
                (1.0, [750, 751, 763, 764] + range(907, 910)),
            ]),
            'gradient':
            OrderedDict([
                (0., range(935, 938)),
                (0.1, range(938, 941)),
                (0.2, range(941, 944)),
                (0.3, range(944, 947)),
                (0.4, range(947, 950)),
                (0.5, range(950, 953)),
                (0.6, range(953, 956)),
                (0.7, range(956, 959)),
                (0.8, range(959, 962)),
                (0.9, range(962, 965)),
                (1.0, range(965, 968)),
            ])
        }
        super().__init__(datasets[surface])


class AllMethanolWater(Dataset):
    def __init__(self):
        """
        MeDIW 1: Methanol-Water, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        """
        exp_set = OrderedDict([
            ('MeDIW 1', [341] + range(410, 417)),
        ])
        super().__init__(exp_set)


class TernaryGradients(Dataset):  # add binary + unary references
    def __init__(self):
        """
        C5C6C8: Pentane-Hexane-Octane, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        EtMeDIW: Ethanol-Methanol-Water, tall cuvettes, SM30 sensor, 1.3mL injected at 15.0s, at 6.0mL/min
        """
        exp_set = OrderedDict([('C5C6C8', range(231, 239)),
                               ('EtMeDIW', range(426, 434))])
        super().__init__(exp_set)


#------------------------------------------------------------#
# Different surfaces
#------------------------------------------------------------#


class COOPSurfaces(Dataset):  # need control later
    def __init__(self):
        exp_set = OrderedDict([
            ('DIWonSi', [744, 758] + [830, 831, 832] +
             [870, 871, 872, 873, 874, 875, 876] + [911, 912, 913, 914, 915]),
            ('Sivs13F', [758, 871, 873, 915] + [775, 776, 777] + [778, 779] +
             [745, 760] + [878, 879] + [891, 920] + [898, 899] + [907, 908]),
            ('Sivs13F 2', [170, 406, 324] + [915, 776, 760] + [879, 891, 907])
        ])
        super().__init__(exp_set)

from abc import ABC
from collections import namedtuple
from pathlib import Path
import sqlite3
from typing import Union, Tuple, List

import pandas as pd

ANALYTES_DB = Path(__file__).parent.joinpath('Analytes.db')
LIBRARY_DB = Path(__file__).parent.joinpath('Library.db')

CHAMBER_IDS = {
    'short': [4, 6, 9, 15],
    'medium': [10, 11, 14],
    'tall': [1, 2, 5, 7, 8, 12, 13, 16, 20, 26],
    'other': [
        3, 17, 18, 19, 21, 22, 23, 24
    ]  # 3 is a glass vial, 17,18,19 are Teflon (narrow, med, wide), 21 is Talcon, 22,23,24 Teflon
}

SENSOR_IDS = {
    'SM30_Sensor': [1, 6, 10, 11, 16, 18, 19, 20, 21, 23],
    'TM40_Sensor': [12, 13, 14, 22],
    'Hybrid_Sensor': [15, 17],
    'Ida_Sensor': [2, 3, 4, 5],
    'func_Sensor': [7, 8, 9]
}

EXPERIMENTAL_MEASUREMENTS = [
    'Vapor Pressure', 'Boiling Point', 'Flash Point', 'Viscosity'
]


class ExperimentParameter(ABC):
    sql_label: str
    value: list

    @property
    def sql_query(self) -> str:
        return f"{self.sql_label} IN ({', '.join(map(str, self.value))})"

    def _sql_query_for_floats_and_ranges(self) -> str:
        try:
            return f"{self.sql_label} BETWEEN {str(self.value[0])} AND {str(self.value[1])}"
        except (IndexError, TypeError):
            return f"{self.sql_label} = {str(self.value)}"


class Chamber(ExperimentParameter):
    sql_label = "Chamber_ID"

    def __init__(self, size: str) -> None:
        if not size in CHAMBER_IDS:
            raise ValueError(f"Invalid parameter: {size}")
        self.value = CHAMBER_IDS[size]


class Sensor(ExperimentParameter):
    sql_label = "Sensor_ID"

    def __init__(self, kind: str) -> None:
        if not kind in SENSOR_IDS:
            raise ValueError(f"Invalid parameter: {kind}")
        self.value = SENSOR_IDS[kind]


class InjectionTime(ExperimentParameter):
    sql_label = "Injection_Time_s"

    def __init__(self, time: Union[float, Tuple[float, float]]) -> None:
        self.value = time

    @property
    def sql_query(self) -> str:
        return self._sql_query_for_floats_and_ranges()


class InjectionRate(ExperimentParameter):
    sql_label = "Injection_Rate_mL_per_min"

    def __init__(self, rate: Union[float, Tuple[float, float]]) -> None:
        self.value = rate

    @property
    def sql_query(self) -> str:
        return self._sql_query_for_floats_and_ranges()


class InjectionVolume(ExperimentParameter):
    sql_label = "Injection_Volume_mL"

    def __init__(self, volume: Union[float, Tuple[float, float]]) -> None:
        self.value = volume

    @property
    def sql_query(self) -> str:
        return self._sql_query_for_floats_and_ranges()


class Custom(ExperimentParameter):
    def __init__(self, query: str) -> None:
        self.value = query

    def sql_query(self) -> str:
        return self.value


class ExperimentFilter(object):
    def __init__(self, *parameters: ExperimentParameter) -> None:
        self.conn = sqlite3.connect(LIBRARY_DB)
        self.parameters = parameters

    def __call__(self, items: Union[list, set]) -> list:
        query_base = f"""SELECT Experiment_ID FROM experiments
                        WHERE Experiment_ID IN ({', '.join(map(str, items))})"""
        parameter_queries = [
            parameter.sql_query for parameter in self.parameters
        ]

        sub_df = pd.read_sql_query(
            ' AND '.join([query_base] + parameter_queries), self.conn)

        return list(set(items) & set(sub_df.Experiment_ID))


class DataFilter(object):
    """Removes compounds for which experimental data is not available.

    parameters: Vapor Pressure, Boiling Point, Flash Point, Viscosity"""
    def __init__(self, *parameters: str):
        for parameter in parameters:
            if parameter not in EXPERIMENTAL_MEASUREMENTS:
                raise ValueError(
                    f"Parameter must be in: {EXPERIMENTAL_MEASUREMENTS}")
        self.conn = sqlite3.connect(ANALYTES_DB)
        self.parameters = parameters

    def __call__(self, items: Union[list, set]) -> list:
        query = f"""SELECT "Experiment Number" FROM ExperimentData
                    WHERE "Experiment Number" IN ({', '.join(map(str, items))})
                    AND "Analyte ID" IN
                        (SELECT "Analyte ID" FROM AnalyteData
                        WHERE %s)
                    """
        query = query % " AND ".join([
            f"""'{parameter}' IS NOT null)""" for parameter in self.parameters
        ])

        exp_id_df = pd.read_sql_query(query, self.conn)

        return list(set(items) & set(exp_id_df['Experiment Number']))


def _create_experiment_label_df(experiments):
    conn = sqlite3.connect(ANALYTES_DB)

    experiments_df = pd.read_sql_query(
        f"""SELECT "Experiment Number", "Analyte ID" FROM ExperimentData
            WHERE "Experiment Number" IN ({', '.join(map(str, experiments))})""",
        conn)

    label_df = pd.read_sql_query(
        f"""SELECT * FROM AnalyteData WHERE "Analyte ID"
                IN ({",".join(experiments_df["Analyte ID"].astype(str))})""",
        conn)

    experiment_label_df = experiments_df.merge(
        label_df, 'left', on='Analyte ID').set_index('Experiment Number')
    return experiment_label_df


def create_physical_property_label(experiments, *properties: str):
    """Get physical property label.

    Args:
        experiments (list): List of experiment indices
        *properties (str): Vapor Pressure, Boiling Point, Flash Point, Viscosity

    Returns:
        New labels as namedtuple
    """
    if len(properties) < 1:
        raise ValueError("Must pass at least one property")
    properties = [property.title() for property in properties]

    Label = namedtuple("Properties",
                       ["_".join(property.split()) for property in properties])

    experiments_label_df = _create_experiment_label_df(experiments)

    labels = {}
    for exp_id in experiments:
        labels[exp_id] = Label(*experiments_label_df.loc[exp_id, properties])

    return labels


def create_concentration_label(experiments):
    """Get physical property label.

    Args:
        experiments (list): List of experiment indices

    Returns:
        New labels as namedtuple
    """
    experiments_label_df = _create_experiment_label_df(experiments)

    labels = {}
    for exp_id in experiments:
        components = experiments_label_df.loc[
            exp_id,
            sorted([
                column for column in experiments_label_df.columns
                if 'Component' in column
            ])].dropna()
        concentrations = experiments_label_df.loc[
            exp_id,
            sorted([
                column for column in experiments_label_df.columns
                if 'Concentration' in column
            ])]

        Label = namedtuple(
            "Concentration",
            ["_".join(component.split()) for component in components])
        labels[exp_id] = Label(*concentrations[:len(components)])

    return labels

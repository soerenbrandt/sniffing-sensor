import sqlite3
from abc import ABC
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd

ANALYTES_DB = Path(__file__).parent.joinpath("databases", "Analytes.db")
LIBRARY_DB = Path(__file__).parent.joinpath("databases", "Library.db")


EXPERIMENTAL_MEASUREMENTS = [
    "Vapor Pressure",
    "Boiling Point",
    "Flash Point",
    "Viscosity",
]


class Chamber(Enum):
    SHORT = [4, 6, 9, 15]
    MEDIUM = [10, 11, 14]
    TALL = [1, 2, 5, 7, 8, 12, 13, 16, 20, 26]

    # Other:
    #  3 is a glass vial,
    #  17,18,19 are Teflon (narrow, med, wide),
    #  21 is Talcon,
    #  22,23,24 Teflon
    OTHER = [3, 17, 18, 19, 21, 22, 23, 24]

    @property
    def sql_query(self):
        return f"Chamber_ID IN ({', '.join(map(str, self.value))})"

    @classmethod
    def from_id(cls, value):
        for e in cls:
            if value in e.value:
                return e


class Sensor(Enum):
    SM30 = [1, 6, 10, 11, 16, 18, 19, 20, 21, 23]
    TM40 = [12, 13, 14, 22]
    HYBRID = [15, 17]
    IDA = [2, 3, 4, 5]
    FUNCTIONALIZED = [7, 8, 9]

    @property
    def sql_query(self):
        return f"Sensor_ID IN ({', '.join(map(str, self.value))})"

    @classmethod
    def from_id(cls, value):
        for e in cls:
            if value in e.value:
                return e


class ExperimentParameter(ABC):
    sql_label: str

    def __init__(self, *values: float, value_range: List[float] = None) -> None:
        if value_range and not len(value_range) == 2:
            raise ValueError("value_range must contain two values.")
        self.values = values
        self.range = value_range

    @property
    def sql_query(self) -> str:
        query_parts = []
        if self.values:
            query_parts.append(
                f"{self.sql_label} IN ({', '.join(map(str, self.values))})"
            )
        if self.range:
            query_parts.append(
                f"{self.sql_label} BETWEEN {str(self.range[0])} AND {str(self.range[1])}"
            )
        return " AND ".join(query_parts)


class InjectionTime(ExperimentParameter):
    sql_label = "Injection_Time_s"


class InjectionRate(ExperimentParameter):
    sql_label = "Injection_Rate_mL_per_min"


class InjectionVolume(ExperimentParameter):
    sql_label = "Injection_Volume_mL"


class Custom(ExperimentParameter):
    def __init__(self, label: str, **kwargs) -> None:
        self.sql_label = label
        super().__init__(**kwargs)


class ExperimentFilter(object):
    def __init__(
        self, *parameters: Union[ExperimentParameter, Chamber, Sensor]
    ) -> None:
        self.conn = sqlite3.connect(LIBRARY_DB)
        self.parameters = parameters

    def __call__(self, items: Union[list, set]) -> list:
        query_base = f"""SELECT Experiment_ID FROM experiments
                        WHERE Experiment_ID IN ({', '.join(map(str, items))})"""
        parameter_queries = [
            parameter.sql_query for parameter in self.parameters
        ]

        sub_df = pd.read_sql_query(
            " AND ".join([query_base] + parameter_queries), self.conn
        )

        return list(set(items) & set(sub_df.Experiment_ID))


class DataFilter(object):
    """Removes compounds for which experimental data is not available.

    parameters: Vapor Pressure, Boiling Point, Flash Point, Viscosity"""

    def __init__(self, *parameters: str):
        for parameter in parameters:
            if parameter not in EXPERIMENTAL_MEASUREMENTS:
                raise ValueError(
                    f"Parameter must be in: {EXPERIMENTAL_MEASUREMENTS}"
                )
        self.conn = sqlite3.connect(ANALYTES_DB)
        self.parameters = parameters

    def __call__(self, items: Union[list, set]) -> list:
        query = f"""SELECT "Experiment Number" FROM ExperimentData
                    WHERE "Experiment Number" IN ({', '.join(map(str, items))})
                    AND "Analyte ID" IN
                        (SELECT "Analyte ID" FROM AnalyteData
                        WHERE %s)
                    """
        query = query % " AND ".join(
            [f"""'{parameter}' IS NOT null)""" for parameter in self.parameters]
        )

        exp_id_df = pd.read_sql_query(query, self.conn)

        return list(set(items) & set(exp_id_df["Experiment Number"]))


def _create_experiment_label_df(experiments):
    conn = sqlite3.connect(ANALYTES_DB)

    experiments_df = pd.read_sql_query(
        f"""SELECT "Experiment Number", "Analyte ID" FROM ExperimentData
            WHERE "Experiment Number" IN ({', '.join(map(str, experiments))})""",
        conn,
    )

    label_df = pd.read_sql_query(
        f"""SELECT * FROM AnalyteData WHERE "Analyte ID"
                IN ({",".join(experiments_df["Analyte ID"].astype(str))})""",
        conn,
    )

    experiment_label_df = experiments_df.merge(
        label_df, "left", on="Analyte ID"
    ).set_index("Experiment Number")
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

    Label = namedtuple(
        "Properties", ["_".join(property.split()) for property in properties]
    )

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
            sorted(
                [
                    column
                    for column in experiments_label_df.columns
                    if "Component" in column
                ]
            ),
        ].dropna()
        concentrations = experiments_label_df.loc[
            exp_id,
            sorted(
                [
                    column
                    for column in experiments_label_df.columns
                    if "Concentration" in column
                ]
            ),
        ]

        Label = namedtuple(
            "Concentration",
            ["_".join(component.split()) for component in components],
        )
        labels[exp_id] = Label(*concentrations[: len(components)])

    return labels

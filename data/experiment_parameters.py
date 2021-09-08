from abc import ABC
import sqlite3
from typing import Union, Tuple, List

import pandas as pd

CHAMBER_IDS = {
    'short': [4, 6, 9, 15],
    'medium': [10, 11, 14],
    'tall': [1, 2, 5, 7, 8, 12, 13, 16, 20, 26],
    'other': [
        3, 17, 18, 19, 21, 22, 23, 24
    ]  # 3 is a glass vial, 17,18,19 are Teflon (narrow, med, wide), 21 is Talcon, 22,23,24 Teflon
}

SENSOR_IDS = {
    'SM30_Sensors': [1, 6, 10, 11, 16, 18, 19, 20, 21, 23],
    'TM40_Sensors': [12, 13, 14, 22],
    'Hybrid_Sensors': [15, 17],
    'Ida_Sensors': [2, 3, 4, 5],
    'func_Sensors': [7, 8, 9]
}

EXPERIMENTAL_MEASUREMENTS = [
    'Vapor Pressure', 'Boiling Point', 'Flash Point', 'Viscosity'
]


class ExperimentParameter(ABC):
    sql_label: str
    value: list

    def sql_query(self) -> str:
        return f"{self.sql_label} IN ({', '.join(map(str, self.IDs))})"

    def _sql_query_for_floats_and_ranges(self) -> str:
        try:
            return f"{self.sql_label} BETWEEN {str(self.value[0])} AND {str(self.value[1])}"
        except IndexError:
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

    def sql_query(self) -> str:
        return self._sql_query_for_floats_and_ranges


class InjectionRate(ExperimentParameter):
    sql_label = "Injection_Rate_mL_per_min"

    def __init__(self, rate: Union[float, Tuple[float, float]]) -> None:
        self.value = rate

    def sql_query(self) -> str:
        return self._sql_query_for_floats_and_ranges


class InjectionVolume(ExperimentParameter):
    sql_label = "Injection_Volume_mL"

    def __init__(self, volume: Union[float, Tuple[float, float]]) -> None:
        self.value = volume

    def sql_query(self) -> str:
        return self._sql_query_for_floats_and_ranges


class Custom(ExperimentParameter):
    def __init__(self, query: str) -> None:
        self.value = query

    def sql_query(self) -> str:
        return self.value


class ExperimentFilter(object):
    def __init__(self, filters: List[ExperimentParameter]) -> None:
        self.conn = sqlite3.connect("Library copy.db")
        self.filters = filters

    def __call__(self, items: Union[list, set]) -> list:
        query_base = f"""SELECT Experiment_ID FROM experiments
                        WHERE Experiment_ID IN ({', '.join(map(str, items))})"""
        filter_queries = [filter.query for filter in self.filters]

        sub_df = pd.read_sql_query(' AND '.join([query_base] + filter_queries),
                                   self.conn)

        return list(set(items) & set(sub_df.Experiment_ID))


class DataFilter(object):
    """Removes compounds for which experimental data is not available.

    parameters: Vapor Pressure, Boiling Point, Flash Point, Viscosity"""
    def __init__(self, parameter: str):
        if parameter not in EXPERIMENTAL_MEASUREMENTS:
            raise ValueError(
                f"Parameter must be in: {EXPERIMENTAL_MEASUREMENTS}")
        self.conn = sqlite3.connect("Analytes.db")
        self.parameter = parameter

    def __call__(self, items: Union[list, set]) -> list:
        query = f"""SELECT "Experiment Number" FROM ExperimentData
                    WHERE "Experiment Number" IN ({', '.join(map(str, items))})
                    AND "Analyte ID" IN
                        (SELECT "Analyte ID" FROM AnalyteData
                        WHERE "{self.parameter}" IS NOT null)
                    """

        exp_id_df = pd.read_sql_query(query, self.conn2)

        return list(set(items) & set(exp_id_df['Experiment Number']))

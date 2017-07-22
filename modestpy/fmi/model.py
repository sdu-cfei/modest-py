"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from modestpy.log_init import LogInit
LOG_INIT = LogInit(__name__)
LOGGER = LOG_INIT.get_logger()

from pyfmi import load_fmu
from pyfmi.fmi import FMUException
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Model:
    """
    FMU model to be simulated with inputs and parameters provided from files or dataframes.
    """

    # Number of tries to simulate a model
    # (sometimes the solver can't converge at first)
    TRIES = 3

    def __init__(self, fmu_path):
        self.model = load_fmu(fmu_path)
        self.start = None
        self.end = None
        self.timeline = None

        self.input_names = list()
        self.input_values = list()
        self.output_names = list()
        self.parameter_names = list()
        # self.full_io = list()

        self.parameter_df = pd.DataFrame()

        self.res = None

    def inputs_from_csv(self, csv, sep=',', exclude=list()):
        """
        Reads inputs from a CSV file (format of the standard input file in ModelManager).
        It is assumed that time is given in seconds.
        :param csv: Path to the CSV file
        :param exclude: list of strings, columns to be excluded
        :return: None
        """
        df = pd.read_csv(csv, sep=sep)
        assert 'time' in df.columns, "'time' not present in csv..."
        df = df.set_index('time')
        self.inputs_from_df(df, exclude)

    def inputs_from_df(self, df, exclude=list()):
        """
        Reads inputs from dataframe.

        Index must be named 'time' and given in seconds.
        The index name assertion check is implemented to avoid
        situations in which a user read DataFrame from csv
        and forgot to use ``DataFrame.set_index(column_name)``
        (it happens quite often...).

        :param df: DataFrame
        :param exclude: list of strings, names of columns to be omitted
        :return:
        """
        assert df.index.name == 'time', "Index name ('{}') different than 'time'! " \
                                        "Are you sure you assigned index " \
                                        "correctly?".format(df.index.name)
        self.timeline = df.index.values
        self.start = self.timeline[0]
        self.end = self.timeline[-1]

        for col in df:
            if col not in exclude:
                if col not in self.input_names:
                    self.input_names.append(col)
                    self.input_values.append(df[col].values)

    def specify_outputs(self, outputs):
        """
        Specifies names of output variables
        :param outputs: List of strings, names of output variables
        :return: None
        """
        for name in outputs:
            if name not in self.output_names:
                self.output_names.append(name)

    def parameters_from_csv(self, csv, sep=','):
        df = pd.read_csv(csv, sep=sep)
        self.parameters_from_df(df)

    def parameters_from_df(self, df):
        if df is not None:
            df = df.copy()
            for col in df:
                self.parameter_df[col] = df[col]

    def simulate(self, com_points=None, reset=True):
        """
        Performs a simulation
        :param com_points: int, number of communication points
        :param reset: bool, if True, the model will be resetted after simulation (use False with E+ FMU)
        :return: Dataframe with results
        """

        if com_points is None:
            LOGGER.warning('[fmi\\model] Warning! Default number of communication points assumed (500)')
            com_points = 500

        # IC
        self._set_ic()

        # Set parameters
        if not self.parameter_df.empty:
            self._set_all_parameters()

        # Inputs
        i = list()
        i.append(self.timeline)
        i.extend(self.input_values)
        i = Model._merge_inputs(i)
        input_obj = [self.input_names, i]

        # Options
        opts = self.model.simulate_options()
        opts['ncp'] = com_points
        opts['result_handling'] = 'memory'              # Prevents saving a result file
        opts['result_handler'] = 'ResultHandlerMemory'  # Prevents saving a result file
        opts["CVode_options"]["atol"] = 1e-6

        # Simulation
        tries = 0
        while tries < Model.TRIES:
            try:
                assert (self.start is not None) and (self.end is not None), 'start and stop cannot be None'
                self.res = self.model.simulate(start_time=self.start,
                                               final_time=self.end,
                                               input=input_obj,
                                               options=opts)
                break
            except FMUException as e:
                tries += 1
                if tries >= Model.TRIES:
                    raise e


        # Convert result to dataframe
        df = pd.DataFrame()
        df['time'] = self.res['time']
        df = df.set_index('time')
        for var in self.output_names:
            df[var] = self.res[var]

        # Reset model
        if reset:
            try:
                self.reset()
            except FMUException as e:
                LOGGER.warning(e.message)
                LOGGER.warning("If you try to simulate an EnergyPlus FMU, use reset=False")

        # Return
        return df

    def reset(self):
        """
        Resets model. After resetting inputs, parameters and outputs must be set again!
        :return: None
        """
        self.model.reset()

    def _set_ic(self):
        """Sets initial condition (ic)."""
        ic = dict()
        for i in range(len(self.input_names)):
            ic[self.input_names[i]] = self.input_values[i][0]
        # Call PyFMI method
        for var in ic:
            self.model.set(var, ic[var])

    def _set_parameter(self, name, value):
        if name not in self.parameter_names:
            self.parameter_names.append(name)
        self.model.set(name, value)

    def _set_all_parameters(self):
        for var in self.parameter_df:
            self._set_parameter(var, self.parameter_df[var])

    @staticmethod
    def _merge_inputs(inputs):
        return np.transpose(np.vstack(inputs))

    @staticmethod
    def _create_timeline(end, intervals):
        t = np.linspace(0, end, intervals+1)
        return t


if __name__ == "__main__":

    # Exemplary code
    model = Model('C:\\Users\\krza\\Documents\\GitLab\\modelica-models\\fmu\\occ\\TCO2_occ.fmu')

    model.specify_outputs(['T', 'CO2'])
    model.inputs_from_csv('C:\\Users\\krza\\Documents\\GitLab\\modelica-models\\resources\\input\\22_511_2\\input.csv',
                          exclude=['T', 'CO2'])
    model.parameters_from_csv('C:\\Users\\krza\\Documents\\GitLab\\modelica-models\\resources\\input\\22_511_2\\'
                              'parameters.csv')

    res = model.simulate(500)
    # full = model.get_full_result()
    res.plot(subplots=True)
    print res
    # print full
    plt.show()

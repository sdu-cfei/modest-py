"""DEPRECATED. PyFMI-based model. Superseded by FMPy-based class.

Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""
import logging

import numpy as np
import pandas as pd
from pyfmi import load_fmu
from pyfmi.fmi import FMUException


class Model(object):
    """
    FMU model to be simulated with inputs and parameters provided from
    files or dataframes.
    """

    # Number of tries to simulate a model
    # (sometimes the solver can't converge at first)
    TRIES = 15

    def __init__(self, fmu_path, opts=None):
        self.logger = logging.getLogger(type(self).__name__)

        try:
            self.logger.debug("Loading FMU")
            self.model = load_fmu(fmu_path)
        except FMUException as e:
            self.logger.error(type(e).__name__)
            self.logger.error(e.message)

        self.start = None
        self.end = None
        self.timeline = None
        self.opts = opts

        self.input_names = list()
        self.input_values = list()
        self.output_names = list()
        self.parameter_names = list()
        # self.full_io = list()

        self.parameter_df = pd.DataFrame()

        self.res = None

    def inputs_from_csv(self, csv, sep=",", exclude=list()):
        """
        Reads inputs from a CSV file (format of the standard input file
        in ModelManager). It is assumed that time is given in seconds.
        :param csv: Path to the CSV file
        :param exclude: list of strings, columns to be excluded
        :return: None
        """
        df = pd.read_csv(csv, sep=sep)
        assert "time" in df.columns, "'time' not present in csv..."
        df = df.set_index("time")
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
        assert df.index.name == "time", (
            "Index name ('{}') different "
            "than 'time'! "
            "Are you sure you assigned index "
            "correctly?".format(df.index.name)
        )
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

    def parameters_from_csv(self, csv, sep=","):
        df = pd.read_csv(csv, sep=sep)
        self.parameters_from_df(df)

    def parameters_from_df(self, df):
        self.logger.debug(f"parameters_from_df = {df}")
        if df is not None:
            df = df.copy()
            for col in df:
                self.parameter_df[col] = df[col]
        self.logger.debug(f"Updated parameters: {self.parameter_df}")

    def simulate(self, com_points=None, reset=True):
        """
        Performs a simulation.
        :param int com_points: number of communication points, if None,
                               standard value is used (500)
        :param bool reset: if True, the model will be resetted after
                           simulation (use False with E+ FMU)
        :param dict opts: Additional FMI options to be passed to the simulator
                          (consult FMI specification)
        :return: Dataframe with results
        """

        if com_points is None:
            self.logger.warning(
                "[fmi\\model] Warning! Default number "
                "of communication points assumed (500)"
            )
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

        # Options (fixed)
        fmi_opts = self.model.simulate_options()

        # Prevents saving results to a file
        fmi_opts["result_handling"] = "memory"
        fmi_opts["result_handler"] = "ResultHandlerMemory"

        # No output <- works only with CVode solver
        # fmi_opts['CVode_options']['verbosity'] = 50

        # Options (provided by the user)
        fmi_opts["ncp"] = com_points

        if (self.opts is not None) and (type(self.opts) is dict):
            self.logger.debug("User-defined FMI options found: {}".format(self.opts))
            for k in self.opts:
                if type(self.opts[k]) is not dict:
                    fmi_opts[k] = self.opts[k]
                    self.logger.debug(
                        "Setting FMI option: [{}] = {}".format(k, self.opts[k])
                    )
                elif type(self.opts[k]) is dict:
                    for subkey in self.opts[k]:
                        # It works only for single nested sub-dictionaries
                        fmi_opts[k][subkey] = self.opts[k][subkey]
                        self.logger.debug(
                            "Setting FMI option: [{}][{}] = {}".format(
                                k, subkey, self.opts[k][subkey]
                            )
                        )
                else:
                    raise TypeError("Wrong type of values in 'opts' dictionary")

        # Save all options to log
        self.logger.debug("All FMI options: {}".format(fmi_opts))

        # Simulation
        tries = 0
        while tries < Model.TRIES:
            try:
                assert (self.start is not None) and (
                    self.end is not None
                ), "start and stop cannot be None"  # Shouldn't it be OR?
                self.logger.debug("Starting simulation")
                self.res = self.model.simulate(
                    start_time=self.start,
                    final_time=self.end,
                    input=input_obj,
                    options=fmi_opts,
                )
                break
            except FMUException as e:
                tries += 1
                self.logger.warning("Simulation failed, failure no. {}".format(tries))
                self.logger.warning(type(e).__name__)
                self.logger.warning(str(e))
                if tries >= Model.TRIES:
                    self.logger.error(
                        "Maximum number of failures "
                        "reached ({}). "
                        "Won't try again...".format(Model.TRIES)
                    )
                    raise e

        # Convert result to dataframe
        df = pd.DataFrame()
        df["time"] = self.res["time"]
        df = df.set_index("time")
        for var in self.output_names:
            df[var] = self.res[var]

        # Reset model
        if reset:
            try:
                self.reset()
            except FMUException as e:
                self.logger.warning(e.message)
                self.logger.warning(
                    "If you try to simulate an EnergyPlus FMU, " "use reset=False"
                )

        # Return
        return df

    def reset(self):
        """
        Resets model. After resetting inputs, parameters and outputs
        must be set again!
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
        t = np.linspace(0, end, intervals + 1)
        return t

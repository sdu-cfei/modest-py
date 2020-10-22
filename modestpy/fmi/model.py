"""FMI wrapper for the model.

Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.L
See LICENSE file in the project root for license terms.
"""
import logging
from fmpy import simulate_fmu, read_model_description, instantiate_fmu, extract
import numpy as np
import pandas as pd


def df_to_struct_arr(df):
    """Converts a DataFrame to structured array."""
    # time index must be reset to pass it to the struct_arr
    df = df.reset_index()
    struct_arr = np.rec.fromrecords(df, names=df.columns.tolist())

    return struct_arr


def struct_arr_to_df(arr):
    """Converts a structured array to DataFrame."""
    df = pd.DataFrame(arr).set_index('time')

    return df


class Model(object):
    """FMU model to be simulated with inputs and parameters provided from
    files or dataframes.
    """
    def __init__(self, fmu_path, opts=None):
        self.logger = logging.getLogger(type(self).__name__)

        try:
            self.logger.debug("Loading FMU")
            # Load FMU
            model_desc = read_model_description(fmu_path)
            self.unzipdir = extract(fmu_path)
            self.model = instantiate_fmu(self.unzipdir, model_desc)

        except Exception as e:
            self.logger.error(type(e).__name__)
            self.logger.error(str(e))
            raise e

        self.start = None
        self.end = None
        self.timeline = None
        self.opts = opts  # TODO: Not used at the moment

        self.input_arr = None
        self.output_names = list()
        self.parameters = dict()

    def inputs_from_csv(self, csv, sep=',', exclude=list()):
        """Reads inputs from a CSV file.

        Time (column `time`) should be given in seconds.

        :param csv: Path to the CSV file
        :param exclude: list of strings, columns to be excluded
        :return: None
        """
        df = pd.read_csv(csv, sep=sep)
        assert 'time' in df.columns, "'time' not present in csv..."
        df = df.set_index('time')
        self.inputs_from_df(df, exclude)

    def set_input(self, df, exclude=list()):
        return self.inputs_from_df(df, exclude)

    def inputs_from_df(self, df, exclude=list()):
        """Reads inputs from dataframe.

        Index must be named 'time' and given in seconds.
        The index name assertion check is implemented to avoid
        situations in which a user read DataFrame from csv
        and forgot to use ``DataFrame.set_index(column_name)``
        (it happens quite often...).

        :param df: DataFrame
        :param exclude: list of strings, names of columns to be omitted
        :return:
        """
        assert df.index.name == 'time', "Index name ('{}') different " \
                                        "than 'time'! " \
                                        "Are you sure you assigned index " \
                                        "correctly?".format(df.index.name)
        if len(exclude) > 0:
            df = df.drop(exclude, axis='columns')
        self.timeline = df.index.values
        self.start = self.timeline[0]
        self.end = self.timeline[-1]
        self.input_arr = df_to_struct_arr(df)

    def set_outputs(self, outputs):
        self.specify_outputs(outputs)

    def specify_outputs(self, outputs):
        """Specifies names of output variables.

        :param outputs: List of strings, names of output variables
        :return: None
        """
        for name in outputs:
            if name not in self.output_names:
                self.output_names.append(name)

    def parameters_from_csv(self, csv, sep=','):
        """Read parameters from a CSV file."""
        df = pd.read_csv(csv, sep=sep)
        self.parameters_from_df(df)

    def set_param(self, df):
        self.parameters_from_df(df)

    def parameters_from_df(self, df):
        """Get parameters from a DataFrame."""
        self.logger.debug(f'parameters_from_df = {df}')
        if df is not None:
            df = df.copy()
            for col in df:
                self.parameters[col] = df[col]
        self.logger.debug(f'Updated parameters: {self.parameters}')

    def simulate(self, reset=True):
        """Performs a simulation.

        :param bool reset: if True, the model will be resetted after
                           simulation (use False with E+ FMU)
        :return: Dataframe with results
        """
        # Calculate output interval (in seconds)
        assert self.input_arr is not None, "No inputs assigned"
        output_interval = self.input_arr[1][0] - self.input_arr[0][0]

        # Initial condition
        start_values = dict()
        input_names = self.input_arr.dtype.names
        for name in input_names:
            if name != 'time':
                start_values[name] = self.input_arr[name][0]

        assert 'time' in input_names, "time must be the first input"

        # Set parameters
        for name, value in self.parameters.items():
            if name != 'time':
                start_values[name] = value

        # Inputs
        assert self.input_arr is not None, "Inputs not assigned!"

        # Options (fixed)
        pass  # TODO

        # Options (user)
        pass  # TODO

        # Simulation
        self.logger.debug("Starting simulation")
        result = simulate_fmu(
            self.unzipdir,
            start_values=start_values,
            start_time=self.start,
            stop_time=self.end,
            input=self.input_arr,
            output=self.output_names,
            output_interval=output_interval,
            fmu_instance=self.model
            #solver='Euler',  # TODO: It might be useful to add solver/step to options
            #step_size=0.005
        )

        # Convert result to DataFrame
        res_df = struct_arr_to_df(result)

        # Reset model
        if reset:
            try:
                self.model.reset()
            except Exception as e:
                self.logger.warning(str(e))
                self.logger.warning(
                    "If you try to simulate an EnergyPlus FMU, "
                    "use reset=False"
                )
        # Return
        return res_df

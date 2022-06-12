"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""
import copy
import logging
import os
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

import modestpy.estim.error
import modestpy.utilities.figures as figures
from modestpy.estim.ga.ga import GA
from modestpy.estim.ga_parallel.ga_parallel import MODESTGA
from modestpy.estim.plots import plot_comparison
from modestpy.estim.ps.ps import PS
from modestpy.estim.scipy.scipy import SCIPY
from modestpy.fmi.model import Model
from modestpy.loginit import config_logger


class Estimation(object):
    """Public interface of `modestpy`.

    Index in DataFrames `inp` and `ideal` must be named 'time'
    and given in seconds. The index name assertion check is
    implemented to avoid situations in which a user reads DataFrame
    from a csv and forgets to use `DataFrame.set_index(column_name)`
    (it happens quite often...).

    Currently available estimation methods:
        - MODESTGA  - parallel genetic algorithm (default GA in modestpy)
        - GA_LEGACY - single-process genetic algorithm (legacy implementation, discouraged)
        - PS        - pattern search (Hooke-Jeeves)
        - SCIPY     - interface to algorithms available through
                      scipy.optimize.minimize()

    Parameters:
    -----------
    workdir: str
        Output directory, must exist
    fmu_path: str
        Absolute path to the FMU
    inp: pandas.DataFrame
        Input data, index given in seconds and named 'time'
    known: dict(str: float)
        Dictionary with known parameters (`parameter_name: value`)
    est: dict(str: tuple(float, float, float))
        Dictionary defining estimated parameters,
        (`par_name: (guess value, lo limit, hi limit)`)
    ideal: pandas.DataFrame
        Ideal solution (usually measurements),
        index in seconds and named `time`
    lp_n: int or None
        Number of learning periods, one if `None`
    lp_len: float or None
        Length of a single learning period, entire `lp_frame` if `None`
    lp_frame: tuple of floats or None
        Learning period time frame, entire data set if `None`
    vp: tuple(float, float) or None
        Validation period, entire data set if `None`
    ic_param: dict(str, str) or None
        Mapping between model parameters used for IC and variables from
        `ideal`
    methods: tuple(str, str)
        List of methods to be used in the pipeline
    ga_opts: dict
        Genetic algorithm options
    ps_opts: dict
        Pattern search options
    scipy_opts: dict
        SciPy solver options
    ftype: string
        Cost function type. Currently 'NRMSE' (advised for multi-objective
        estimation) or 'RMSE'.
    default_log: bool
        If true, use default logging settings. Use false if you want to
        use own logging.
    logfile: str
        If default_log=True, this argument can be used to specify the log
        file name
    """

    # Number of attempts to find nonzero learning data set
    NONZERO_ATTEMPTS = 20

    # Ploting settings
    FIG_DPI = 150
    FIG_SIZE = (10, 6)

    def __init__(
        self,
        workdir,
        fmu_path,
        inp,
        known,
        est,
        ideal,
        lp_n=None,
        lp_len=None,
        lp_frame=None,
        vp=None,
        ic_param=None,
        methods=("MODESTGA", "PS"),
        ga_opts={},
        ps_opts={},
        scipy_opts={},
        modestga_opts={},
        ftype="RMSE",
        default_log=True,
        logfile="modestpy.log",
    ):

        # Default logging configuration?
        if default_log:
            config_logger(filename=logfile, level="WARNING")

        self.logger = logging.getLogger(type(self).__name__)

        # Sanity checks
        assert inp.index.equals(ideal.index), "inp and ideal indexes are not matching"

        init, lo, hi = 0, 1, 2  # Init. value, lower bound, upper bound indices
        for v in est:
            assert (est[v][init] >= est[v][lo]) and (
                est[v][init] <= est[v][hi]
            ), "Initial value out of limits ({})".format(v)

        # Input data
        self.workdir = workdir
        self.fmu_path = fmu_path
        self.inp = inp
        self.known = known
        self.est = est
        self.ideal = ideal
        self.methods = methods
        self.ftype = ftype

        # Results placeholders
        self.best_per_run = pd.DataFrame()
        self.final = pd.DataFrame()

        # Estimation options
        # GA options
        self.GA_OPTS = {
            "maxiter": 50,
            "pop_size": max((4 * len(est.keys()), 20)),
            "tol": 1e-6,
            "mut": 0.10,
            "mut_inc": 0.33,
            "uniformity": 0.5,
            "look_back": 50,
            "lhs": False,
            "ftype": ftype,
        }  # Default

        # Default
        self.GA_OPTS["trm_size"] = max(self.GA_OPTS["pop_size"] // 6, 2)
        # User options
        self.GA_OPTS = self._update_opts(self.GA_OPTS, ga_opts, "GA_LEGACY")

        # PS options
        self.PS_OPTS = {
            "maxiter": 500,
            "rel_step": 0.02,
            "tol": 1e-11,
            "try_lim": 1000,
            "ftype": ftype,
        }  # Default

        # User options
        self.PS_OPTS = self._update_opts(self.PS_OPTS, ps_opts, "PS")

        # SCIPY options
        self.SCIPY_OPTS = {
            "solver": "L-BFGS-B",
            "options": {"disp": True, "iprint": 2, "maxiter": 150, "full_output": True},
            "ftype": ftype,
        }  # Default

        # User options
        self.SCIPY_OPTS = self._update_opts(self.SCIPY_OPTS, scipy_opts, "SCIPY")

        # MODESTGA options
        self.MODESTGA_OPTS = {
            "workers": 3,  # CPU cores to use
            "generations": 50,  # Max. number of generations
            "pop_size": 30,  # Population size
            "mut_rate": 0.01,  # Mutation rate
            "trm_size": 3,  # Tournament size
            "tol": 1e-3,  # Solution tolerance
            "inertia": 100,  # Max. number of non-improving generations
            "ftype": ftype,
        }  # Default

        # User options
        self.MODESTGA_OPTS = self._update_opts(
            self.MODESTGA_OPTS, modestga_opts, "MODESTGA"
        )

        # Method dictionary
        self.method_dict = {
            "MODESTGA": (MODESTGA, self.MODESTGA_OPTS),
            "GA_LEGACY": (GA, self.GA_OPTS),
            "PS": (PS, self.PS_OPTS),
            "SCIPY": (SCIPY, self.SCIPY_OPTS),
        }  # Key -> method name, value -> (method class, method options)

        # List of learning periods (tuples with start, stop)
        self.lp = self._select_lp(lp_n, lp_len, lp_frame)

        # Validation period (a tuple with start, stop)
        if vp is not None:
            self.vp = vp
        else:
            self.vp = (ideal.index[0], ideal.index[-1])

        # Initial condition parameters
        # Take first value from time series from 'ideal' column
        self.ic_param = ic_param  # dict (par_name: ideal_column_name)

    # PUBLIC METHODS =====================================================

    def estimate(self, get="best"):
        """
        Estimates parameters.

        Returns average or best estimates depending on `get`.
        Average parameters are calculated as arithmetic average
        from all learning periods. Best parameters are those which
        resulted in the lowest error during respective learning period.
        It is advised to use 'best' parameters.

        The chosen estimates ('avg' or 'best') are saved
        in a csv file `final.csv` in the working directory.
        In addition estimates and errors from all learning periods
        are saved in `best_per_run.csv`.

        Parameters
        ----------
        get: str, default 'best'
            Type of returned estimates: 'avg' or 'best'

        Returns
        -------
        dict(str: float)
        """
        # (0) Sanity checks
        allowed_types = ["best", "avg"]
        assert get in allowed_types, "get={} is not allowed".format(get)

        # (1) Initialize local variables
        # Tuple with method names
        # e.g. ('MODESTGA', 'PS'), ('MODESTGA', 'SCIPY') or ('MODESTGA', )
        methods = self.methods

        # List of plots to be saved
        plots = list()

        cols = ["_method_", "_error_"] + [par_name for par_name in self.est]

        # Estimates and errors from all iterations from all methods
        summary = pd.DataFrame(columns=cols)
        summary.index.name = "_iter_"

        # List of DataFrames with summaries from all runs
        summary_list = list()

        # (2) Double step estimation
        n = 1  # Learning period counter

        for period in self.lp:
            # (2.1) Copy initial parameters
            est = copy.copy(self.est)

            # (2.2) Slice data
            start, stop = period[0], period[1]
            inp_slice = self.inp.loc[start:stop]
            ideal_slice = self.ideal.loc[start:stop]

            # (2.3) Get data for IC parameters and add to known parameters
            if self.ic_param:
                for par in self.ic_param:
                    ic = ideal_slice[self.ic_param[par]].iloc[0]
                    self.known[par] = ic

            # (2.4) Iterate over estimation methods (append results from all)
            m = 0  # Method counter
            for m_name in methods:
                # (2.4.1) Instantiate method class
                m_class = self.method_dict[m_name][0]
                m_opts = self.method_dict[m_name][1]

                m_inst = m_class(
                    self.fmu_path, inp_slice, self.known, est, ideal_slice, **m_opts
                )

                # (2.4.2) Estimate
                m_estimates = m_inst.estimate()

                # (2.4.3) Update current estimates
                # (stored in self.est dictionary)
                for key in est:
                    new_value = m_estimates[key].iloc[0]
                    est[key] = (new_value, est[key][1], est[key][2])

                # (2.4.4) Append summary
                full_traj = m_inst.get_full_solution_trajectory()
                if m > 0:
                    # Add iterations from previous methods
                    full_traj.index += summary.index[-1]
                summary = summary.append(full_traj, verify_integrity=True)
                summary.index.rename("_iter_", inplace=True)

                # (2.4.5) Save method's plots
                plots = m_inst.get_plots()
                for p in plots:
                    fig = figures.get_figure(p["axes"])
                    fig_file = os.path.join(
                        self.workdir, "{}_{}.png".format(p["name"], n)
                    )
                    fig.set_size_inches(Estimation.FIG_SIZE)
                    fig.savefig(fig_file, dpi=Estimation.FIG_DPI)
                plt.close("all")

                # (2.4.6) Increase method counter
                m += 1

            # (2.5) Add summary from this run to the list of all summaries
            summary_list.append(summary)
            summary = pd.DataFrame(columns=cols)  # Reset

            # (2.6) Increase learning period counter
            n += 1

        # (3) Get and save best estimates per run and final estimates
        best_per_run = self._get_finals(summary_list)
        best_per_run.to_csv(os.path.join(self.workdir, "best_per_run.csv"))

        if get == "best":
            cond = best_per_run["_error_"] == best_per_run["_error_"].min()
            # Take only one if more than one present
            final = best_per_run.loc[cond].iloc[0:1]
            final = final.drop("_error_", axis=1)
        elif get == "avg":
            final = best_per_run.drop("_error_", axis=1).mean().to_frame().T
        else:
            # This shouldn't happen, because the type is checked at (0)
            raise RuntimeError("Unknown type of estimates: {}".format(get))

        final_file = os.path.join(self.workdir, "final.csv")
        final.to_csv(final_file, index=False)

        # (4) Save summaries from all learning periods
        for s, n in zip(summary_list, range(1, len(summary_list) + 1)):
            sfile = os.path.join(self.workdir, "summary_{}.csv".format(n))
            s.to_csv(sfile)

        # (5) Save error plot including all learning periods
        ax = self._plot_error_per_run(summary_list, err_type=self.ftype)
        fig = figures.get_figure(ax)
        fig.savefig(os.path.join(self.workdir, "errors.png"))

        # (6) Assign results to instance attributes
        self.best_per_run = best_per_run
        self.final = final

        # (7) Estimates to dict
        final = final.to_dict("records")[0]

        # (8) Delete temp dirs
        self._clean()

        # (9) Return final estimates
        return final

    def validate(self, vp=None):
        """
        Performs a simulation with estimated parameters
        for the previously selected validation period. Other period
        can be chosen with the `vp` argument. User chosen `vp` in this method
        does not override the validation period chosen during instantiation
        of this class.

        Parameters
        ----------
        vp: tuple or None
            Validation period given as a tuple of start and stop time in
            seconds.

        Returns
        -------
        dict
            Validation error, keys: 'tot', '<var1>', '<var2>', ...
        pandas.DataFrame
            Simulation result
        """
        # Get estimates
        est = self.final
        est.index = [0]  # Reset index (needed by model.set_param())

        self.logger.info(
            "Validation of parameters: {}".format(str(est.iloc[0].to_dict()))
        )

        # Slice data
        if vp is None:
            start, stop = self.vp[0], self.vp[1]
        else:
            start, stop = vp[0], vp[1]
        inp_slice = self.inp.loc[start:stop]
        ideal_slice = self.ideal.loc[start:stop]

        # Initialize IC parameters and add to known
        if self.ic_param:
            for par in self.ic_param:
                ic = ideal_slice[self.ic_param[par]].iloc[0]
                self.known[par] = ic

        # Initialize model
        model = Model(self.fmu_path)
        model.set_input(inp_slice)
        model.set_param(est)
        model.set_param(self.known)
        model.set_outputs(list(self.ideal.columns))

        # Simulate and get error
        try:
            result = model.simulate()
        except Exception as e:
            msg = "Problem found inside FMU. Did you set all parameters?"
            self.logger.error(str(e))
            self.logger.error(msg)
            raise e

        err = modestpy.estim.error.calc_err(result, ideal_slice)

        # Create validation plot
        ax = plot_comparison(result, ideal_slice, f=None)
        fig = figures.get_figure(ax)
        fig.set_size_inches(Estimation.FIG_SIZE)
        fig.savefig(
            os.path.join(self.workdir, "validation.png"), dpi=Estimation.FIG_DPI
        )

        # Remove temp dirs
        self._clean()

        # Return
        return err, result

    # PRIVATE METHODS ====================================================

    def _get_finals(self, summary_list):
        """
        Returns final estimates and errors from all learning periods

        :param list(DataFrame) summary_list: List of all summaries
                                             from all runs
        :param bool avg: If true, return average estimates, else return
                         best estimates
        :return: DataFrame with final estimates
        """
        finals = pd.DataFrame()
        for s in summary_list:
            finals = finals.append(
                s.drop("_method_", axis=1).iloc[-1:], ignore_index=True
            )
        finals.index += 1  # Start from 1
        finals.index.name = "_run_"

        return finals

    def _update_opts(self, opts, new_opts, method):
        """
        Updates the dictionary with method options.

        :param dict opts: Options to be updated
        :param dict new_opts: New options (can contain a subset of opts keys)
        :param str method: Method name, 'MODESTGA', 'PS' etc. (used only for logging)
        :return: Updated dict
        """
        if len(new_opts) > 0:
            for key in new_opts:
                if key not in opts.keys():
                    msg = "Unknown key: {}".format(key)
                    self.logger.error(msg)
                    raise KeyError(msg)
                self.logger.info(
                    "User defined option ({}): {} = {}".format(
                        method, key, new_opts[key]
                    )
                )
                opts[key] = new_opts[key]
        return opts

    def _plot_error_per_run(self, summary_list, err_type):
        """
        :param list(DataFrame) summary_list: Summary list
        :param str err_type: Error type
        :return: Axes
        """
        # Error evolution per estimation run
        err = pd.DataFrame()
        for s, n in zip(summary_list, range(1, len(summary_list) + 1)):
            next_err = pd.Series(data=s["_error_"], name="error #{}".format(n))
            err = pd.concat([err, next_err], axis=1)

        # Plot
        fig, ax = plt.subplots(
            1, 1, figsize=Estimation.FIG_SIZE, dpi=Estimation.FIG_DPI
        )
        err.plot(ax=ax)

        # Get line colors
        lines = ax.get_lines()
        colors = [l.get_color() for l in lines]

        # Method switch marks
        xloc, yloc = self._get_method_switch_xy(summary_list)

        # In cases there is more switches than lines
        mltp = len(xloc) // len(colors)

        if mltp > 1:
            colors_copy = copy.copy(colors)
            colors = list()
            for c in colors_copy:
                for i in range(mltp):
                    colors.append(c)

        for x, y, c in zip(xloc, yloc, colors * mltp):
            ax.scatter(x, y, marker="o", c="white", edgecolors=c, lw=1.5, zorder=10)

        # Formatting
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Error ({})".format(err_type))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

        return ax

    def _get_method_switch_xy(self, summary_list):
        """
        Returns a tuple with two lists describing x, y coordinates
        marking when there was a switch to a next method in the estimation.

        The first list contains x coordinates, the second list contains
        y coordinates. x coordinates represent iterations. y coordinates
        represent simulation error.

        :param list(DataFrame) summary_list: List of DataFrames with summary
        :return: tuple(list(int), list(int))
        """

        # Construct an array with indices marking method switches.
        # E.g. if there were 3 methods used in 2 estimation runs,
        # the returned list looks as follows:
        #
        # [[11, 34, 20],  -> estimation run #1
        #  [23, 49, 15]]  -> estimation run #2
        #    |   |   |
        #    m1  m2  m3
        #
        # where m1, m2, m3 are iteration numbers when method 1, 2, 3 started.
        switch_array = list()
        for s in summary_list:
            methods = s["_method_"].values
            switch = list()
            last = None
            for n, i in zip(methods, range(len(methods))):
                if last is None:
                    last = n
                if n != last:
                    switch.append(i)
                last = n
            switch_array.append(switch)

        # Generate x, y lists
        xloc = list()
        yloc = list()
        i = 0
        for run in switch_array:
            for index in run:
                xloc.append(index)
                yloc.append(summary_list[i]["_error_"].iloc[index - 1])
            i += 1

        return xloc, yloc

    def _select_lp(self, lp_n=None, lp_len=None, lp_frame=None):
        """
        Selects random learning periods within `lp_frame`.

        Each learning period has the length of `lp_len`. Periods may overlap.
        Ensures that a period with null data for any `ideal` variable is not
        selected.

        Parameters
        ----------
        lp_n: int, optional
            Number of periods, default: 1
        lp_len: int, optional
            Period length in seconds, default: all data
        lp_frame: tuple of floats or ints, optional
            Learning periods are selected within this time frame (start, end),
            default: all data

        Returns
        -------
        list of tuples of floats
        """
        # Defaults
        if lp_n is None:
            lp_n = 1
        if lp_frame is None:
            lp_frame = (self.ideal.index[0], self.ideal.index[-1])
        if lp_len is None:
            lp_len = lp_frame[1] - lp_frame[0]

        # Assign time frame
        t0 = lp_frame[0]
        tend = lp_frame[1]
        assert lp_len <= tend - t0, (
            "Learning period length cannot be " "longer than data length!"
        )

        # Return variable
        lp = []

        for i in range(lp_n):
            chosen_lp = False
            tries_left = Estimation.NONZERO_ATTEMPTS

            while not chosen_lp and tries_left > 0:
                new_t0 = random.randint(t0, tend - lp_len)
                new_tend = new_t0 + lp_len

                ideal_nonzero = self._all_columns_nonzero(
                    self.ideal.loc[new_t0:new_tend]
                )

                if ideal_nonzero:
                    lp.append((new_t0, new_tend))
                    chosen_lp = True
                else:
                    tries_left -= 1
                    self.logger.warning(
                        "Zero ideal solution not allowed, " "selecting another one..."
                    )
                    self.logger.warning("Number of tries left: {}".format(tries_left))

            if tries_left == 0:
                self.logger.error(
                    "Nonzero ideal solution not found "
                    "({} attempts)".format(Estimation.NONZERO_ATTEMPTS)
                )
                raise Exception

        return lp

    def _all_columns_nonzero(self, df):
        """
        Checks whether all columns in DataFrame are nonzero.

        Parameters
        ----------
        df: DataFrame

        Returns
        -------
        boolean
        """
        assert df.empty is False, (
            "This DataFrame should not be empty. " "Something went wrong."
        )
        all_zero = (df == 0).all().all()
        if all_zero:
            return False
        return True

    def _clean(self):
        """Clean temp dirs."""
        if Path(Model.tmpdir_file).exists():
            with open(Model.tmpdir_file, "r") as f:
                dirs = f.read().splitlines()
                self.logger.info(f"Temp dirs to remove: {dirs}")
                for d in dirs:
                    self.logger.info(f"-> removing temp dir: {d}")
                    shutil.rmtree(d, ignore_errors=True)
            os.remove(Model.tmpdir_file)

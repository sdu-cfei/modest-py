"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""
import logging
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyDOE as doe

import modestpy.estim.plots as plots
from modestpy.estim.estpar import EstPar
from modestpy.estim.ga import algorithm
from modestpy.estim.ga.population import Population


class GA(object):
    """DEPRECATED. Use MODESTGA instead.

    Genetic algorithm for FMU parameter estimation.
    This is the main class of the package, containing the high-level
    algorithm and some result plotting methods.
    """

    # Ploting settings
    FIG_DPI = 150
    FIG_SIZE = (15, 10)

    NAME = "GA"
    METHOD = "_method_"
    ITER = "_iter_"
    ERR = "_error_"

    def __init__(
        self,
        fmu_path,
        inp,
        known,
        est,
        ideal,
        maxiter=100,
        tol=0.001,
        look_back=10,
        pop_size=40,
        uniformity=0.5,
        mut=0.05,
        mut_inc=0.3,
        trm_size=6,
        ftype="RMSE",
        init_pop=None,
        lhs=False,
    ):
        """
        The population can be initialized in various ways:
        - if `init_pop` is None, one individual is initialized using
          initial guess from `est`
        - if `init_pop` contains less individuals than `pop_size`,
          then the rest is random
        - if `init_pop` == `pop_size` then no random individuals are generated

        :param fmu_path: string, absolute path to the FMU
        :param inp: DataFrame, columns with input timeseries, index in seconds
        :param known: Dictionary, key=parameter_name, value=value
        :param est: Dictionary, key=parameter_name, value=tuple
                    (guess value, lo limit, hi limit), guess can be None
        :param ideal: DataFrame, ideal solution to be compared with model
                      outputs (variable names must match)
        :param maxiter: int, maximum number of generations
        :param tol: float, when error does not decrease by more than
                    ``tol`` for the last ``lookback`` generations,
                    simulation stops
        :param look_back: int, number of past generations to track
                          the error decrease (see ``tol``)
        :param pop_size: int, size of the population
        :param uniformity: float (0.-1.), uniformity rate, affects gene
                           exchange in the crossover operation
        :param mut: float (0.-1.), mutation rate, specifies how often genes
                    are to be mutated to a random value,
                    helps to reach the global optimum
        :param mut_inc: float (0.-1.), increased mutation rate, specifies
                        how often genes are to be mutated by a
                        small amount, used when the population diversity
                        is low, helps to reach a local optimum
        :param trm_size: int, size of the tournament
        :param string ftype: Cost function type. Currently 'NRMSE'
                             (advised for multi-objective estimation)
                             or 'RMSE'.
        :param DataFrame init_pop: Initial population. DataFrame with
                                   estimated parameters. If None, takes
                                   initial guess from est.
        :param bool lhs: If True, init_pop and initial guess in est are
                         neglected, and the population is chosen using
                         Lating Hypercube Sampling.
        """
        self.logger = logging.getLogger(type(self).__name__)

        deprecated_msg = "This GA implementation is deprecated. Use MODESTGA instead."
        print(deprecated_msg)
        self.logger.warning(
            "This GA implementation is deprecated. Use MODESTGA instead."
        )

        self.logger.info("GA constructor invoked")

        assert inp.index.equals(ideal.index), "inp and ideal indexes are not matching"

        # Evolution parameters
        algorithm.UNIFORM_RATE = uniformity
        algorithm.MUT_RATE = mut
        algorithm.MUT_RATE_INC = mut_inc
        algorithm.TOURNAMENT_SIZE = int(trm_size)

        self.max_generations = maxiter
        self.tol = tol
        self.look_back = look_back

        # History of fittest errors from each generation (list of floats)
        self.fittest_errors = list()

        # History of all estimates and errors from all individuals
        self.all_estim_and_err = pd.DataFrame()

        # Initiliaze EstPar objects
        estpars = list()
        for key in sorted(est.keys()):
            self.logger.info(
                "Add {} (initial guess={}) to estimated parameters".format(
                    key, est[key][0]
                )
            )
            estpars.append(
                EstPar(name=key, value=est[key][0], lo=est[key][1], hi=est[key][2])
            )

        # Put known into DataFrame
        known_df = pd.DataFrame()
        for key in known:
            assert (
                known[key] is not None
            ), "None is not allowed in known parameters (parameter {})".format(key)
            known_df[key] = [known[key]]
            self.logger.info("Known parameters:\n{}".format(str(known_df)))

        # If LHS initialization, init_pop is disregarded
        if lhs:
            self.logger.info("LHS initialization")
            init_pop = GA._lhs_init(
                par_names=[p.name for p in estpars],
                bounds=[(p.lo, p.hi) for p in estpars],
                samples=pop_size,
                criterion="c",
            )
            self.logger.debug("Current population:\n{}".format(str(init_pop)))
        # Else, if no init_pop provided, generate one individual
        # based on initial guess from `est`
        elif init_pop is None:
            self.logger.info(
                "No initial population provided, one individual will be based "
                "on the initial guess and the other will be random"
            )
            init_pop = pd.DataFrame({k: [est[k][0]] for k in est})
            self.logger.debug("Current population:\n{}".format(str(init_pop)))

        # Take individuals from init_pop and add random individuals
        # until pop_size == len(init_pop)
        # (the number of individuals in init_pop can be lower than
        # the desired pop_size)
        if init_pop is not None:
            missing = pop_size - init_pop.index.size
            self.logger.debug("Missing individuals = {}".format(missing))
            if missing > 0:
                self.logger.debug("Add missing individuals (random)...")
                while missing > 0:
                    init_pop = init_pop.append(
                        {
                            key: random.random() * (est[key][2] - est[key][1])
                            + est[key][1]
                            for key in sorted(est.keys())
                        },
                        ignore_index=True,
                    )
                    missing -= 1
            self.logger.debug("Current population:\n{}".format(str(init_pop)))

        # Initialize population
        self.logger.debug("Instantiate Population ")
        self.pop = Population(
            fmu_path=fmu_path,
            pop_size=pop_size,
            inp=inp,
            known=known_df,
            est=estpars,
            ideal=ideal,
            init=True,
            ftype=ftype,
            init_pop=init_pop,
        )

    def estimate(self):
        """
        Proxy method. Each algorithm from ``estim``
        package should have this method

        :return: DataFrame
        """
        self.evolution()
        return self.get_estimates()

    def evolution(self):

        gen_count = 1
        err_decreasing = True

        # Generation 1 (initialized population)
        self.logger.info("Generation " + str(gen_count))
        self.logger.info(str(self.pop))

        # Update results
        self._update_res(gen_count)

        gen_count += 1

        # Next generations (evolution)
        while (gen_count <= self.max_generations) and err_decreasing:

            # Evolve
            self.pop = algorithm.evolve(self.pop)

            # Update results
            self._update_res(gen_count)

            # Print info
            self.logger.info("Generation " + str(gen_count))
            self.logger.info(str(self.pop))

            # Look back
            if len(self.fittest_errors) > self.look_back:
                err_past = self.fittest_errors[-self.look_back]
                err_now = self.fittest_errors[-1]
                err_decrease = err_past - err_now
                if err_decrease < self.tol:
                    self.logger.info(
                        "Error decrease smaller than tol: {0:.5f} < {1:.5f}".format(
                            err_decrease, self.tol
                        )
                    )
                    self.logger.info("Stopping evolution...")
                    err_decreasing = False
                else:
                    self.logger.info(
                        "'Look back' error decrease = {0:.5f} > "
                        "tol = {1:.5f}\n".format(err_decrease, self.tol)
                    )
            # Increase generation count
            gen_count += 1

        # Print summary
        self.logger.info("FITTEST PARAMETERS:\n{}".format(self.get_estimates()))

        # Return
        return self.pop.get_fittest()

    def get_estimates(self, as_dict=False):
        """
        Gets estimated parameters of the best (fittest) individual.

        :param as_dict: boolean (True to get dictionary instead DataFrame)
        :return: DataFrame
        """
        return self.pop.get_fittest_estimates()

    def get_error(self):
        """
        :return: float, last error
        """
        return self.pop.get_fittest_error()

    def get_errors(self):
        """
        :return: list, all errors from all generations
        """
        return self.fittest_errors

    def get_sim_res(self):
        """
        Gets simulation result of the best individual.

        :return: DataFrame
        """
        return self.pop.get_fittest().result.copy()

    def get_full_solution_trajectory(self):
        """
        Returns all parameters and errors from all iterations.
        The returned DataFrame contains columns with parameter names,
        additional column '_error_' for the error and the index
        named '_iter_'.

        :return: DataFrame
        """
        df = self.all_estim_and_err.copy()
        summary = pd.DataFrame()
        for i in range(1, df[GA.ITER].max() + 1):
            summary = summary.append(self._get_best_from_gen(i))

        summary[GA.ITER] = summary[GA.ITER].astype(int)
        summary = summary.set_index(GA.ITER)

        summary[GA.METHOD] = GA.NAME

        return summary

    def get_plots(self):
        """
        Returns a list with important plots produced by this estimation method.
        Each list element is a dictionary with keys 'name' and 'axes'. The name
        should be given as a string, while axes as matplotlib.Axes instance.

        :return: list(dict)
        """
        plots = list()
        plots.append({"name": "GA", "axes": self.plot_pop_evo()})
        return plots

    def save_plots(self, workdir):
        self.plot_comparison(os.path.join(workdir, "ga_comparison.png"))
        self.plot_error_evo(os.path.join(workdir, "ga_error_evo.png"))
        self.plot_parameter_evo(os.path.join(workdir, "ga_param_evo.png"))
        self.plot_pop_evo(os.path.join(workdir, "ga_pop_evo.png"))

    def plot_error_evo(self, file=None):
        """Returns a plot of the error evolution.

        :param file: string (path to the file, if None, file not created)
        :return: Axes
        """
        fig, ax = plt.subplots()
        ax.plot(self.fittest_errors)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Error (NRMSE)")
        if file:
            fig = ax.get_figure()
            fig.set_size_inches(GA.FIG_SIZE)
            fig.savefig(file, dpi=GA.FIG_DPI)
        return ax

    def plot_comparison(self, file=None):
        """
        Creates a plot with a comparison of simulation results
        (fittest individual) vs. measured result.

        :param file: string, path to the file. If ``None``, file not created.
        :return: Axes
        """
        simulated = self.get_sim_res()
        measured = self.pop.ideal.copy()
        return plots.plot_comparison(simulated, measured, file)

    def plot_parameter_evo(self, file=None):
        """
        Returns a plot of the parameter evolution.

        :param file: string (path to the file, if None, file not created)
        :return: Axes
        """
        parameters = self.get_full_solution_trajectory()
        parameters = parameters.drop("generation", axis=1)
        return plots.plot_parameter_evo(parameters, file)

    def plot_inputs(self, file=None):
        """
        Returns a plot with inputs.

        :param file: string
        :return: axes
        """
        inputs = self.pop.inputs
        return plots.plot_inputs(inputs, file)

    def plot_pop_evo(self, file=None):
        """
        Creates a plot with the evolution of all parameters as a scatter plot.
        Can be interpreted as the *population diversity*.
        The color of the points is darker for higher accuracy.

        :param file: string, path to the file. If ``None``, file not created.
        :return: Axes
        """
        estimates = self.all_estim_and_err
        pars = list(estimates.columns)
        pars.remove("individual")
        pars.remove(GA.ITER)
        pars.remove(GA.ERR)
        assert len(pars) > 0, "No parameters found"

        fig, axes = plt.subplots(nrows=len(pars), sharex=True, squeeze=False)
        fig.subplots_adjust(right=0.75)
        i = 0

        last_err = self.fittest_errors[-1]
        first_err = self.fittest_errors[0]

        for v in pars:
            ax = axes[i, 0]
            scatter = ax.scatter(
                x=estimates[GA.ITER],
                y=estimates[v],
                c=estimates[GA.ERR],
                cmap="viridis",
                edgecolors="none",
                vmin=last_err,
                vmax=first_err,
                alpha=0.25,
            )
            ax.set_xlim([0, estimates[GA.ITER].max() + 1])
            ax.text(
                x=1.05,
                y=0.5,
                s=v,
                transform=ax.transAxes,
                fontweight="bold",
                horizontalalignment="center",
                verticalalignment="center",
            )
            i += 1
        axes[-1, 0].set_xlabel("Generation")

        # Color bar on the side
        cbar_ax = fig.add_axes([0.85, 0.10, 0.05, 0.8])
        fig.colorbar(scatter, cax=cbar_ax, label="Error")

        if file:
            fig.set_size_inches(GA.FIG_SIZE)
            fig.savefig(file, dpi=GA.FIG_DPI)
        return axes

    def _update_res(self, gen_count):
        # Save estimates
        generation_estimates = self.pop.get_all_estimates_and_errors()
        generation_estimates[GA.ITER] = gen_count
        self.all_estim_and_err = pd.concat(
            [self.all_estim_and_err, generation_estimates]
        )

        # Append error lists
        self.fittest_errors.append(self.pop.get_fittest_error())

    def _get_best_from_gen(self, generation):
        """
        Gets fittest individuals (parameter sets) from the chosen generation.

        :param generation: int (generation number)
        :return: DataFrame
        """
        df = self.all_estim_and_err.copy()
        df.index = df[GA.ITER]
        # Select individuals with minimum error from the chosen individuals
        fittest = df.loc[df[GA.ERR] == df.loc[generation][GA.ERR].min()].loc[generation]
        # Check how many individuals found
        if isinstance(fittest, pd.DataFrame):
            # More than 1 found...
            # Get the first one
            fittest = fittest.iloc[0]
        elif isinstance(fittest, pd.Series):
            # Only 1 found...
            pass
        # Drop column 'individual'
        fittest = fittest.drop("individual")

        return fittest

    def _get_n_param(self):
        """
        Returns number of estimated parameters

        :return: int
        """
        return len(self.get_estimates())

    @staticmethod
    def _lhs_init(par_names, bounds, samples, criterion="c"):
        """
        Returns LHS samples.

        :param par_names: List of parameter names
        :type par_names: list(str)
        :param bounds: List of lower/upper bounds,
                       must be of the same length as par_names
        :type bounds: list(tuple(float, float))
        :param int samples: Number of samples
        :param str criterion: A string that tells lhs how to sample the
                              points. See docs for pyDOE.lhs().
        :return: DataFrame
        """
        lhs = doe.lhs(len(par_names), samples=samples, criterion="c")
        par_vals = {}
        for par, i in zip(par_names, range(len(par_names))):
            par_min = bounds[i][0]
            par_max = bounds[i][1]
            par_vals[par] = lhs[:, i] * (par_max - par_min) + par_min

        # Convert dict(str: np.ndarray) to pd.DataFrame
        par_df = pd.DataFrame(columns=par_names, index=np.arange(samples))
        for i in range(samples):
            for p in par_names:
                par_df.loc[i, p] = par_vals[p][i]

        logger = logging.getLogger(GA.__name__)
        logger.info("Initial guess based on LHS:\n{}".format(par_df))
        return par_df

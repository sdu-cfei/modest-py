"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from modestpy.log_init import LogInit
LOG_INIT = LogInit(__name__)
LOGGER = LOG_INIT.get_logger()

import algorithm
import modestpy.estim.plots as plots
from modestpy.estim.estpar import EstPar
from population import Population
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Plot files
DPI = 100
FIG_SIZE = (15, 10)


class GA:
    """
    Genetic algorithm for FMU parameter estimation.
    This is the main class of the package, containing the high-level algorithm and some result plotting methods.
    """
    def __init__(self, fmu_path, inp, known, est, ideal,
                 generations=100, tolerance=0.001, look_back=10,
                 pop_size=40, uniformity=0.5, mut=0.1, mut_inc=0.3, trm_size=6):
        """
        :param fmu_path: string, absolute path to the FMU
        :param inp: DataFrame, columns with input timeseries, index in seconds
        :param known: Dictionary, key=parameter_name, value=value
        :param est: Dictionary, key=parameter_name, value=tuple (guess value, lo limit, hi limit), guess can be None
        :param ideal: DataFrame, ideal solution to be compared with model outputs (variable names must match)
        :param generations: int, maximum number of generations
        :param tolerance: float, when error does not decrease by more than ``tolerance`` for the last ``lookback``
               generations, simulation stops
        :param look_back: int, number of past generations to track the error decrease (see ``tolerance``)
        :param pop_size: int, size of the population
        :param uniformity: float (0.-1.), uniformity rate, affects gene exchange in the crossover operation
        :param mut: float (0.-1.), mutation rate, specifies how often genes are to be mutated to a random value,
               helps to reach the global optimum
        :param mut_inc: float (0.-1.), increased mutation rate, specifies how often genes are to be mutated by a
               small amount, used when the population diversity is low, helps to reach a local optimum
        :param trm_size: int, size of the tournament
        """
        self.logger = LOGGER

        assert inp.index.equals(ideal.index), 'inp and ideal indexes are not matching'

        # Evolution parameters
        algorithm.UNIFORM_RATE = uniformity
        algorithm.MUT_RATE = mut
        algorithm.MUT_RATE_INC = mut_inc
        algorithm.TOURNAMENT_SIZE = trm_size

        self.max_generations = generations
        self.tolerance = tolerance
        self.look_back = look_back

        # Results
        self.fittest_errors = list()  # history of fittest errors from each generation (list of floats)
        self.all_estim_and_err = pd.DataFrame()  # history of all estimates and errors from all individuals

        # Initiliaze EstPar objects
        estpars = list()
        for key in est:
            estpars.append(EstPar(name=key, value=est[key][0], lo=est[key][1], hi=est[key][2]))

        # Put known into DataFrame
        known_df = pd.DataFrame()
        for key in known:
            assert known[key] is not None, 'None is not allowed in known parameters (parameter {})'.format(key)
            known_df[key] = [known[key]]

        # Initialize population
        self.logger.info('GENETIC ALGORITHM INSTANCE CREATED...')
        self.logger.info('Initializing the population...')
        self.pop = Population(fmu_path=fmu_path,
                              pop_size=pop_size,
                              inp=inp,
                              known=known_df,
                              est=estpars,
                              ideal=ideal,
                              init=True)

    def estimate(self):
        """
        Proxy method. Each algorithm from ``estim`` package should have this method

        :return: DataFrame
        """
        self.evolution()
        return self.get_estimates()

    def evolution(self):

        gen_count = 1
        err_decreasing = True

        # Generation 1 (initialized population)
        self.logger.info('Generation ' + str(gen_count))
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
            self.logger.info('Generation ' + str(gen_count))
            self.logger.info(str(self.pop))

            # Look back
            if len(self.fittest_errors) > self.look_back:
                err_past = self.fittest_errors[-self.look_back]
                err_now = self.fittest_errors[-1]
                err_decrease = err_past - err_now
                if err_decrease < self.tolerance:
                    self.logger.info('Error decrease smaller than tolerance: {0:.5f} < {1:.5f}'
                              .format(err_decrease, self.tolerance))
                    self.logger.info('Stopping evolution...')
                    err_decreasing = False
                else:
                    self.logger.info("'Look back' error decrease = {0:.5f} > tolerance = {1:.5f}\n"
                              .format(err_decrease, self.tolerance))
            # Increase generation count
            gen_count += 1

        # Print summary
        self.logger.info('FITTEST PARAMETERS:\n{}'.format(self.get_estimates()))

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

    def get_parameter_evolution(self):
        """
        Gets a DataFrame the best parameters from each generation.

        :return: DataFrame
        """
        df = self.all_estim_and_err.copy()
        par_evo = pd.DataFrame()
        for i in range(1, df['generation'].max() + 1):
            par_evo = par_evo.append(self._get_best_from_gen(i))
        return par_evo

    def save_plots(self, workdir):
        self.plot_comparison(os.path.join(workdir, 'ga_comparison.png'))
        self.plot_error_evo(os.path.join(workdir, 'ga_error_evo.png'))
        self.plot_parameter_evo(os.path.join(workdir, 'ga_param_evo.png'))
        self.plot_pop_evo(os.path.join(workdir, 'ga_pop_evo.png'))

    def plot_error_evo(self, file=None):
        """ Returns a plot of the error evolution.

        :param file: string (path to the file, if None, file not created)
        :return: Axes
        """
        fig, ax = plt.subplots()
        ax.plot(self.fittest_errors)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Error (NRMSE)')
        if file:
            fig = ax.get_figure()
            fig.set_size_inches(FIG_SIZE)
            fig.savefig(file, dpi=DPI, figsize=FIG_SIZE)
        return ax

    def plot_comparison(self, file=None):
        """
        Creates a plot with a comparison of simulation results (fittest individual) vs. measured result.

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
        parameters = self.get_parameter_evolution()
        parameters = parameters.drop('error', axis=1)
        parameters = parameters.drop('generation', axis=1)
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
        Creates a plot with the evolution of all parameters as a scatter plot. Can be interpreted as the
        *population diversity*. The color of the points is darker for higher accuracy.

        :param file: string, path to the file. If ``None``, file not created.
        :return: Axes
        """
        estimates = self.all_estim_and_err
        pars = list(estimates.columns)
        pars.remove('individual')
        pars.remove('generation')
        pars.remove('error')
        assert len(pars) > 0, 'No parameters found'

        fig, axes = plt.subplots(nrows=len(pars), sharex=True)
        fig.subplots_adjust(right=0.75)
        i = 0

        last_err = self.fittest_errors[-1]
        first_err = self.fittest_errors[0]

        for v in pars:
            ax = axes[i]
            scatter = ax.scatter(x=estimates['generation'], y=estimates[v], c=estimates['error'],
                                 cmap='viridis', edgecolors='none', vmin=last_err, vmax=first_err, alpha=0.25)
            ax.set_xlim([0, estimates['generation'].max() + 1])
            ax.text(x=1.05, y=0.5, s=v, transform=ax.transAxes, fontweight='bold',
                    horizontalalignment='center', verticalalignment='center')
            i += 1
        axes[-1].set_xlabel('Generation')

        # Color bar on the side
        cbar_ax = fig.add_axes([0.85, 0.10, 0.05, 0.8])
        fig.colorbar(scatter, cax=cbar_ax, label='Error')

        if file:
            fig.set_size_inches(FIG_SIZE)
            fig.savefig(file, dpi=DPI)
        return axes

    # def info(self, txt):
    #     class_name = self.__class__.__name__
    #     if VERBOSE:
    #         if isinstance(txt, str):
    #             print '[' + class_name + '] ' + txt
    #         else:
    #             print '[' + class_name + '] ' + repr(txt)

    def _update_res(self, gen_count):
        # Save estimates
        generation_estimates = self.pop.get_all_estimates_and_errors()
        generation_estimates['generation'] = gen_count
        self.all_estim_and_err = pd.concat([self.all_estim_and_err, generation_estimates])

        # Append error lists
        self.fittest_errors.append(self.pop.get_fittest_error())

    def _get_best_from_gen(self, generation):
        """
        Gets fittest individuals (parameter sets) from the chosen generation.

        :param generation: int (generation number)
        :return: DataFrame
        """
        df = self.all_estim_and_err.copy()
        df.index = df['generation']
        # Select individuals with minimum error from the chosen individuals
        fittest = df.loc[df['error'] == df.loc[generation]['error'].min()].loc[generation]
        # Check how many individuals found
        if isinstance(fittest, pd.DataFrame):
            # More than 1 found...
            # Get the first one
            fittest = fittest.iloc[0]
        elif isinstance(fittest, pd.Series):
            # Only 1 found...
            pass
        # Drop column 'individual'
        fittest = fittest.drop('individual')

        return fittest

    def _get_n_param(self):
        """
        Returns number of estimated parameters

        :return: int
        """
        return len(self.get_estimates())

"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from modestpy.log_init import LogInit
LOG_INIT = LogInit(__name__)
LOGGER = LOG_INIT.get_logger()

import random
import pandas as pd
import numpy as np
import copy
from modestpy.estim.error import calc_err


class Individual:

    COM_POINTS = 500

    def __init__(self, est_objects, population, genes=None):

        # Reference to the population object
        self.population = population

        # Assign variables shared across the population
        self.ideal = population.ideal
        self.model = population.model

        # Adjust COM_POINTS
        Individual.COM_POINTS = len(self.ideal) - 1 # CVODE solver complained without "-1"

        # Deep copy EstPar instances to avoid sharing between individuals
        self.est_par_objects = copy.deepcopy(est_objects)

        # Generate genes
        if not genes:
            est_names = Individual._get_names(self.est_par_objects)
            self.genes = Individual._random_genes(est_names)
        else:
            self.genes = copy.deepcopy(genes)

        # Update parameters
        self._update_parameters()

        # Individual result
        self.result = None
        self.error = None

    # Main methods ------------------------------
    def calculate(self):
        # Just in case, individual result and error are cleared before simulation
        self.reset()
        # Important to set estimated parameters just before simulation,
        # because all individuals share the same model instance
        self.model.set_param(self.est_par_df)
        # Simulation
        self.result = self.model.simulate(Individual.COM_POINTS)
        # Make sure the returned result is not empty
        assert self.result.empty is False, 'Empty result returned from simulation... (?)'
        # Calculate error
        self.error = calc_err(self.result, self.ideal)

    def reset(self):
        self.result = None
        self.error = None
        self.est_par_objects = copy.deepcopy(self.est_par_objects)

    def set_gene(self, name, value):
        self.genes[name] = value
        self._update_parameters()

    def get_gene(self, name):
        return self.genes[name]

    def get_estimates(self, as_dict=False):
        """
        :param as_dict: boolean (True to get dictionary instead DataFrame)
        :return: DataFrame with estimated parameters
        """
        df = pd.DataFrame()
        for par in self.est_par_objects:
            df[par.name] = np.array([par.value])
        if as_dict:
            return df.to_dict()
        else:
            return df

    def get_estimates_and_error(self):
        estimates = self.get_estimates()
        estimates['error'] = self.error['tot']
        return estimates

    def get_clone(self):
        clone = Individual(self.est_par_objects, self.population, self.genes)
        return clone

    # Private methods ---------------------------
    def _update_parameters(self):
        # Calculate parameter values
        self.est_par_objects = self._calc_parameters(self.genes)
        # Convert estimated parameters to dataframe
        self.est_par_df = Individual._est_pars_2_df(self.est_par_objects)

    @staticmethod
    def _est_pars_2_df(est_pars):
        df = pd.DataFrame()
        for p in est_pars:
            df[p.name] = np.array([p.value])
        return df

    def _calc_parameters(self, genes):
        """
        Calculates parameters based on genes and limits.
        :return: None
        """
        for par in self.est_par_objects:
            gene = genes[par.name]
            par.value = par.lo + gene * (par.hi - par.lo)
        return self.est_par_objects

    @staticmethod
    def _random_genes(par_names):
        """
        Generates random genes.
        :return: None
        """
        genes = dict()
        for par in par_names:
            g = 0
            while g == 0:  # Because random.random() can return 0
                g = random.random()
            genes[par] = g
        return genes

    @staticmethod
    def _get_names(est_params):
        names = list()
        for par in est_params:
            names.append(par.name)
        return names

    # Overriden methods --------------------------
    def __str__(self):
        s = 'Individual ('
        for par in self.est_par_objects:
            s += par.name + '={0:.3f}'.format(par.value)
            s += ', '
        # Delete trailing comma
        s = s[:-2]
        s += '), err='
        if self.error:
            s += '{:.4f} '.format(self.error['tot'])
        else:
            s += 'None'
        return s



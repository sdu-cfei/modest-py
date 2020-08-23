# -*- coding: utf-8 -*-
"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""
import logging
import random
import pandas as pd
import numpy as np
import copy
from modestpy.estim.error import calc_err


class Individual(object):

    def __init__(self, est_objects, population, genes=None,
                 use_init_guess=False, ftype='NRMSE'):
        """
        Individual can be initialized using `genes` OR initial guess
        in `est_objects` (genes are inferred from parameters and vice versa).
        Otherwise, random genes are assumed.

        :param est_objects: List of EstPar objects with estimated parameters
        :type est_objects: list(EstPar)
        :param Population population: Population instance
        :param genes: Genes (can be also inferred from `parameters`)
        :type genes: dict(str: float)
        :param bool use_init_guess: If True, use initial guess from
                                    `est_objects`
        :param str ftype: Cost function type, 'RMSE' or 'NRMSE'
        """

        self.logger = logging.getLogger(type(self).__name__)

        # Reference to the population object
        self.population = population

        # Assign variables shared across the population
        self.ideal = population.ideal
        self.model = population.model

        # Cost function type
        self.ftype = ftype

        # Deep copy EstPar instances to avoid sharing between individuals
        self.est_par_objects = copy.deepcopy(est_objects)

        # Generate genes
        if not genes and not use_init_guess:
            # Generate random genes
            est_names = Individual._get_names(self.est_par_objects)
            self.genes = Individual._random_genes(est_names)
        elif genes and not use_init_guess:
            # Use provided genes
            self.genes = copy.deepcopy(genes)
        elif use_init_guess and not genes:
            # Infer genes from parameters
            self.genes = dict()
            for p in self.est_par_objects:
                self.genes[p.name] = (p.value - p.lo) / (p.hi - p.lo)
                assert self.genes[p.name] >= 0. and self.genes[p.name] <= 1., \
                    'Initial guess outside the bounds'
        else:
            msg = 'Either genes or parameters have to be None'
            self.logger.error(msg)
            raise ValueError(msg)

        # Update parameters
        self._update_parameters()

        # Individual result
        self.result = None
        self.error = None

    # Main methods ------------------------------
    def calculate(self):
        # Just in case, individual result and error
        # are cleared before simulation
        self.reset()
        # Important to set estimated parameters just before simulation,
        # because all individuals share the same model instance
        self.model.set_param(self.est_par_df)
        # Simulation
        self.result = self.model.simulate()
        # Make sure the returned result is not empty
        assert self.result.empty is False, \
            'Empty result returned from simulation... (?)'
        # Calculate error
        self.logger.debug("Calculating error ({}) in individual {}"
                          .format(self.ftype, self.genes))
        self.error = calc_err(self.result, self.ideal, ftype=self.ftype)

    def reset(self):
        self.result = None
        self.error = None
        self.est_par_objects = copy.deepcopy(self.est_par_objects)

    def set_gene(self, name, value):
        self.genes[name] = value
        self._update_parameters()

    def get_gene(self, name):
        return self.genes[name]

    def get_sorted_gene_names(self):
        return sorted(self.genes.keys())

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
        estimates['_error_'] = self.error['tot']
        return estimates

    def get_clone(self):
        clone = Individual(self.est_par_objects, self.population,
                           self.genes, ftype=self.ftype)
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
        :return: dict(str: float)
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

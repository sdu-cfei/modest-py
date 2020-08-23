# -*- coding: utf-8 -*-

"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from modestpy.estim.ga.individual import Individual
from modestpy.fmi.model import Model
import pandas as pd
import copy


class Population(object):

    def __init__(self, fmu_path, pop_size, inp, known, est, ideal,
                 init=True, opts=None, ftype='NRMSE', init_pop=None):
        """
        :param fmu_path: string
        :param pop_size: int
        :param inp: DataFrame
        :param known: DataFrame
        :param est: List of EstPar objects
        :param ideal: DataFrame
        :param init: bool
        :param dict opts: Additional FMI options to be passed to the simulator
        :param string ftype: Cost function type. Currently 'NRMSE' or 'RMSE'.
        :param DataFrame init_pop: Initial population, DataFrame with initial
                                   guesses for estimated parameters
        """
        self.logger = logging.getLogger(type(self).__name__)

        # Initialize list of individuals
        self.individuals = list()

        # Assign attributes
        self.fmu_path = fmu_path
        self.pop_size = pop_size
        self.inputs = inp
        self.known_pars = known
        self.estpar = est
        self.outputs = [var for var in ideal]
        self.ideal = ideal
        self.ftype = ftype

        # Instantiate model
        self.model = None

        if init:
            # Instiate individuals before initialization
            self.instantiate_model(opts=opts)
            self._initialize(init_pop)
            self.calculate()

    def instantiate_model(self, opts):
        self.model = Model(self.fmu_path, opts=opts)
        self.model.set_input(self.inputs)
        self.model.set_param(self.known_pars)
        self.model.set_outputs(self.outputs)

    def add_individual(self, indiv):
        assert isinstance(indiv, Individual), \
            'Only Individual instances allowed...'
        indiv.reset()
        self.individuals.append(indiv)

    def calculate(self):
        for i in self.individuals:
            i.calculate()

    def size(self):
        return self.pop_size

    def get_fittest(self):
        fittest = self.individuals[0]
        for ind in self.individuals:
            if ind.error['tot'] < fittest.error['tot']:
                fittest = ind
        fittest = copy.copy(fittest)
        return fittest

    def get_fittest_error(self):
        return self.get_fittest().error['tot']

    def get_population_errors(self):
        err = list()
        for i in self.individuals:
            err.append(i.error['tot'])
        return err

    def get_fittest_estimates(self):
        return self.get_fittest().get_estimates()

    def get_all_estimates_and_errors(self):
        all_estim = pd.DataFrame()
        i = 1
        for ind in self.individuals:
            i_estim = ind.get_estimates_and_error()
            i_estim['individual'] = i
            all_estim = pd.concat([all_estim, i_estim])
            i += 1
        return all_estim

    def get_estpars(self):
        """Returns EstPar list"""
        return self.estpar

    def _initialize(self, init_pop=None):
        self.logger.debug('Initialize population with init_pop=\n{}'
                          .format(init_pop))

        self.individuals = list()

        # How to initialize? Random or explicit initial guess?
        if init_pop is not None:
            assert len(init_pop.index) == self.pop_size, \
                "Population size does not match initial guess {} != {}" \
                .format(init_pop.index.size, self.pop_size)
            init_guess = True
        else:
            init_guess = False

        for i in range(self.pop_size):
            # --------------------------------------------------> BUG HERE
            if init_guess:
                # Update value in EstPar objects with the next initial guess
                for n in range(len(self.estpar)):
                    self.estpar[n].value = \
                        init_pop.loc[i, self.estpar[n].name]
                    self.logger.debug(
                        'Individual #{} <- {}'
                        .format(i, self.estpar[n]))

            self.add_individual(
                Individual(est_objects=self.estpar, population=self,
                           ftype=self.ftype, use_init_guess=init_guess)
                )
            # <-------------------------------------------------- BUG HERE

    def __str__(self):
        fittest = self.get_fittest()
        s = repr(self)
        s += '\n'
        s += 'Number of individuals: ' + str(len(self.individuals)) + '\n'
        if len(self.individuals) > 0:
            for i in self.individuals:
                s += str(i) + '\n'
            s += '-' * 110 + '\n'
            s += 'Fittest: ' + str(fittest) + '\n'
            s += '-' * 110 + '\n'
        else:
            s += 'EMPTY'
        return s

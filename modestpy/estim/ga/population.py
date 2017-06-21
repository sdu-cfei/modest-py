"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from modestpy.log_init import LogInit
LOG_INIT = LogInit(__name__)
LOGGER = LOG_INIT.get_logger()

from individual import Individual
from modestpy.estim.model import Model
import pandas as pd
import copy


class Population:

    def __init__(self, fmu_path, pop_size, inp, known, est, ideal, init=True):
        """
        :param fmu_path: string
        :param pop_size: int
        :param inp: DataFrame
        :param known: DataFrame
        :param est: dict
        :param ideal: DataFrame
        :param init: bool
        """

        # Initialize list of individuals
        self.individuals = list()

        # Assign attributes
        self.fmu_path = fmu_path
        self.pop_size = pop_size
        self.inputs = inp
        self.known_pars = known
        self.est_obj = est
        self.outputs = [var for var in ideal]
        self.ideal = ideal

        # Instantiate model
        self.model = None

        if init:
            self.instantiate_model()  # Must be done before initialization of individuals
            self._initialize()
            self.calculate()

    def instantiate_model(self):
        self.model = Model(self.fmu_path)
        self.model.set_input(self.inputs)
        self.model.set_param(self.known_pars)
        self.model.set_outputs(self.outputs)

    def add_individual(self, indiv):
        assert isinstance(indiv, Individual), 'Only Individual instances allowed...'
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

    def _initialize(self):
        self.individuals = list()
        for i in range(self.pop_size):
            self.add_individual(Individual(est_objects=self.est_obj, population=self))

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

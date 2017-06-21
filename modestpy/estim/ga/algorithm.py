"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

# Logger
from modestpy.log_init import LogInit
LOG_INIT = LogInit(__name__)
LOGGER = LOG_INIT.get_logger()

from population import Population
import random

# Constants controlling the evolution
UNIFORM_RATE = 0.5  # affects crossover
MUT_RATE = 0.05  # standard mutation rate
MUT_RATE_INC = 0.3  # increased mutation rate when population diversity is low
INC_MUT_PROP = 0.7  # proportion of population undergoing increased mutation with slight changes of genes
MAX_CHANGE = 1  # [%] maximum change of gene in increased mutation (change in parameter depends on lo/hi limits)
TOURNAMENT_SIZE = 10  # number of individuals in the tournament
DIVERSITY_LIM = 0.7  # if DIVERSITY_LIM * 100% of individuals is the same, increased mutation is turned on
ELITISM = True  # if True, the fittest individuals always goes to the next generation

# Printing on the screen
VERBOSE = True


def evolve(pop):
    """
    Evolves the population.

    :param pop: Population
    :return: Population
    """
    new_pop = Population(fmu_path=pop.fmu_path,
                         pop_size=pop.size(),
                         inp=pop.inputs,
                         known=pop.known_pars,
                         est=pop.est_obj,
                         ideal=pop.ideal,
                         init=False)

    elite_offset = 0
    if ELITISM:
        new_pop.add_individual(pop.get_fittest())
        elite_offset = 1
        LOGGER.info('Elitism = True, best individual saved (index=0)')

    # Crossover
    for i in range(elite_offset, new_pop.size()):
        ind1 = tournament_selection(pop, TOURNAMENT_SIZE)
        ind2 = tournament_selection(pop, TOURNAMENT_SIZE)
        child = crossover(ind1, ind2, UNIFORM_RATE)
        new_pop.add_individual(child)

    # Mutation
    # Check population diversity
    if is_population_diverse(new_pop, DIVERSITY_LIM):
        # Low mutation rate, completely random new values
        for i in range(elite_offset, new_pop.size()):
            mutation(new_pop.individuals[i], MUT_RATE)
    else:
        # Population is not diverse
        for i in range(elite_offset, new_pop.size()):
            if random.random() < INC_MUT_PROP:
                # Increased mutation rate, slightly changed values
                LOGGER.info('Increased mutation, SLIGHT changes in genes, individual no. ' + str(i))
                slight_mutation(new_pop.individuals[i], MUT_RATE_INC, MAX_CHANGE)
            else:
                # Increased mutation rate, completely random new values
                LOGGER.info('Increased mutation, RANDOM changes in genes, individual no. ' + str(i))
                mutation(new_pop.individuals[i], MUT_RATE_INC)

    # Calculate
    new_pop.calculate()

    # Return
    return new_pop


def is_population_diverse(pop, diversity_lim):
    """
    Check if the population is diverse. Returns true if the share of identical individuals
    in the population is higher than ``diversity_lim``.

    :param pop: Population
    :param diversity_lim: float (0-1), minimum share of identical individuals defining non-diverse population
    :return: boolean
    """
    identical_count = 0
    total_count = len(pop.individuals)
    genes = []

    for ind in pop.individuals:
        genes.append(ind.genes)

    # Count duplicates
    for row in genes:
        dup = genes.count(row)
        if dup > identical_count:
            identical_count = dup

    # Check against the limit
    is_diverse = True
    if (float(identical_count) / float(total_count)) > diversity_lim:
        is_diverse = False
    return is_diverse


def crossover(ind1, ind2, uniformity):
    """
    Crossover operation. The child takes genes from both parents.
    If ``uniformity`` =0.5 then on average the child's gene pool is composed from 50%/50% of ``ind1``/``ind2``.

    :param ind1: Individual (parent 1)
    :param ind2: Individual (parent 2)
    :param uniformity: float, uniformity rate
    :return: Individual (child)
    """
    # Avoid working on the same objects!
    # Otherwise, changing child's genes
    # affects parent's genes (since genes are stored in a dict).
    # The parent might be used again in another crossover operation,
    # so it must not be modified.
    child = ind1.get_clone()
    i1_clone = ind1.get_clone()
    i2_clone = ind2.get_clone()

    for name in child.genes:
        if random.random() <= uniformity:
            child.set_gene(name, i1_clone.genes[name])
        else:
            child.set_gene(name, i2_clone.genes[name])

    return child


def mutation(ind, mut_rate):
    """
    Standard mutation. Mutates genes in place.

    :param ind: Individual
    :param mut_rate: mutation rate
    :return: None
    """
    for g_name in ind.genes:
        if random.random() < mut_rate:
            ind.set_gene(g_name, random.random())


def slight_mutation(ind, mut_rate, max_change):
    """
    Slightly mutates the genes.

    :param ind: Individual
    :param mut_rate: float, mutation rate
    :param max_change: float (0-100), maximum allowed percentage change of the gene
    :return: None
    """
    for g_name in ind.genes:
        if random.random() < mut_rate:
            value = ind.genes[g_name]
            new_value = value + random.uniform(-1., 1.) * max_change / 100
            if new_value > 1.:
                new_value = 1.
            if new_value < 0.:
                new_value = 0.
            ind.set_gene(g_name, new_value)


def tournament_selection(pop, tournament_size):
    # Create tournament population
    t_pop = Population(pop.fmu_path, tournament_size, pop.inputs, pop.known_pars,
                       pop.est_obj, pop.ideal, init=False)
    # For each place in the tournament get a random individual
    for i in range(tournament_size):
        rand_index = random.randint(0, pop.size()-1)
        t_pop.individuals.append(pop.individuals[rand_index])

    return t_pop.get_fittest()


def info(txt):
    if VERBOSE:
        if isinstance(txt, str):
            print '[ALGORITHM]', txt
        else:
            print '[ALGORITHM]', repr(txt)

"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import pandas as pd
import numpy as np


class EstPar:
    """
    Estimated parameter for PS.
    """
    def __init__(self, name, lo=None, hi=None, value=None):
        self.name = name
        self.value = value
        self.lo = lo  # Lower limit
        self.hi = hi  # Upper limit

    def __str__(self):
        s = self.name + '={:.4f}'.format(self.value) + ' (' + str(self.lo) + '-' + str(self.hi) + ')'
        return s


def estpars_2_df(est_pars):
    """Converts list of EstPar instances into DataFrame.

    :param est_pars: list of EstPar instances
    :return: None"""
    df = pd.DataFrame()
    for p in est_pars:
        df[p.name] = np.array([p.value])
    return df


def df_2_estpars(df):
    """Converts single-row data frame into a list of EstPar instances.
    hi/lo limits are unknown and assumed to be +/- inf.

    :param df: DataFrame with parameters (single row)
    :return: list of EstPar instances"""
    estpars = []
    for p in df:
        ep = EstPar(p, float('-inf'), float('+inf'), df[p][0])
        estpars.append(ep)
    return estpars


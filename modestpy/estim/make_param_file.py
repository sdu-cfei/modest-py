"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import pandas as pd
import copy


def make_param_file(est, known, path):
    """
    Saves parameter file from ``est`` and ``known`` dictionaries.

    :param est: Dictionary, key=parameter_name, value=tuple (guess value, lo limit, hi limit), guess can be None
    :param known: Dictionary, key=parameter_name, value=value
    :param path: string, path to the file
    :return: None
    """
    par = copy.copy(known)
    for p in est:
        par[p] = est[p][0]
    for p in par:
        par[p] = [par[p]]
    par = pd.DataFrame.from_dict(par)
    par.to_csv(path, index=False)

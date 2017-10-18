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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fmi.model import Model

# Description
# ===========
# This script can be used to simulate an FMU, 
# e.g. to produce results from a model 
# with known inputs and parameters.
if __name__ == "__main__":
    
    model = Model('./tests/resources/simple2R1C/Simple2R1C.fmu')

    # Inputs
    inp = pd.DataFrame()
    inp['time'] = np.arange(3600) * 60.
    inp['Ti1'] = np.tanh(np.linspace(-3, 3, 3600)) * 10. + 273.15
    inp['Ti2'] = np.sin(np.arange(3600)/200.) * 10. + 273.15
    inp = inp.set_index('time')
    inp.to_csv('./tests/resources/simple2R1C/inputs.csv')

    model.inputs_from_csv('./tests/resources/simple2R1C/inputs.csv')
    model.parameters_from_csv('./tests/resources/simple2R1C/parameters.csv')
    model.specify_outputs(['T'])
    res = model.simulate(len(inp.index) - 1)
    
    res.to_csv('./tests/resources/simple2R1C/result.csv')

    res.plot()
    plt.show()

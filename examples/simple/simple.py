"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.

Author: Krzysztof Arendt
"""

import pandas as pd
import os
import json
from modestpy.estim.learnman import LearnMan


if __name__ == "__main__":
    """
    This file is supposed to be run from the root directory.
    Otherwise the paths have to be corrected.
    """
    # DATA PREPARATION ==============================================
    # Resources
    fmu_path = os.path.join('examples', 'simple', 'resources', 'Simple2R1C.fmu')
    inp_path = os.path.join('examples', 'simple', 'resources', 'inputs.csv')
    ideal_path = os.path.join('examples', 'simple', 'resources', 'result.csv')
    est_path = os.path.join('examples', 'simple', 'resources', 'est.json')
    known_path = os.path.join('examples', 'simple', 'resources', 'known.json')

    # Working directory
    workdir = os.path.join('examples', 'simple', 'workdir')
    if not os.path.exists(workdir):
        os.mkdir(workdir)
        assert os.path.exists(workdir), "Work directory does not exist"

    # Load inputs
    inp = pd.read_csv(inp_path).set_index('time')

    # Load measurements (ideal results)
    ideal = pd.read_csv(ideal_path).set_index('time')

    # Load definition of estimated parameters (name, initial value, bounds)
    with open(est_path) as f:
        est = json.load(f)
    
    # Load definition of known parameters (name, value)
    with open(known_path) as f:
        known = json.load(f)
    
    # MODEL IDENTIFICATION ==========================================
    # Learning session settings
    lp_n = 5                    # Number of learning periods
    lp_bounds = (0., 150000.)   # Time frame within which learning periods are selected
    lp_length = 50000.          # Single learning period length
    sensors = {'Tstart': 'T'}   # Parameters defining initial condition (based on measurements)

    # Validation session settings
    vp = (150000., 215940.)     # Validation period

    # Learning session
    lm = LearnMan(workdir, fmu_path, inp, known, est, ideal)
    lm.ic_from_sensors(sensors)
    lm.select_learning_periods(lp_n, lp_length, lp_bounds)
    estimates = lm.estimate()

    # Validation with average parameters (from all estimation runs)
    lm.select_validation_period(vp)
    lm.validate(lm.get_avg_est())

    # Save estimates and plots
    lm.save_best_est('best_estimates.csv', incl_known=False)
    lm.save_avg_est('average_estimates.csv', incl_known=False)
    lm.save_plots()

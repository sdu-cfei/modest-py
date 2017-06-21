"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import os
import shutil
from pymodelica import compile_fmu


def compile(model_name, mo_path, fmu_path=None):
    """
    Compiles FMU 2.0 CS from a MO file (Modelica model).

    :param model_name: string, path to the model in the MO file
    :param mos_path: string, path to the input MO file
    :param fmu_path: string or None, path to the output FMU file; "CWD/model_name.fmu" if None
    :return: string, path to the resulting FMU
    """
    # opts = {'nle_solver_min_tol': 1e-10}
    opts = {}

    fmu = compile_fmu(model_name, mo_path, target='cs', version='2.0', \
                      compiler_options=opts)
    std_fmu_path = os.path.join(os.getcwd(), model_name.replace('.', '_') + '.fmu')
    if fmu_path is not None:
        print "Moving FMU to: {}".format(fmu_path)
        shutil.move(std_fmu_path, fmu_path)
        return fmu_path
    return std_fmu_path

# Example
if __name__ == "__main__":
    mo_path = '/home/krza/github/modest/tests/resources/simple2R1C/Simple2R1C.mo'
    model_name = "Simple2R1C"
    fmu_path = '/home/krza/github/modest/tests/resources/simple2R1C/Simple2R1C.fmu'
    compile(model_name, mo_path, fmu_path)
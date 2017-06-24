"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.

Author: Krzysztof Arendt
"""

import os
import shutil
from pymodelica import compile_fmu
from modestpy.utilities.sysarch import get_sys_arch


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

    platform = get_sys_arch()

    ## FMU from examples
    # mo_path = os.path.join('.', 'examples', 'simple', 'resources', 'Simple2R1C.mo')
    # fmu_path = os.path.join('.', 'examples', 'simple', 'resources', 'Simple2R1C_{}.fmu'.format(platform))
    # model_name = "Simple2R1C"

    ## FMU from tests
    mo_path = os.path.join('.', 'tests', 'resources', 'simple2R1C', 'Simple2R1C.mo')
    fmu_path = os.path.join('.', 'tests', 'resources', 'simple2R1C', 'Simple2R1C_{}.fmu'.format(platform))
    model_name = "Simple2R1C"
    
    # Compilation
    compile(model_name, mo_path, fmu_path)
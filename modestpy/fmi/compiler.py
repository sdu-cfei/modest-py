"""DEPRECATED.

WARNING, THIS FILE RELIES ON PYMODELICA WHICH WILL NOT BE
INCLUDED IN THE DEPENDENCIES IN THE NEXT RELEASE.

Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

from modestpy.utilities.sysarch import get_sys_arch

try:
    from pymodelica import compile_fmu
except ImportError as e:
    raise ImportError("pymodelica is required to run this script!")


def mo_2_fmu(model_name, mo_path, fmu_path=None):
    """
    Compiles FMU 2.0 CS from a MO file (Modelica model).

    :param model_name: string, path to the model in the MO file
    :param mos_path: string, path to the input MO file
    :param fmu_path: string or None, path to the output FMU file;
                     "CWD/model_name.fmu" if None
    :return: string, path to the resulting FMU
    """
    # opts = {'nle_solver_min_tol': 1e-10}
    opts = {}

    compile_fmu(model_name, mo_path, target="cs", version="2.0", compiler_options=opts)

    std_fmu_path = os.path.join(os.getcwd(), model_name.replace(".", "_") + ".fmu")
    if fmu_path is not None:
        print("Moving FMU to: {}".format(fmu_path))
        shutil.move(std_fmu_path, fmu_path)
        return fmu_path
    return std_fmu_path


# Example
if __name__ == "__main__":

    platform = get_sys_arch()

    # FMU from examples
    # mo_path = os.path.join('.', 'examples', 'simple', 'resources',
    #                        'Simple2R1C.mo')
    # fmu_path = os.path.join('.', 'examples', 'simple', 'resources',
    #                         'Simple2R1C_{}.fmu'.format(platform))
    # model_name = "Simple2R1C"

    # FMU from tests
    mo_path = os.path.join(".", "tests", "resources", "simple2R1C", "Simple2R1C.mo")
    fmu_path = os.path.join(
        ".", "tests", "resources", "simple2R1C", "Simple2R1C_{}.fmu".format(platform)
    )
    model_name = "Simple2R1C"

    # Compilation
    mo_2_fmu(model_name, mo_path, fmu_path)

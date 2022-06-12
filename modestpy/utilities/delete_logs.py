"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.
This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def delete_logs(directory=os.getcwd()):
    """
    Deletes log files in the directory.

    :param directory: string, path to the directory
    :return: None
    """
    content = os.listdir(directory)
    for el in content:
        if el.split(".")[-1] == "log":
            # This is a log file
            fpath = os.path.join(directory, el)
            print("Removing {}".format(fpath))
            try:
                os.remove(fpath)
            except WindowsError as e:
                print(e.message)
    return

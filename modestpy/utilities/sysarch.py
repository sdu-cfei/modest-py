"""
Copyright (c) 2017, University of Southern Denmark
All rights reserved.

This code is licensed under BSD 2-clause license.
See LICENSE file in the project root for license terms.
"""

import platform


def get_sys_arch():
    """
    Returns system architecture:
    'win32', 'win64', 'linux32', 'linux64' or 'unknown'.

    Modestpy supports only windows and linux at the moment,
    so other platforms are not recognized.

    .. warning:: It relies on the ``platform.architecture`` function
                 which on Windows 64bit with 32bit Python interpereter
                 returns 32bit. However since JModelica on Windows
                 works by default as 32bit and requires Python 32bit,
                 for the time being this solution is accepted.

    Returns
    -------
    str or None
    """
    sys_type = platform.system()
    bit_arch = platform.architecture()[0]
    
    sys = None
    bits = None

    if ('win' in sys_type) or ('Win' in sys_type) or ('WIN' in sys_type):
        sys = 'win'
    elif ('linux' in sys_type) or ('Linux' in sys_type) or ('LINUX' in sys_type):
        sys = 'linux'
    
    if '32' in bit_arch:
        bits = '32'
    elif '64' in bit_arch:
        bits = '64'
    
    if (sys and bits):
        sys_bits = sys + bits
    else:
        sys_bits = None

    return sys_bits


if __name__ == "__main__":
    # Test
    print get_sys_arch()

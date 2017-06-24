import platform


def get_sys_arch():
    """
    Returns system architecture:
    'win32', 'win64', 'linux32', 'linux64' or 'unknown'.

    Modestpy supports only windows and linux at the moment,
    so other platforms are not recognized.

    Returns
    -------
    str
    """
    arch = platform.architecture()
    bits = None
    sys = None

    if '32' in arch[0]:
        bits = '32'
    elif '64' in arch[0]:
        bits = '64'

    if ('win' in arch[1]) or ('Win' in arch[1]) or ('WIN' in arch[1]):
        sys = 'win'
    elif ('linux' in arch[1]) or ('Linux' in arch[1]) or ('LINUX' in arch[1]):
        sys = 'linux'
    
    if (sys and bits):
        sys_bits = sys + bits
    else:
        sys_bits = 'not supported'
    
    return sys_bits


if __name__ == "__main__":
    # Test
    print get_sys_arch()

FMI-compliant Model Estimation in Python
========================================

.. figure:: /docs/img/modest-logo.png
   :alt: modestpy

Description
-----------

**modestpy** facilitates parameter estimation in models compliant with
`Functional Mock-up Interface <https://fmi-standard.org/>`__. The
estimation can be performed on a single or multiple random learning
periods (to avoid overfitting). Currently the estimation is based on
`genetic algorithm <https://en.wikipedia.org/wiki/Genetic_algorithm>`__
(GA) and `pattern
search <https://en.wikipedia.org/wiki/Pattern_search_(optimization)>`__
(PS) methods. The user is free to choose whether to use GA, PS or
combined GA+PS. Both methods can deal with non-continuous and
non-differentiable models.

The package is compatible with Python 2.7 due to its reliance on
`PyFMI <https://pypi.python.org/pypi/PyFMI>`__.

Installation
------------

The package can be installed from PyPI:

::

    python -m pip install modestpy

On Ubuntu 16.04 or newer, if you get a permission error, you might have to install in the user local directory:

::

    python -m pip install --user modestpy

To get the latest development version download directly from GitHub repository:

::

    git clone https://github.com/sdu-cfei/modest-py modestpy
    cd modestpy
    python -m pip install .

or 

::

    python -m pip install https://github.com/sdu-cfei/modest-py/archive/master.zip
    
**modestpy** relies on `PyFMI <https://pypi.python.org/pypi/PyFMI>`__
for FMU simulation, which is advised to be installed as part of
`JModelica <http://jmodelica.org/>`__ (current version 2.0). Other
dependencies will be installed automatically.

To run tests:

.. code:: python

    >>> from modestpy.test import run
    >>> run.tests()

or

::

    cd <project_directory>
    python ./bin/test.py


Usage
-----

Users are supposed to call only the high level API included in
``modestpy.Estimation``. The API is fully discussed in `this
wiki <https://github.com/sdu-cfei/modest-py/wiki/modestpy-API>`__. You
can also check out this `simple example </examples/simple>`__. The basic
usage is as follows:

.. code:: python

    >>> from modestpy import Estimation
    >>> session = Estimation(workdir, fmu_path, inp, known, est, ideal)
    >>> estimates = session.estimate()
    >>> err, res = session.validate()

More control is possible via optional arguments, as discussed in the `documentation 
https://github.com/sdu-cfei/modest-py/wiki/modestpy-API>`__.

``modestpy`` automatically saves results in the working
directory including csv files with estimates and some useful plots,
e.g.:

1) Error evolution in combined GA+PS estimation (dots represent switch
   from GA to PS): |Error-evolution|

2) Visualization of GA evolution: |GA-evolution|

3) Scatter matrix plot for interdependencies between parameters:
   |Intedependencies|

Author
------

`Krzysztof Arendt <https://github.com/krzysztofarendt>`__, Center for
Energy Informatics, University of Southern Denmark

License
-------

Copyright (c) 2017, University of Southern Denmark. All rights reserved.

This code is licensed under BSD 2-clause license. See
`LICENSE </LICENSE>`__ file in the project root for license terms.

.. |Error-evolution| image:: /docs/img/err_evo.png
.. |GA-evolution| image:: /docs/img/ga_evolution.png
.. |Intedependencies| image:: /docs/img/all_estimates.png


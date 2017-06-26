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

The project is compatible with 2.7, because it relies on
`PyFMI <https://pypi.python.org/pypi/PyFMI>`__ which is available only
for Python 2.7.

Installation
------------

The package can be installed from PyPI:

::

    pip install modestpy

To get the latest development version as well as get access to test resources
download directly from GitHub repository:

::

    git clone https://github.com/sdu-cfei/modest-py modestpy
    cd modestpy
    pip install .

**modestpy** relies on `PyFMI <https://pypi.python.org/pypi/PyFMI>`__
for FMU simulation, which is advised to be installed as part of
`JModelica <http://jmodelica.org/>`__ (current version 2.0). Other
dependencies are included in `requirements.txt </requirements.txt>`__.

To run tests (if the development version was downloaded):

.. code:: python

    >>> from modestpy.test import run
    >>> run.tests()

From command line (assuming that you are in the project root directory):

.. code::

    python test/run.py

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

To get more control, use the optional arguments:

.. code:: python

    >>> from modestpy import Estimation
    >>> session = Estimation(workdir, fmu_path, inp, known, est, ideal,
                             lp_n=3, lp_len=3600, lp_frame=(0, 86400), vp=(86400, 172800),
                             ic_param={'Tstart': 'T'}, ga_iter=30, ps_iter=30)
    >>> estimates = session.estimate()
    >>> err, res = session.validate(use_type='avg')  # Validation using average estimates from all learning periods
    >>> err, res = session.validate(use_type='best') # Validation using best estimates from all learning periods

``modestpy.Estimation`` automatically saves results in the working
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


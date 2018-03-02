FMI-compliant Model Estimation in Python
========================================

.. figure:: /docs/img/modest-logo.png
   :alt: modestpy

Description
-----------

**modestpy** facilitates parameter estimation in physical models compliant with
`Functional Mock-up Interface <https://fmi-standard.org/>`__. Features:

- combination of global and local search methods (genetic algorithm, pattern search),
- suitable also for non-continuous and non-differentiable models,
- compatible with both Python 2.7 and 3.

Installation with conda
-----------------------

It is advised to use ``conda`` to install the required dependencies.
``modestpy`` itself still has to be installed using ``pip``
(a ``conda`` package is planned in the future), but it can be installed
in the ``conda`` environment.

Create separate environment (optional):

::

    conda create --name modestpy
    conda activate modestpy

Install dependencies:

::

    conda install pandas numpy matplotlib
    conda install -c chria pyfmi
    conda install -c conda-forge pydoe

Install ``modestpy``:

    python -m pip install modestpy

Installation with pip
---------------------

Alternatively, ``pyfmi`` can be installed as part of `JModelica <http://www.jmodelica.org/>`__.

To install ``modestpy`` use ``pip`` (other dependencies will be installed automatically):

::

    python -m pip install modestpy

To get the latest development version download directly from GitHub repository:

::

    python -m pip install https://github.com/sdu-cfei/modest-py/archive/master.zip

Test your installation
----------------------

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
<https://github.com/sdu-cfei/modest-py/wiki/modestpy-API>`__.

``modestpy`` automatically saves results in the working
directory including csv files with estimates and some useful plots,
e.g.:

1) Error evolution in combined GA+PS estimation (dots represent switch
   from GA to PS): |Error-evolution|

2) Visualization of GA evolution: |GA-evolution|

3) Scatter matrix plot for interdependencies between parameters:
   |Intedependencies|

Credits
-------

- `Krzysztof Arendt <https://github.com/krzysztofarendt>`__, Center for
Energy Informatics, University of Southern Denmark

License
-------

Copyright (c) 2017-2018, University of Southern Denmark. All rights reserved.

This code is licensed under BSD 2-clause license. See
`LICENSE </LICENSE>`__ file in the project root for license terms.

.. |Error-evolution| image:: /docs/img/err_evo.png
.. |GA-evolution| image:: /docs/img/ga_evolution.png
.. |Intedependencies| image:: /docs/img/all_estimates.png


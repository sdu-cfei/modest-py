FMI-compliant Model Estimation in Python
========================================

.. figure:: /docs/img/modest-logo.png
   :alt: modestpy

Description
-----------

**modestpy** facilitates parameter estimation in models compliant with
`Functional Mock-up Interface <https://fmi-standard.org/>`__.

Features:

- combination of global and local search methods (genetic algorithm, pattern search, truncated Newton method, L-BFGS-B, sequential least squares),
- suitable also for non-continuous and non-differentiable models,
- can run on multiple cores (GA based on `modestga <https://github.com/krzysztofarendt/modestga>__`),
- Python 3.

Installation with pip (recommended)
-----------------------------------

It is now possible install ModestPy with a single command:

::
    pip install modestpy

Alternatively:

::
    pip install https://github.com/sdu-cfei/modest-py/archive/master.zip

Installation with conda
-----------------------

Conda is installation is less frequently tested, but should work:

::
   conda config --add channels conda-forge
   conda install modestpy

Test your installation
----------------------

The unit tests will work only if you installed modestpy with conda or cloned the project from GitHub. To run tests:

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

Cite
----

To cite ModestPy, please use:

\K. Arendt, M. Jradi, M. Wetter, C.T. Veje, ModestPy: An Open-Source Python Tool for Parameter Estimation in Functional Mock-up Units, *Proceedings of the American Modelica Conference 2018*, Cambridge, MA, USA, October 9-10, 2018.

The preprint version of the conference paper presenting ModestPy is available `here
<https://findresearcher.sdu.dk:8443/ws/portalfiles/portal/145001430/ModestPy_preprint_2018.pdf>`__. The paper was based on v.0.0.8.

License
-------

Copyright (c) 2017-2019, University of Southern Denmark. All rights reserved.

This code is licensed under BSD 2-clause license. See
`LICENSE </LICENSE>`__ file in the project root for license terms.

.. |Error-evolution| image:: /docs/img/err_evo.png
.. |GA-evolution| image:: /docs/img/ga_evolution.png
.. |Intedependencies| image:: /docs/img/all_estimates.png


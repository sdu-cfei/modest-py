# FMI-compliant Model Estimation in Python

![modestpy](/docs/img/modest-logo.png)


## Description

**modestpy** facilitates parameter estimation in models compliant with [Functional Mock-up Interface](https://fmi-standard.org/). The estimation can be performed on a single or multiple random learning periods (to avoid overfitting). Currently the estimation is based on genetic algorithm (GA) and pattern search (PS) methods. The user is free to choose whether to use GA, PS or combined GA+PS.

## Installation

This package is still in its early phase of development. Currently the only way to install it is by cloning this repository and adding its directory to ``PYTHONPATH``:
```
git clone https://github.com/sdu-cfei/modest-py modestpy
```

**modestpy** relies [PyFMI](https://pypi.python.org/pypi/PyFMI) for FMU simulation, which is advised to be installed as part of [JModelica](http://jmodelica.org/) (current version 2.0). Other dependencies are included in [requirements.txt](/requirements.txt).

## Usage

Users are supposed to call only the high level API included in ``modestpy.Estimation``. The API is fully discussed in [this wiki page](https://github.com/sdu-cfei/modest-py/wiki/modestpy-API). You can also check out this [simple example](/examples/simple). The basic usage is as follows:

```python
>>> from modestpy import Estimation
>>> session = Estimation(workdir, fmu_path, inp, known, est, ideal)
>>> estimates = session.estimate()
>>> err, res = session.validate()
```

To get more control, use the optional arguments:
```python
>>> from modestpy import Estimation
>>> session = Estimation(workdir, fmu_path, inp, known, est, ideal,
                         lp_n=3, lp_len=3600, lp_frame=(0, 86400), vp=(86400, 172800),
                         ic_param={'Tstart': 'T'}, ga_iter=30, ps_iter=30)
>>> estimates = session.estimate()
>>> err, res = session.validate(use_type='avg')  # Validation using average estimates from all learning periods
>>> err, res = session.validate(use_type='best') # Validation using best estimates from all learning periods
```

``modestpy.Estimation`` automatically saves results in the working directory including csv files with estimates and some useful plots, e.g.:

1) Error evolution in combined GA+PS estimation (dots represent switch from GA to PS):
![Error-evolution](/docs/img/err_evo.png)

2) Visualization of GA evolution:
![GA-evolution](/docs/img/ga_evolution.png)

3) Scatter matrix plot for interdependencies between parameters:
![Intedependencies](/docs/img/all_estimates.png)

## Author

[Krzysztof Arendt](https://github.com/krzysztofarendt), Center for Energy Informatics, University of Southern Denmark

## License

Copyright (c) 2017, University of Southern Denmark. All rights reserved.

This code is licensed under BSD 2-clause license.
See [LICENSE](/LICENSE) file in the project root for license terms.



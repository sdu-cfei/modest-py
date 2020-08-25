# modestpy
## Introduction

Users are supposed to use only `modestpy.Estimation` class and its two
methods `estimate()` and `validate()`. The class defines a single interface
for different optimization algorithms. Currently, the available algorithms are:
- genetic algorithm (GA),
- pattern search (PS)
- SciPy solvers (e.g. 'TNC', 'L-BFGS-B', 'SLSQP').

The methods can be used in a sequence, e.g. MODESTGA+PS (default),
using the argument `methods`. All estimation settings are set during instantiation.
Results of estimation and validation are saved in the working directory `workdir`
(it must exist).

## Learn by examples

First define the following variables:

* `workdir` (`str`) - path to the working directory (it must exist)
* `fmu_path` (`str`) - path to the FMU compiled for your platform
* `inp` (`pandas.DataFrame`) - inputs, index given in seconds and named `time`
* `est` (`dict(str : tuple(float, float, float))`) - dictionary mapping parameter names to tuples (initial guess, lower bound, upper bound)
* `known` (`dict(str : float)`) - dictionary mapping parameter names to known values
* `ideal` (`pandas.DataFrame`) - ideal solution (usually measurements), index given in seconds and named `time`

Indexes of `inp` and `ideal` must be equal, i.e. `inp.index == ideal.index` must be `True`.
Columns in `inp` and `ideal` must have the same names as model inputs and outputs, respectively.
All model inputs must be present in `inp`, but only chosen outputs may be included in `ideal`.
Data for each variable present in `ideal` are used to calculate the error function that is minimized by **modestpy**.

Now the parameters can be estimated using default settings:

```
python
>>> session = Estimation(workdir, fmu_path, inp, known, est, ideal)
>>> estimates = session.estimate()  # Returns dict(str: float)
>>> err, res = session.validate()   # Returns tuple(dict(str: float), pandas.DataFrame)
```

All results are also saved in `workdir`.

By default all data from `inp` and `ideal` (all rows) are used in both estimation and validation.
To slice the data into separate learning and validation periods, additional arguments need to be defined:

* `lp_n` (`int`) - number of learning periods, randomly selected within `lp_frame`
* `lp_len` (`float`) - length of single learning period
* `lp_frame` (`tuple(float, float)`) - beginning and end of learning time frame
* `vp` (`tuple(float, float)`) - validation period

Often model parameters are used to define the initial conditions in the model,
in example initial temperature. The initial values have to be read from the measured data stored in `ideal`.
You can do this with the optional argument `ic_param`:

* `ic_param` (`dict(str : str)`) - maps model parameters to column names in `ideal`

Estimation algorithms (GA, PS, SQP) can be tuned by overwriting specific keys in `modestga_opts`, `ps_opts` and `scipy_opts`.
The default options are:

TODO

Exemplary estimation using customized settings:

TODO

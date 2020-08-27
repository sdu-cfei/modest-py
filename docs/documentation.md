# modestpy
## Introduction

Users are supposed to use only `modestpy.Estimation` class and its two
methods `estimate()` and `validate()`. The class defines a single interface
for different optimization algorithms. Currently, the available algorithms are:
- parallel genetic algorithm (MODESTGA) - recommended,
- legacy single-process genetic algorithm (GA),
- pattern search (PS),
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

Estimation algorithms (MODESTGA, PS, SQP) can be tuned by overwriting specific keys in `modestga_opts`, `ps_opts` and `scipy_opts`.
The default options are:

```
# Default MODESTGA options
MODESTGA_OPTS = {
    'workers': 3,              # CPU cores to use
    'generations': 50,         # Max. number of generations
    'pop_size': 30,            # Population size
    'mut_rate': 0.01,          # Mutation rate
    'trm_size': 3,             # Tournament size
    'tol': 1e-3,               # Solution tolerance
    'inertia': 100,            # Max. number of non-improving generations
    'ftype': 'RMSE'
}

# Default PS options
self.PS_OPTS = {
    'maxiter':  500,
    'rel_step': 0.02,
    'tol':      1e-11,
    'try_lim':  1000,
    'ftype':    'RMSE'
}

# Default SCIPY options
SCIPY_OPTS = {
    'solver': 'L-BFGS-B',
    'options': {'disp': True,
                'iprint': 2,
                'maxiter': 150,
                'full_output': True},
    'ftype': 'RMSE'
}
```

## Docstrings

```python
class Estimation(object):
    """Public interface of `modestpy`.

    Index in DataFrames `inp` and `ideal` must be named 'time'
    and given in seconds. The index name assertion check is
    implemented to avoid situations in which a user reads DataFrame
    from a csv and forgets to use `DataFrame.set_index(column_name)`
    (it happens quite often...).

    Currently available estimation methods:
        - MODESTGA  - parallel genetic algorithm (default GA in modestpy)
        - GA_LEGACY - single-process genetic algorithm (legacy implementation, discouraged)
        - PS        - pattern search (Hooke-Jeeves)
        - SCIPY     - interface to algorithms available through
                      scipy.optimize.minimize()

    Parameters:
    -----------
    workdir: str
        Output directory, must exist
    fmu_path: str
        Absolute path to the FMU
    inp: pandas.DataFrame
        Input data, index given in seconds and named 'time'
    known: dict(str: float)
        Dictionary with known parameters (`parameter_name: value`)
    est: dict(str: tuple(float, float, float))
        Dictionary defining estimated parameters,
        (`par_name: (guess value, lo limit, hi limit)`)
    ideal: pandas.DataFrame
        Ideal solution (usually measurements),
        index in seconds and named `time`
    lp_n: int or None
        Number of learning periods, one if `None`
    lp_len: float or None
        Length of a single learning period, entire `lp_frame` if `None`
    lp_frame: tuple of floats or None
        Learning period time frame, entire data set if `None`
    vp: tuple(float, float) or None
        Validation period, entire data set if `None`
    ic_param: dict(str, str) or None
        Mapping between model parameters used for IC and variables from
        `ideal`
    methods: tuple(str, str)
        List of methods to be used in the pipeline
    ga_opts: dict
        Genetic algorithm options
    ps_opts: dict
        Pattern search options
    scipy_opts: dict
        SciPy solver options
    ftype: string
        Cost function type. Currently 'NRMSE' (advised for multi-objective
        estimation) or 'RMSE'.
    seed: None or int
        Random number seed. If None, current time or OS specific
        randomness is used.
    default_log: bool
        If true, use default logging settings. Use false if you want to
        use own logging.
    logfile: str
        If default_log=True, this argument can be used to specify the log
        file name
    """
```

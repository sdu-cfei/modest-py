# Simple example: estimation of 3 parameters in R2C1 thermal network

This example shows how to set up a learning session using ``Estimation`` class.

Simulation results are saved in ``modest-py/examples/simple/workdir``.

Notice that the parameters are interdependent and the same model behavior is 
achieved with different sets of estimates (e.g. compare estimates vs. error on 
`scatter.png`). Hence, it is likely that you get different parameters, than
the ones used to produce the measured data (``ideal``). Compare your result
with ``resources/true_parameters.csv`` and look at the validation results. 

---
**Note 1**

This example was tested on Linux 64bit with JModelica 2.0 compiled from
source and on Windows 64bit with Python 32bit and JModelica 2.0 installed
from a binary. If you cannot find an FMU for you platform, please compile
the Modelica model by yourself (from .mo file).

---
**Note 2**

Example ``simple_old_API.py`` shows how to use ``LearnMan`` class.
This class is however deprecated and will be removed in the future.

---

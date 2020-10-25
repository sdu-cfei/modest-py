# Simple example: estimation of 3 parameters in R2C1 thermal network

This example shows how to set up a learning session using ``Estimation`` class.

Simulation results are saved in ``modest-py/examples/simple/workdir``.

Notice that the parameters are interdependent and the same model behavior is 
achieved with different sets of estimates (e.g. compare estimates vs. error on 
`scatter.png`). Hence, it is likely that you get different parameters, than
the ones used to produce the measured data (``ideal``). Compare your result
with ``resources/true_parameters.csv`` and look at the validation results. 

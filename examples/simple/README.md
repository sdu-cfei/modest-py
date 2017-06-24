# Simple example: estimation of 3 parameters in 2R1C thermal network

This example shows how to set up a learning session using ``LearnMan`` class.

Simulation results are saved in `modest-py/examples/simple/workdir`.

Notice that the parameters are interdependent and the same model behavior is 
achieved with different sets of estimates (e.g. compare estimates vs. error on 
`scatter.png`).

.. note:: This example was testes on Linux 64bit with JModelica 2.0 compiled from
          source and on Windows 64bit with Python 32bit and JModelica 2.0 installed
          from a binary. If you cannot find an FMU for you platform, please compile
          the Modelica model by yourself (.mo file).

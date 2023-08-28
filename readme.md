# Interval Signal Temporal Logic from Natural Inclusion Functions

This repository accompanies the paper submitted to L-CSS with ACC "Interval Signal Temporal Logic from Natural Inclusion Functions." The code necessary to generate the figures in our paper is contained in this repository.

This code builds on `stlpy`. See [stlpy](https://stlpy.readthedocs.io/en/latest/)'s documentation.

## Monitoring Example
Python 3.6 or greater must be installed.  
Install the following packages
- numpy
- matplotlib
- scipy
- [npinterval](https://github.com/gtfactslab/npinterval)

Run `python blimp-trace-monitoring.py`.

## Runtime Assurance Example
Install the following package
- pypoman (runtime assurance only)

Follow the instructions to install [gurobi](https://www.gurobi.com/downloads/free-academic-license/) for Python. It is free for academia. Next,
- Clone [the stlpy repo](https://github.com/vincekurtz/stlpy)
- Replace the file `stlpy/solvers/gurobi/gurobi_micp.py` with the one in this repo
- Replace `stlpy/STL/predicate.py` with the one in this repo  
- Replace `stlpy/STL/formula.py` with the one in this repo  
- Run `python setup.py install` from the home directory of the `stlpy` repo.

Run `python double-integrator-interval-control synthesis.py`.

## Optional Installations
- [MikTex](https://miktex.org/) or another LaTeX interpreter, for LaTeX to appear in PyPlot plots.
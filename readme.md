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
    - If npinterval doesn't install correctly, you can also manually put the `interval` folder in the main folder of this repo and it will work fine

To set up stlpy: 
- Clone [the stlpy repo](https://github.com/vincekurtz/stlpy)
- Replace `stlpy/STL/predicate.py` with the one in this repo  
- Replace `stlpy/STL/formula.py` with the one in this repo  
- Run `python setup.py install` from the home directory of the `stlpy` repo.

Run `python blimp-trace-monitoring.py`.

Fot MITLL data, run `mitll-trace-monitoring.py` or `mitll-trace-monitoring.ipynb`

## Control Synthesis Example - Figure 2
Install the following package
- pypoman (control synthesis only)

Follow the instructions to install [gurobi](https://www.gurobi.com/downloads/free-academic-license/) for Python. It is free for academia. Next,
- Clone [the stlpy repo](https://github.com/vincekurtz/stlpy)
- Replace the file `stlpy/solvers/gurobi/gurobi_micp.py` with the one in this repo
- Replace `stlpy/STL/predicate.py` with the one in this repo  
- Replace `stlpy/STL/formula.py` with the one in this repo  
- Run `python setup.py install` from the home directory of the `stlpy` repo.

Run `python double-integrator-interval-control-synthesis.py`.

## Control Synthesis Example - Figure 3
Follow the setup instructions above, except replace `stlpy/solvers/gurobi/gurobi_micp.py` with `gurobi_micp_interval.py`.

Run `python double-integrator-true-robustness-milp.py` for the comparison.

## Optional Installations
- [MikTex](https://miktex.org/) or another LaTeX interpreter, for LaTeX to appear in PyPlot plots.
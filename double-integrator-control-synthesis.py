# File: double-integrator-control-synthesis.py
# Purpose: No interval version of the double integrator interval control synthesis
#          file for the purpose of comparing computation times.
#          Note that this file is faster due to the magical inner workings of Gurobi.
# 
# Author: Luke Baird
# (c) Georgia Institute of Technology, 2023

import numpy as np
import time
from stlpy.systems import LinearSystem
from stlpy.STL import LinearPredicate
from stlpy.solvers import GurobiMICPSolver
import interval
from matplotlib import pyplot as plt # Python plotting
import matplotlib.animation as animation
from matplotlib import rc
from polytopes import Polytope, MatrixMath

from inclusion import d_positive

def main():
    rc('text', usetex=True)

    # dT = 1 # for first run
    dT = 0.25 # uncomment for other plots

    # x0 = np.array([[1.1], [0.]]) # uncomment for constant input first run in the paper
    # x0 = np.array([[0.0], [0.]]) # uncomment for constant input second run
    x0 = np.array([[1.0], [0.]]) # uncomment for sinusoidal input

    sim_length_seconds = 30 # in seconds
    t_np = np.arange(0, sim_length_seconds, dT)
    sim_length = t_np.shape[0]

    # u_nominal = np.ones((1, sim_length)) # uncomment for constant input, first & second runs.
    u_nominal = -0.3 * np.cos(t_np * 3.1415926535 / 30).reshape((1, sim_length)) + 0.2
    
    u0 = np.array([[0.0]]) # Initial input (placeholder)
    u_max = 1 # Maximum Control Acutation

    # Define the system. Double integrator.
    A = np.array([[1., dT],
                  [0, 1.]])
    B = np.array([[0.],
                  [dT]])
    C = np.array([[1, 0.]]) # output is position.
    D = np.array([[0.]])

    sys = LinearSystem(A, B, C, D) # Create the linear system.

    # Build the STL Formula. "Positive Normal Form" (can only negate predicates.)
    lb_A = np.array([1])
    ub_A = np.array([-1])
    lb = LinearPredicate(lb_A, 0.7)# lower bound, A.T x - b >= 0
    ub = LinearPredicate(ub_A, -1.3) # upper bound, A.T x - b >= 0
    x_in_bounds = lb & ub

    # Recall: implication a=>b is ~a | b.
    # Add in a since.
    # exceed_once = ob.since(ob_neg, round(1/dT) - 1, np.inf)
    pi = x_in_bounds | x_in_bounds.always(0, round(2/dT)-1).eventually(0, round(2 / dT))
    print(pi)

    horizon = 2 * round(2 / dT)
    fts = 2*horizon # future time steps to project the system out.
    N = horizon
    # (N=horizon, b=1, applying Proposition 1 in the Baird, Coogan ACC 2023 paper).

    # Create data structures to save past states and inputs.
    x_hist = np.zeros((sys.n, sim_length))
    x_hist[:, 0:1] = x0
    u_hist = np.zeros((1, sim_length))

    # Create data structure to save proposed safe backup trajectories at each time step.
    x_persistently_safe = np.zeros((sys.n, sim_length, fts+1))

    print('System dimensions.')
    print(f'p: {sys.p}')
    print(f'n: {sys.n}')
    print(f'm: {sys.m}')

    # Create two polytopes: one representing bounds on x, the other representing bounds on u.
    H_x = np.array([[1,0], [-1, 0], [0, 1], [0, -1]])
    h_x = np.array([[1.3], [-0.7], [1], [1]]) # velocity bounds are an initial overestimate.
    H_u = np.array([[1], [-1]])
    h_u = np.array([[u_max], [u_max]])
    Hx = Polytope(H_x, h_x)
    Hu = Polytope(H_u, h_u)
    controllable_set_start = time.time()
    Qp = MatrixMath.controllable_set(A, B, dT, Hx, Hu)
    print("--- Controllable set computation took %s seconds ---" % (time.time() - controllable_set_start))
    Qp.plot_polytope(
        m_title=f'Maximal control invariant set, $\Delta t={dT}$', m_xlabel='$x_1$', m_ylabel='$x_2$', save=True)

    basic_line = np.ones((sim_length)) # for plotting "y in bounds"

    solver = GurobiMICPSolver(spec=pi, sys=sys, x0=x_hist[:, 0:1], T=horizon+fts,
             robustness_cost=False, hard_constraint=True, horizon=horizon, verbose=False, M=100, infinite_spec=None,
             N=N)
    solver.AddControlBounds(-u_max, u_max)
    solver.AddRecursiveFeasibilityConstraint(Qp)
    solver.AddLPCost()

    start_time = time.time() # to measure execution time.
    for i in range(1, sim_length):
        # Only update what we must at each time step.
        solver.updateModel(sys=sys, x0=x_hist[:, 0:i], i=i, u_hat=u_nominal[0, i-1:i])

        solution = solver.Solve()
        # print(f'i = {i}')
        # print(solution)

        x_1 = solution[0]
        x_persistently_safe[:, i-1, :] = x_1[:, horizon:]
        u_1 = solution[1]
        obj_1 = solution[4]

        # Now, propagate the state to be exactly known.
        x_hist[:, i:i+1] = A @ x_hist[:, i-1:i] + B @ u_1[:, horizon-1:horizon]

        # x_hist[:, i:i+1] = np.vstack((x_1[:2, horizon:horizon+1], x_1[:2, horizon:horizon+1]))
        # x_hist[:, i:i+1] = x_1[:, horizon:horizon+1]#x_1[:, indexer:indexer+1] # i:i+1
        u_hist[0, i-1] = u_1[0, horizon-1]#u_1[0, i-1]

    print("--- Execution time: %s seconds ---" % (time.time() - start_time))
    return # We only run this code to compare execution times.
main()

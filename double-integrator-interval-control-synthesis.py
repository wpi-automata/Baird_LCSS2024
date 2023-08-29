# File: double-integrator-interval-control-synthesis.py
# Purpose: Code for generating Figures 2 and 3 in the LCSS submission by Baird et al.
# 
# Author: Luke Baird
# (c) Georgia Institute of Technology, 2023

import numpy as np
from numpy import clip, empty, inf
import time
from stlpy.systems import LinearSystem
from stlpy.STL import LinearPredicate
from stlpy.solvers import GurobiMICPSolver
import interval
from matplotlib import pyplot as plt # Python plotting
import matplotlib.animation as animation
from matplotlib import rc
from polytopes import Polytope, MatrixMath


def d_positive (B, separate=True) :
    Bp = clip(B, 0, inf); Bn = clip(B, -inf, 0)
    if separate :
        return Bp, Bn
    else :
        n,m = B.shape
        ret = empty((2*n,2*m))
        ret[:n,:m] = Bp; ret[n:,m:] = Bp
        ret[:n,m:] = Bn; ret[n:,:m] = Bn
        return ret

def main():
    rc('text', usetex=True)

    # dT = 1 # for first run
    dT = 0.25 # uncomment for other plots

    # x0 = np.array([[1.1], [0.]]) # uncomment for constant input first run in the paper
    # x0 = np.array([[0.0], [0.]]) # uncomment for constant input second run
    x0 = np.array([[1.0], [0.], [1.0], [0]]) # uncomment for sinusoidal input

    sim_length_seconds = 30 # in seconds
    t_np = np.arange(0, sim_length_seconds, dT)
    sim_length = t_np.shape[0]

    u_nominal = np.ones((1, sim_length)) # uncomment for constant input, first & second runs.
    # u_nominal = -0.3 * np.cos(t_np * 3.1415926535 / 30).reshape((1, sim_length)) + 0.2
    
    print('nominal input vector')
    print(u_nominal)
    
    u_max = 1 # Maximum Control Acutation

    # Define the system. Double integrator.
    A = np.array([[1., dT],
                  [0, 1.]])
    B = np.array([[0.],
                  [dT]])
    C = np.array([[1, 0.]]) # output is position.
    D = np.array([[0.]])

    # [_x, x_]^T

    Ae = d_positive(A, False)
    Be = np.vstack((B,B))
    Ce = np.array([[1,0,0,0],
                   [0,0,1,0]])
    De = np.array([[0, 0]]).T

    # sys = LinearSystem(A, B, C, D) # Create the linear system.
    sys = LinearSystem(Ae, Be, Ce, De)

    # Build the STL Formula. "Positive Normal Form" (can only negate predicates.)
    lb_A = np.array([1 + np.interval(-0.05, 0.05)])
    ub_A = np.array([-1 + np.interval(-0.05, 0.05)])
    lb = LinearPredicate(lb_A, 0.7 + np.interval(-0.02, 0.02))# lower bound, A.T x - b >= 0
    ub = LinearPredicate(ub_A, -1.3 + np.interval(-0.02, 0.02)) # upper bound, A.T x - b >= 0
    x_in_bounds = lb & ub

    pi = x_in_bounds | x_in_bounds.always(0, round(2/dT)-1).eventually(0, round(2 / dT))
    print(pi)

    horizon = 2 * round(2 / dT)
    fts = 2*horizon # future time steps to project the system out.
    N = horizon
    # (N=horizon, b=1, applying Proposition 1 in the Baird, Coogan ACC 2023 paper).

    # Create data structures to save past states and inputs.
    x_hist = np.zeros((4, sim_length))
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

    Qp.P = np.block([[Qp.P, np.zeros(Qp.P.shape)],
                     [np.zeros(Qp.P.shape), Qp.P]])
    Qp.p = np.vstack((Qp.p, Qp.p))

    w = np.array([[-0.001],
                  [-0.001],
                  [0.001],
                  [0.001]])

    solver = GurobiMICPSolver(spec=pi, sys=sys, x0=x_hist[:, 0:1], T=horizon+fts,
             robustness_cost=False, hard_constraint=True, horizon=horizon, verbose=False, M=100, infinite_spec=None,
             N=N, w=w, intervals=True)
    solver.AddControlBounds(-u_max, u_max)
    solver.AddRecursiveFeasibilityConstraint(Qp)
    solver.AddLPCost()

    start_time = time.time() # to measure execution time.
    for i in range(1, sim_length):
        # Only update what we must at each time step.
        solver.updateModel(sys=sys, x0=x_hist[:, 0:i], i=i, u_hat=u_nominal[0, i-1:i])

        solution = solver.Solve()

        x_1 = solution[0]
        x_persistently_safe[:, i-1, :] = x_1[:, horizon:]
        u_1 = solution[1]
        obj_1 = solution[4]

        # Now, propagate the state to be exactly known.
        x_hist[:, i:i+1] = Ae @ x_hist[:, i-1:i] + Be @ u_1[:, horizon-1:horizon]
        u_hist[0, i-1] = u_1[0, horizon-1]

    print("--- Execution time: %s seconds ---" % (time.time() - start_time))

    # Compute the robustness interval afterwards for plotting..
    rho = []
    for j in range(sim_length - horizon):
        rho.append(pi.robustness(x_hist[2:3, :], j))

    rho = np.array(rho)

    _ub, ub_ = interval.get_lu((ub.b / ub.a)[0])
    _lb, lb_ = interval.get_lu((lb.b / lb.a)[0])

    # uncomment to print the results as individual plots.
    # position (x_1)
    x_figure, x_axis = plt.subplots()
    x_axis.plot(t_np, x_hist[0, 0:sim_length], 'b.-')

    x_axis.plot(t_np, _ub * basic_line, 'r')
    x_axis.plot(t_np, ub_ * basic_line, 'r')
    x_axis.fill_between(t_np, _ub * basic_line, ub_ * basic_line, facecolor='r', alpha=0.25)

    x_axis.plot(t_np, _lb * basic_line, 'r')
    x_axis.plot(t_np, lb_ * basic_line, 'r')
    x_axis.fill_between(t_np, _lb * basic_line, lb_ * basic_line, facecolor='r', alpha=0.25)

    x_axis.plot(t_np, basic_line)

    x_axis.grid(True)
    x_axis.set_title('Output $y[t]$')
    x_axis.set_xlabel('$t$ (s)')
    x_axis.set_ylabel('$y[t]$')
    x_figure.set_figheight(4)
    x_figure.savefig('output/x.pdf', format='pdf')

    # velocity (x_2)
    v_figure, v_axis = plt.subplots()
    v_axis.plot(t_np, x_hist[1, 0:sim_length], 'b.-')
    v_axis.grid(True)
    v_axis.set_title('$x_2[t]$')
    v_axis.set_xlabel('t (s)')
    v_axis.set_ylabel('$x_2[t]$')
    v_figure.set_figheight(4)
    v_figure.savefig('output/v.pdf', format='pdf')

    # input
    u_figure, u_axis = plt.subplots()
    i_u_1, = u_axis.plot(t_np, u_hist[0, 0:sim_length], 'b.-', alpha=0.6)
    i_u_2, = u_axis.plot(t_np, u_nominal[0, 0:sim_length], 'r-', alpha=0.6)
    u_axis.grid(True)
    u_axis.set_title('Input $u[t]$')
    u_axis.set_xlabel('$t$ (s)')
    u_axis.set_ylabel('$u[t]$')
    u_axis.legend([i_u_1, i_u_2], ['Filtered', 'Nominal'], loc="lower right")
    u_figure.set_figheight(4)
    u_figure.savefig('output/u.pdf', format='pdf')

    # Robustness
    _rho, rho_ = interval.get_lu(rho)
    rho_figure, rho_axis = plt.subplots()
    rho_axis.plot(t_np[:_rho.shape[0]], _rho, 'm.-')
    rho_axis.plot(t_np[:rho_.shape[0]], rho_, 'm.-')
    rho_axis.fill_between(t_np[:rho_.shape[0]], _rho, rho_, facecolor='m', alpha=0.25)
    rho_axis.grid(True)
    rho_axis.set_title('Robustness  $[\\rho](y, t)$')
    rho_axis.set_xlabel('$t$ (s)')
    rho_axis.set_ylabel('$\\rho$')
    rho_figure.set_figheight(5)
    rho_figure.savefig('output/rho.pdf', format='pdf')
    plt.show()

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(autoscale_on=False, xlim=(0, sim_length_seconds))
    if dT==1.0:
        ax.set_ylim(-0.1, 2.1)
    elif dT==0.5:
        ax.set_ylim(0.2, 1.8)
    elif dT > 0.25:
        ax.set_ylim(0.3, 1.7)
    else:
        ax.set_ylim(0.5, 1.8)
    ax.set_xlabel('$t (s)$')
    ax.set_ylabel('$y[t]$')
    ax.grid()

    historical, = ax.plot([], [], 'b.-', lw=2)

    _ub_ax, = ax.plot(t_np, _ub * basic_line, 'r')
    ub__ax, = ax.plot(t_np, ub_ * basic_line, 'r')
    ub_fill = ax.fill_between(t_np, _ub * basic_line, ub_ * basic_line, facecolor='r', alpha=0.25)

    _lb_ax, = ax.plot(t_np, _lb * basic_line, 'r')
    lb__ax, = ax.plot(t_np, lb_ * basic_line, 'r')
    lb_fill = ax.fill_between(t_np, _lb * basic_line, lb_ * basic_line, facecolor='r', alpha=0.25)

    lower_corner, = ax.plot([], [], 'g.-', linestyle='dotted')
    upper_corner, = ax.plot([], [], 'm.-', linestyle='dotted')

    ax.legend([historical], ['Historical Trajectory'], loc="lower right")
    bob=ax.fill_between([0,1], [1,1], [1,1], alpha=0.25) # placeholder.
    print(bob.get_paths()[0].vertices)
    def animate(i):
        i = i+1
        # print(f'calling animate {i}')
        this_past_t = t_np[:i]
        this_past_x = x_hist[0, :i]
        this_predicted_t = t_np[i-1:min(i+N, t_np.shape[0])] # add one step of the past for visualization.
        n = this_predicted_t.shape[0]
        this_predicted_x = x_persistently_safe[:, i, :] #fts + 1 steps predicted in the future
        this_predicted_x = np.hstack((x_hist[:, i-1:i], this_predicted_x))
        this_predicted_x = this_predicted_x[:, :n]

        xl = this_predicted_x[0, :]
        xu = this_predicted_x[2, :]
        
        vertices = np.zeros((n * 2 + 3, 2))
        vertices[1:n+1, 0] = this_predicted_t
        vertices[n+2:-1:, 0] = this_predicted_t[::-1]
        vertices[1:n+1, 1] = xl
        vertices[n+2:-1:, 1] = xu[::-1]

        vertices[0, 0] = this_predicted_t[0]
        vertices[-1, 0] = this_predicted_t[-1]
        vertices[n+1, 0] = this_predicted_t[-1]

        vertices[0, 1] = xl[0]
        vertices[n+1, 1] = xl[-1]
        vertices[-1, 1] = xu[-1]
        
        bob.set_paths([vertices])

        lower_corner.set_data(this_predicted_t, xl)
        upper_corner.set_data(this_predicted_t, xu)

        historical.set_data(this_past_t, this_past_x)
        return historical, lower_corner, upper_corner, bob, _ub_ax, ub__ax, _lb_ax, lb__ax, ub_fill, lb_fill

    ani = animation.FuncAnimation(
        fig, animate, sim_length - (N), interval=int(1000 * dT), blit=True, repeat=False
    )

    writer = animation.PillowWriter(fps=int(1/dT))#animation.writers['ffmpeg'](fps=30)
    ani.save('output/y.gif', writer=writer)

    plt.show()

    main_figure, (x_axis, u_axis) = plt.subplots(2, 1)

    # Create the x-axis plots.
    i_x_1, = x_axis.plot(t_np, x_hist[0, 0:sim_length], 'b.-')

    # With the interval plots.
    i_x_2, = x_axis.plot(t_np, _ub * basic_line, 'r')
    x_axis.plot(t_np, ub_ * basic_line, 'r')
    x_axis.fill_between(t_np, _ub * basic_line, ub_ * basic_line, facecolor='r', alpha=0.25)

    x_axis.plot(t_np, _lb * basic_line, 'r')
    x_axis.plot(t_np, lb_ * basic_line, 'r')
    x_axis.fill_between(t_np, _lb * basic_line, lb_ * basic_line, facecolor='r', alpha=0.25)

    x_axis.grid(True)
    x_axis.set_title('Output $y[t]$')
    x_axis.set_xlabel('$t$ (s)')
    x_axis.set_ylabel('$y[t]$')
    x_axis.legend([i_x_1, i_x_2], ['A safe trajectory', '$y$ in bounds'], loc="lower right")

    # And then add the input plots.
    i_u_1, = u_axis.plot(t_np, u_hist[0, 0:sim_length], 'b.-', alpha=0.7)
    i_u_2, = u_axis.plot(t_np, u_nominal[0, 0:sim_length], 'k-', alpha=0.8)
    u_axis.grid(True)
    u_axis.set_title('Input $u[t]$')
    u_axis.set_xlabel('$t$ (s)')
    u_axis.set_ylabel('$u[t]$')
    u_axis.legend([i_u_1, i_u_2], ['Filtered', 'Nominal'], loc="lower right")

    main_figure.subplots_adjust(hspace=0.5)
    main_figure.set_figheight(5) # Adjust this parameter to make plots fit in the paper.

    # x_axis.set_yticks([0.7,0.9,1.1,1.3,1.5])

    main_figure.savefig('output/u_and_x.pdf', format='pdf')

    plt.show()
main()
# import cProfile
# import pstats
# from pstats import SortKey
# p = pstats.Stats('restats')
# p = cProfile.run('main()')
# p.sort_stats(SortKey.CUMULATIVE).print_stats()

# File: blimp-trace-monitoring.py
# Purpose: Code for generating Figure 1 in the paper
# 
# Author: Luke Baird
# (c) Georgia Institute of Technology, 2023
import numpy as np
import sympy as sp
import time
import interval
from inclusion import NaturalInclusionFunction
import matplotlib.animation as animation
from scipy.io import loadmat
from stlpy.STL import LinearPredicate, NonlinearPredicate
from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)

# Load all relevant data from mat.
mat = loadmat('blimp-rta-output-pd-video.mat')
x = mat['x'].T # nt columns, 12 rows.
t = mat['t'].squeeze(0)

# Cutoff some of the first values
t = t[80:] # Probably good enough here.

# And the last values...
t = t[:2000] # Sanity!

t -= t[0] # Start at time zero.
dT = np.round((t[100] - t[0]) / 100, 2)
print(x.shape)
print(t.shape)
nt = t.shape[0]

# Create an STL formula - start first with a known STL formula.
square_dim = 1.41
a_x = np.zeros((1,12)); a_x[0,6] = 1
a_y = np.zeros((1,12)); a_y[0,7] = 1
left = LinearPredicate(a_x, square_dim) # x - 1.23 >= 0, x >= 1.23
right = LinearPredicate(-a_x, square_dim) # -x - 1.23 >= 0, x <= -1.23
top = LinearPredicate(a_y, square_dim)
bottom = LinearPredicate(-a_y, square_dim)

square_spec = left | right | top | bottom
pi = square_spec | square_spec.always(0, int(2/dT)-1).eventually(0, int(3 / dT))

# Nonlinear Predicate Handling
v_x, v_y, v_z, _, _, _, _, _, _, _, _, _ = (v_vars := sp.symbols('v_x v_y v_z w_x w_y w_z x y z theta phi psi'))
g = sp.Matrix([2 - sp.sqrt(v_x ** 2 + v_y ** 2 + v_z ** 2)])
g_nif = NaturalInclusionFunction(g, v_vars)
psi = NonlinearPredicate(g_nif, 12) # lambda: x 

pi = pi & psi

# Interval analysis time.
# Create an uncertainty for each of the values.
eps = np.array([0.075, 0.075, 0.075, 0.0065, 0.0065, 0.0065, 0.002, 0.002, 0.002, 0.02, 0.02, 0.02]) * 10
epsilon = interval.get_iarray(-1*eps, eps)

x_no_uncertainty = x.copy()
x = x + epsilon # Add uncertainty. For a "no uncertainty" version, comment this line.

# Compute the robustness vector.
rho = []
robustness_start_time = time.time() # For reporting the computation time.
for j in range(nt - int(5/dT)):
    rho.append(pi.robustness((x[j:j+int(5/dT) + 4,:]).T, 0))
print("--- Robustness computation took %s seconds ---" % (time.time() - robustness_start_time))
nt = len(rho)
rho = np.array(rho)

# For plotting, convert rho back to two values
_rho, rho_ = interval.get_lu(rho)

lineTop = np.ones((nt,)) * square_dim
lineBottom = np.ones((nt,)) * square_dim * -1

rho_fig, rho_axes = plt.subplots()
rho_fig.subplots_adjust(bottom=0.21, top=0.99, left=0.08, right=0.99)
t_range = np.array([t * dT for t in range(nt)])
rho_axes.plot(t_range, _rho, 'b')
rho_axes.plot(t_range, rho_, 'b')
rho_axes.fill_between(t_range, _rho, rho_, facecolor='b', alpha=.25)
rho_axes.set_ylabel('$[\\rho]$')
rho_axes.set_xlabel('$t$ (s)')
rho_axes.grid(True)
# x_axes.grid(True)
# x_axes.set_title('$x$')
rho_fig.set_figheight(2.0)

#Consider adding in a mkdir here
rho_fig.savefig('output/blimp_trace_monitoring.pdf', format='pdf')
plt.show()


flying_figure = plt.figure(figsize=(2.5,2.5))
flying_ax = flying_figure.add_subplot(autoscale_on=False)
flying_ax.set_xlim(-2.5, 2.5)
flying_ax.set_ylim(-2.5, 2.5)
flying_ax.set_xlabel('$x$')
flying_ax.set_ylabel('$y$')
# flying_ax.set_title('2D plot of trajectory')
square_x = np.array([-square_dim, -square_dim, square_dim, square_dim, -square_dim])
square_y = np.array([-square_dim, square_dim, square_dim, -square_dim, -square_dim])
flying_ax.plot(square_x, square_y, 'r')
flying_ax.grid()
flying_trajectory, = flying_ax.plot(x_no_uncertainty[1:nt-1,6], x_no_uncertainty[1:nt-1,7], 'k.-')
# flying_ax.legend(['Square', 'Historical', 'Safe Backup'])
plt.subplots_adjust(left=0.18, bottom=0.18)
flying_figure.savefig('output/flying.pdf')
plt.show()

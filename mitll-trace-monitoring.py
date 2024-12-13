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
import pickle 

rc('text', usetex=True)

# Part 1: Un-pickling and loading MITLL data 
with open('sim_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Get all the GCS values for agent 1
t = []
lat_1 = []
lon_1 = []
alt_1 = []

# lat_2 = []
# lon_2 = []
# alt_2 = []

# lat_3 = []
# lon_3 = []
# alt_3 = []

# lat_4 = []
# lon_4 = []
# alt_4 = []
x = np.array([:,2])

for timestamp in data:
    t.append(timestamp[0]) 
    # lat_1.append(timestamp[1][1]["lat"])
    # lon_1.append(timestamp[1][1]["lon"])
    x.append(timestamp[1][1]["lat"], timestamp[1][1]["lon"])
    # alt_1.append(timestamp[1][1]["alt"])

    # Left these in since it would be fun to build on them later 

    # lat_2.append(timestamp[1][2]["lat"])
    # lon_2.append(timestamp[1][2]["lon"])
    # alt_2.append(timestamp[1][2]["alt"])

    # lat_3.append(timestamp[1][3]["lat"])
    # lon_3.append(timestamp[1][3]["lon"])
    # alt_3.append(timestamp[1][3]["alt"])

    # lat_4.append(timestamp[1][4]["lat"])
    # lon_4.append(timestamp[1][4]["lon"])
    # alt_4.append(timestamp[1][4]["alt"])


#x = np.array([[lat_1, lon_1]])# Our signal- just lat, lon position 
t = np.array(t)
print(x.shape)
print(t.shape)
nt = t.shape[0]

# Create the STL specification 
# #1: Position only 
# For timestamp t1 to t2, trajectory should always be within bounds lat, -lat, lon, -lon

offset = 0.0001
lat_center = 42.49464
lon_center = -71.6734

a = [1, 1] #Signal multiplier- will be useful once other variables are included in spec

left = LinearPredicate(a, lat_center - offset) 
right = LinearPredicate(-a, -lat_center - offset) 
top = LinearPredicate(-a, -lon_center - offset)
bottom = LinearPredicate(a, lon_center - offset)

square_spec = left & right & top & bottom
pi = square_spec.always(51.3, 58.2) # times determined manually - I might change this in the future 


# Interval analysis time.
# Create an uncertainty for each of the values.
eps = np.array([0.000005, 0.000005]) #* 10 
epsilon = interval.get_iarray(-1*eps, eps)

x_no_uncertainty = x.copy()
x = x + epsilon # Add uncertainty. For a "no uncertainty" version, comment this line.

# Compute the robustness vector.
rho = []
robustness_start_time = time.time() # For reporting the computation time.
# for j in range(nt - int(5/dT)):
#     rho.append(pi.robustness((x[j:j+int(5/dT) + 4,:]).T, 0))
# print("--- Robustness computation took %s seconds ---" % (time.time() - robustness_start_time))
# nt = len(rho)
# rho = np.array(rho)

for j in range(len(time)):
    rho.append(pi.robustness((x).T, 0)) #CHANGE? 
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


# 2D Trajectory Plot
plt.figure()
plt.title("Agent 1 2D GCS Trajectory")
plt.xlabel("Latitude")
plt.ylabel("lonitude")
plt.axis('equal')

square = plt.Rectangle((42.49464 - offset, -71.6734 - offset), 2*offset, 2*offset, facecolor='none', ec="red")
plt.gca().add_patch(square)
plt.scatter(lat_1, lon_1)
flying_figure.savefig('output/flying.pdf')
plt.show()

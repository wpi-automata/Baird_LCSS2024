# This script will: 
    # Unpkl the LL data and convert it to a usable data struct 
    # Plot the 2D and 3D trajectories 
    # Do the same for the blimp data- see how it lines up with plots created by other scripts 

import pickle 
from scipy.io import loadmat
# import pandas as pd
import numpy
import numpy.core.multiarray 
import matplotlib.pyplot as plt
import sys

print(sys.path)

# Part 1: Un-pickling and loading MITLL data 
with open('sim_data.pkl', 'rb') as f:
    data = pickle.load(f)

# There's probably a better way to do this with a df
# df = pd.read_pickle('sim_data.pkl')

# Get all the GCS values for agent 1
lat_1 = []
lon_1 = []
alt_1 = []

lat_2 = []
lon_2 = []
alt_2 = []

lat_3 = []
lon_3 = []
alt_3 = []


lat_4 = []
lon_4 = []
alt_4 = []

time = []

t1 = 0
t2 = 0
offset = 0.0001

for timestamp in data:
    lat_1.append(timestamp[1][1]["lat"])
    lon_1.append(timestamp[1][1]["lon"])
    alt_1.append(timestamp[1][1]["alt"])

    # Getting the time 
    if(timestamp[1][1]["lat"] >= 42.49464 - offset and timestamp[1][1]["lat"] <= 42.49464 + offset): 
        time.append(timestamp[0])

    lat_2.append(timestamp[1][2]["lat"])
    lon_2.append(timestamp[1][2]["lon"])
    alt_2.append(timestamp[1][2]["alt"])

    lat_3.append(timestamp[1][3]["lat"])
    lon_3.append(timestamp[1][3]["lon"])
    alt_3.append(timestamp[1][3]["alt"])

    lat_4.append(timestamp[1][4]["lat"])
    lon_4.append(timestamp[1][4]["lon"])
    alt_4.append(timestamp[1][4]["alt"])
   
# Part 2: Loading the blimp data 
mat = loadmat('blimp-rta-output-pd-video.mat')
x = mat['x'] # nt columns, 12 rows.
p = x[6:9]
p_x = x[0]
p_y = x[1]
p_z = x[2]


# Part 3: Plotting the data 

print(f"Start time: {time[0]}")
print(f"End time: {time[len(time) - 1]}")

# 3D (1 agent)
fig = plt.figure()
plt.title("Agent 1 3D GCS Trajectory")
ax = fig.add_subplot(projection='3d')
ax.scatter(lat_1, lon_1, alt_1)
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_zlabel("Altitude")

# 2D
plt.figure()
plt.title("Agent 1 2D GCS Trajectory")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.axis('equal')

square = plt.Rectangle((42.49464 - offset, -71.6734 - offset), 2*offset, 2*offset, facecolor='none', ec="red")
plt.gca().add_patch(square)
plt.scatter(lat_1, lon_1)


# 3D (all agents)
fig = plt.figure()
plt.title("All Agents 3D GCS Trajectory")
ax = fig.add_subplot(projection='3d')
ax.scatter(lat_1, lon_1, alt_1)
ax.scatter(lat_2, lon_2, alt_2)
ax.scatter(lat_3, lon_3, alt_3)
ax.scatter(lat_4, lon_4, alt_4)
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_zlabel("Altitude")

# 2D
plt.figure()
plt.title("Agent 1 2D GCS Trajectory")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.scatter(lat_1, lon_1)
plt.scatter(lat_2, lon_2)
plt.scatter(lat_3, lon_3)
plt.scatter(lat_4, lon_4)

# 2D Blimp - This is likely not right 
plt.figure()
plt.title("Blimp 2D Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(p_x, p_y)

plt.show()
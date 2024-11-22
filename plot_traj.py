# This script will: 
    # Unpkl the LL data and convert it to a usable data struct 
    # Plot the 2D and 3D trajectories 
    # Do the same for the blimp data- see how it lines up with plots created by other scripts 

import pickle 
# from scipy.io import loadmat
import pandas as pd
import numpy
import matplotlib as plt
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
alt = []

for timestamp in data: 
    lat.append(timestamp[1][1]["lat"])
    lat.append(timestamp[1][1]["lon"])
    lat.append(timestamp[1][1]["alt"])
   
# Part 2: Loading the blimp data 
# mat = loadmat('blimp-rta-output-pd-video.mat')
# x = mat['x'] # nt columns, 12 rows.

# Part 3: Plotting the data 
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(lat, lon, alt)
# plt.show()
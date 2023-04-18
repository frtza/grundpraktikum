# essential libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# read columns of data from txt file
p_6, qty_6, max_6, chnl_6 = np.genfromtxt('data/stats_6cm.txt', unpack=True)
p_4, qty_4, max_4, chnl_4 = np.genfromtxt('data/stats_4cm.txt', unpack=True)
stat = np.genfromtxt('data/qty_4cm_300mbar.txt', unpack=True)

# parameter and error arrays, covariance matrix for regression

# graphical representation of correlation

# format and export latex table

# format and export calculated values to build directory

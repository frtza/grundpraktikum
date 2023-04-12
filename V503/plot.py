# essential libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds
from uncertainties import ufloat

# read columns of data from txt file
v, U = np.genfromtxt('data/filterkurve.txt', unpack=True)

# implementation of gaussian filter for smoothing unevenly spaced data
def gaussian_sum_smooth(x_data, y_data, x_eval, sig):
	delta_x = x_eval[:, None] - x_data
	weights = np.exp((-delta_x**2)/(2*sig**2)) / (np.sqrt(2*np.pi)*sig)
	weights /= np.sum(weights, axis=1, keepdims=True)
	return np.dot(weights, y_data)

# physical constants
eps_0 = const.epsilon_0
e_0 = const.e
m_0 = const.m_e
k_B = const.k
h_0 = const.h


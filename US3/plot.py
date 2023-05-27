import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import sqrt
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties import unumpy as unp 
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds) 

messung1=pd.read_csv('data/messung1.txt',sep=' ', header=None, names=['lei', 'f15', 'f30', 'f60'])
print(messung1.to_latex(index=False, column_format="c c c c"))

speed1=pd.read_csv('data/speed1.txt',sep=' ', header=None, names=['d','s','sig'])
print(speed1.to_latex(index=False, column_format="c c c"))

lei, f15, f30, f60 = np.genfromtxt('data/messung1.txt', unpack=True, skip_header=1)  
d, s, sig = np.genfromtxt('data/speed1.txt', unpack=True, skip_header=1)
d2, s2, sig2 = np.genfromtxt('data/speed2.txt', unpack=True, skip_header=1)

#Prismawinkel
alpha = [80.06, 70.53, 54.74]
alpha = np.multiply(alpha, (np.pi/180))
c = const.speed_of_light
nu0 = 2e6 

def v(nu, a):
    return (nu * c)/(2 * nu0 * np.cos(a)) /1e5

v15 = np.zeros(5)
v30 = np.zeros(5)
v60 = np.zeros(5)

# for j in range(5):
#     v15[j] = v(dnu15[j], alpha[0])
#     v30[j] = v(dnu30[j], alpha[1])
#     v60[j] = v(dnu60[j], alpha[2])

# Plot 1:

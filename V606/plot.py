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
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray

# Plot 1:

v, U = np.genfromtxt('data/filterkurve.txt', unpack=True, skip_header=1)      # Normierung 
n = len(v)                              #the number of data
mean = sum(v*U)/n                       #note this correction
# sigma = sum(U*(nu - mean)**2)/n        #note this correction
sigma = np.sqrt(sum(U*(v - mean)**2))

# Ausgleichsrechung nach Gaußverteilung
def g(x,a,x0,b):
    return a*np.exp(-(x-x0)**2/(b)) # b = 2*sigma**2

para, pcov = curve_fit(g, v, U, p0=[1,mean,sigma])
a, nu0, b = para
pcov = np.sqrt(np.diag(pcov))
fa, fnu0, fb = pcov

# Fehler der Parameter
ua = ufloat(a, fa) 
ub = ufloat(b, fb)
unu0 = ufloat(nu0, fnu0)

xx = np.linspace(0, 31, 10**4)

plt.plot(v, U, 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$v \, / \, kHz$')
plt.ylabel(r'$U_A \, / \, U_E v$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)

plt.savefig('build/plot.pdf', bbox_inches = "tight")
#plt.show()
plt.clf() 

#Tabelle der Filterkurve
#print('Tabelle zur Filterkurve:')
#filter=pd.read_csv('data/filterkurve.txt',sep=' ', header=None, names=['Frequenz', 'Impulse'])
#print(filter.to_latex(index=False, column_format="c c"))
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

md = pd.read_csv('tables/md.csv')

nu, U = np.genfromtxt('tables/U.txt', unpack=True, skip_header=1)
U = U/(2*0.9892978920131985)            # Normierung 

# für den initial guess bei curvefit()
n = len(nu)                             # Anzahl der Daten
mean = sum(nu*U)/n                      # Mittelwert
sigma = np.sqrt(sum(U*(nu - mean)**2))  # Standardabweichung

# Ausgleichsrechung nach Gaußverteilung
def g(x,a,x0,b):
    return a*np.exp(-(x-x0)**2/(b))     # b = 2*sigma**2

para, pcov = curve_fit(g, nu, U, p0=[1,mean,sigma])
a, nu0, b = para
pcov = np.sqrt(np.diag(pcov))
fa, fnu0, fb = pcov
ua = ufloat(a, fa) 
ub = ufloat(b, fb)
unu0 = ufloat(nu0, fnu0)

xx = np.linspace(18, 42, 10**4)

plt.plot(nu, U, 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$\nu \, / \, \mathrm{kHz}$')
plt.ylabel(r'$U_A \, / \, U_E$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
plt.xlim(22, 40)
plt.ylim(-0.05, 1.05)

plt.savefig('build/plot.pdf', bbox_inches = "tight")
plt.clf() 

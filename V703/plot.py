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
                                      # Abweichung:       stds(fehlerarray) = errarray

#latexTabelle der Messdaten zur Kennlinie des Geiger-Müller-Zählrohrs
print('Tabelle zur Kennlinie:')
kennlinie=pd.read_csv('data/kennlinie.txt',sep=' ', header=None, names=['Spannung', 'Zählrate', 'Strom'])
print(kennlinie.to_latex(index=False, column_format="c c c"))

U, N, I = np.genfromtxt('data/kennlinie.txt', unpack=True, skip_header=1)  

#Fehler der Zählrate lamdda
N_fehler = np.round((1/np.sqrt(N))*N)
print(N_fehler)

#Kennlinie Geiger-Müller
#Up = U[5:25]
#Np = N[5:25]

### Vorschlag zur Wahl besserer Plateau-Daten:
# entweder
#Up = U[4:17]
#Np = N[4:17]
# oder
Up = U[3:19]
Np = N[3:19]
###

Np = Np/120

def g(a,b,x):
    return a * x + b

para, pcov = curve_fit(g,Up,Np)
a, b = para
pcov = sqrt(np.diag(pcov))
fa, fb = pcov

ua = ufloat(a, fa)
ub = ufloat(b, fb)
print('a = (%.3f ± %.3f)' % (noms(ua), stds(ua)))
print('b = (%.3f ± %.3f)' % (noms(ub), stds(ub)))

#xx = np.linspace(410, 650, 10000)   # Spannungen für das Plateau-Gebiet
xx = np.linspace(370, 710, 10000)   ### Vorschlag: Gerade etwas breiter als Plateau anzeigen lassen###
fN = sqrt(N)                        # N Poisson-verteilt
uN = uarray(N, fN)
uN = uN/120                         # Impulsrate mit Fehler
plt.errorbar(U, noms(uN), yerr = stds(uN), fmt='r.', elinewidth = 1, capsize = 2, label = 'Messdaten')
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Plateaugerade')

plt.xlabel(r'$U \, / \, \mathrm{V}$')
plt.ylabel(r'$N \, / \, \mathrm{s^{-1}}$')
plt.legend(loc="best")                  # legend position
plt.grid(True) 
plt.savefig('build/plot_1.pdf')


#anzahl Ladungsträger
kennlinie=pd.read_csv('data/kennlinie.txt',sep=' ', header=None, names=['Spannung', 'Zählrate', 'Strom'])
print(kennlinie.to_latex(index=False, column_format="c c c"))

#I_a = ufloat(I, 0.005)

e = const.elementary_charge
def l(i, t):
    e = const.elementary_charge
    return (i* 10**(-6)*t)/e
t = 120
fI = 0.05*1e-6       # Fehler des Stroms in μA
uI = uarray(I, fI)
N_e = l(uI,t)

nomN_e = l(I, t)
print(nomN_e)
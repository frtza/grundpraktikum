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



acryl=pd.read_csv('data/acryl.txt', sep=' ', header=None, names=['n','t'])
#print(acryl.to_latex(index=False, column_format="c c"))

#acryl a scan

n, t = np.genfromtxt('data/acryl.txt', unpack=True, skip_header=1) 
n2, ak, bk, d,f = np.genfromtxt('data/ausmessung.txt', unpack=True , skip_header=1)

acryl=pd.read_csv('data/acryl.txt', sep=' ', header=None, names=['n','t'])
#print(acryl.to_latex(index=False, column_format="c c"))

#acryl a scan

n, t = np.genfromtxt('data/acryl.txt', unpack=True, skip_header=1) 
n2, ak, bk, d,f = np.genfromtxt('data/ausmessung.txt', unpack=True , skip_header=1)
d = d * 10**(-3)
ak = ak * 10**(-3)
bk = bk * 10**(-3)
t = t * 10**(-6)

plt.plot(t,2*ak[1:8], 'xr', markersize=6 , label = 'Messdaten')

def g(x, a, b):
    return a * x + b

para, pcov = curve_fit(g, 2*ak[1:8], t)
a, b = para
fa, fb = np.sqrt(np.diag(pcov))
ua = ufloat(a, fa) 
ub = ufloat(b, fb)

print('Ausgleichsgerade1:')
print(ua)
print(ub)
xx = np.linspace(3,45 , 100)
plt.plot(xx* 10**(-6), g(xx* 10**(-6), a, b), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.show()
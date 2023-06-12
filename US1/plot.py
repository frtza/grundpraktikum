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

atheo = 2730

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

para, pcov = curve_fit(g,t, ak[1:8]*2)
a, b = para
fa, fb = np.sqrt(np.diag(pcov))
ua = ufloat(a, fa) 
ub = ufloat(b, fb)

print('Ausgleichsgerade1:')
print(ua)
print(ub)
abw = (a - atheo) / atheo
print('Abweichung')
print(abw)
xx = np.linspace(3,50 , 10**4)
plt.plot(xx* 10**(-6), g(xx* 10**(-6), a, b), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$t \, / \, ( s \cdot 10^6)$')
plt.ylabel(r'$ 2 d \, / \, mm$')
plt.grid(ls=':')
plt.legend(loc="best")
plt.savefig('build/plot1.pdf',bbox_inches = "tight")
plt.clf()
#plt.show()
#r'$t / \mathrm{s}$
ascan=pd.read_csv('data/ascan.txt', sep=' ', header=None, names=['n','a','b'])
#print(ascan.to_latex(index=False, column_format="c c c"))

na, a, b = np.genfromtxt('data/ascan.txt', unpack=True, skip_header=1)

ak = np.array([13.5, 21.85, 30.3, 38.7, 46.8, 54.7, 62.7, 70.6, 15.2])*10**(-3)
a = a*10**(-3)
da = a - ak
print(np.mean(da))

bk = np.array([61.85,54.4,47.0, 39.5, 31.0,23.0, 15.35, 7.2, 55.8])*10**(-3)
b = a*10**(-3)
db = bk - b
print(np.mean(db))

bscan=pd.read_csv('data/bscan.txt', sep=' ', header=None, names=['n','a','b'])
#print(bscan.to_latex(index=False, column_format="c c c"))
na, a, b = np.genfromtxt('data/bscan.txt', unpack=True, skip_header=1)
print('bscan differenzen:')
a = a*10**(-3)
ba = a - ak
b = a*10**(-3)
bb = bk - b
print(np.mean(bb))
print(np.mean(ba))


### Daten zu Aufloesungsgvermoegen einlesen:
def conv(x):
	return x.replace(',', '.').encode()
asc1 = np.genfromtxt((conv(x) for x in open('data/US1_daten/A-Scan-1MHz.txt')))
asc2 = np.genfromtxt((conv(x) for x in open('data/US1_daten/A-Scan-2MHz.txt')))

# Tiefe [mm]
d1 = asc1[:, 0]
d2 = asc2[:, 0]

# Spannung [V]
U1 = asc1[:, 1]
U2 = asc2[:, 1]

plt.plot(d1, U1, markersize=6, label='Messdaten')
#plt.xlabel(r'$  d \, / \, mm$')
plt.xlabel(r'$t \, / \, ( s \cdot 10^6)$')
plt.ylabel(r'$U / \mathrm{V}$')
plt.savefig('build/plot2.pdf', bbox_inches="tight")
plt.clf()
#plt.show()

plt.plot(d2, U2, markersize=6, label='Messdaten')
#plt.xlabel(r'$  d \, / \, mm$')
plt.xlabel(r'$t \, / \, ( s \cdot 10^6)$')
plt.ylabel(r'$U / \mathrm{V}$')
plt.savefig('build/plot3.pdf', bbox_inches="tight")
plt.clf()
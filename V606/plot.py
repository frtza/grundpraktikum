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
sigma = sum(U*(v - mean)**2)/n        #note this correction
#sigma = np.sqrt(sum(U*(v - mean)**2))

# Ausgleichsrechung nach Gaußverteilung
def g(x,a,x0,b):
    return a*np.exp(-(x-x0)**2/(b)) # b = 2*sigma**2
#return beta/(np.sqrt(2 * np.pi *sigma**(2))) * np.exp(-(x-alpha)**2 / (2* sigma**2))

para, pcov = curve_fit(g, v, U, p0=[1,mean,sigma])
a, nu0, b = para
pcov = np.sqrt(np.diag(pcov))
fa, fnu0, fb = pcov

# Test-Fit alternative Glockenkurve
def test(x, a, b, c, d, e, f):
#	return a / ((x - b)**2 + c)
#	return c / ((x**2 - a**2)**2 + b**2 * a**2)
	return (a * np.exp(b * np.abs((x - c))**d) + np.sqrt(np.exp((x - c)**2 * e))) * f
par_test, cov_test = curve_fit(test, v, U, p0=[4,-1,20,1,-0.1,0.99])
err_test = np.sqrt(np.diag(cov_test))

# Fehler der Parameter
ua = ufloat(a, fa) 
ub = ufloat(b, fb)
unu0 = ufloat(nu0, fnu0)

xx = np.linspace(0, 31, 10**4)

plt.plot(v, U, 'xr', markersize=6 , label = 'Messdaten', alpha=0.5)
#plt.plot(xx, g(xx, *para), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.plot(xx, test(xx, *par_test), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)
plt.xlabel(r'$v \, / \, kHz$')
plt.ylabel(r'$U_A \, / \, U_E v$')
plt.legend(loc="best")                  # legend position
plt.grid(True)                          # grid style
#plt.xlim(22, 40)
#plt.ylim(-0.05, 1.05)

plt.savefig('build/plot.pdf', bbox_inches = "tight")
#plt.show()
plt.clf() 

# Tabelle der Filterkurve
# print('Tabelle zur Filterkurve:')
# filter=pd.read_csv('data/filterkurve.txt',sep=' ', header=None, names=['Frequenz', 'Impulse'])
# print(filter.to_latex(index=False, column_format="c c"))

#Berechnung des Querschnitts
#werte derverschiedenen Proben

d_m = 14.38 ; d_l = 16.3 ; d_r =7.8
g_m = 14.08 ; g_l = 17.3 ; g_r = 7.40
n_m = 18.48 ; n_l = 14.5 ; n_r = 7.24
c_m = 7.87 ; c_l = 16.5  
def Q(m,l,r):
    return m/(l*r)

d_q = Q(d_m,d_l,d_r)
g_q = Q(g_m,g_l,g_r)
n_q = Q(n_m,n_l,n_r)

m = [d_m, g_m, n_m]
l = [d_l, g_l, n_l]
rho = [d_r, g_r, n_r]
Q = np.array([d_q, g_q, n_q])
stoffe = ['Dy2O3', 'Gd2O3', 'Nd2O3']

f = {'Stoff': stoffe, 'm/g': m, 'l/cm': l, 'rho_w/g/cm**3': rho, 'Q/cm**2': Q}
df = pd.DataFrame(data = f)
print(df.to_latex(index = False, column_format= "c c c c", decimal=',')) 

#print('Tabelle Dy:')
#dy=pd.read_csv('data/Dy203.txt',sep=' ', header=None, names=['wiederholung', 'ohne', 'mit'])
#print(dy.to_latex(index=False, column_format="c c c"))
  
#differenz der widerstände

list1 = [0.645, 0.68, 0.8, 0.72]
dy = np.mean(list1)
#print(dy)
list = [0.685, 0.75, 0.75]
gd = np.mean(list)
print(gd)

#Sus berechnen
#querschnitt
quer = (0.4**2)*np.pi
print(quer)

def su(x,d,f,q):
    return 2*((d*f)/(x*q))

#definition der widerstands differenzen
di_d = 0.711 
di_n = 0.19
di_g = 0.728
chi_d = su(2.539,di_d,quer,d_q )
chi_n = su(2.5,di_n,quer,n_q)
chi_g = su(2.485,di_g,quer,g_q)
print('Susz:')
print(chi_d)
print(chi_n)
print(chi_g)

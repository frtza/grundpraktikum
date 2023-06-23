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

# messung1=pd.read_csv('data/messung1.txt',sep=' ', header=None, names=['lei', 'f15', 'f30', 'f60'])
# print(messung1.to_latex(index=False, column_format="c c c c"))

# speed1=pd.read_csv('data/speed1.txt',sep=' ', header=None, names=['d','s','sig'])
# print(speed1.to_latex(index=False, column_format="c c c"))

lei, f15, f30, f60 = np.genfromtxt('data/messung1.txt', unpack=True, skip_header=1)  


#Prismawinkel
alpha = [80.06, 70.53, 54.74]
alpha = np.multiply(alpha, (np.pi/180))
c = 2700 #const.speed_of_light
nu0 = 2e6 

#Strömungsgeschwindigkeit

def v(nu, a):
    return (nu * c)/(2 * nu0 * np.cos(a)) 

v15 = np.zeros(5)
v30 = np.zeros(5)
v60 = np.zeros(5)

for j in range(5):
    v15[j] = v(f15[j], alpha[0])
    v30[j] = v(f30[j], alpha[1])
    v60[j] = v(f60[j], alpha[2])

f = {'v15/m/s': np.around(v15, 3), 'v30/m/s': np.around(v30, 3), 'v60/m/s': np.around(v60, 3)}
df = pd.DataFrame(data = f)
# print(df.to_latex(index = False, column_format= "c c c", decimal=',')) 

def f(v):
    return (2 * nu0 * v)/c

#plt.plot(v15, f(v15)/np.cos((80.06*np.pi)/180), 'xr', markersize=6 , label = 'Messdaten')
plt.plot(v15, f(v15), 'xr', markersize=6 , label = 'Messdaten')
print(v15)

#Ausgleichsgerade 

def g(x, a, b):
    return a * x + b

para, pcov = curve_fit(g, v15, f(v15))
a, b = para
fa, fb = np.sqrt(np.diag(pcov))
ua = ufloat(a, fa) 
ub = ufloat(b, fb)

# print('Ausgleichsgerade1:')
# print(ua)
# print(ub)
xx = np.linspace(0, 1, 10**4)
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)

plt.xlabel(r'$v \, / \, \mathrm{ms^{-1}}$')
plt.ylabel(r'$\frac{\Delta \nu}{\cos (\alpha)}$')
plt.legend(loc="best")                  
plt.grid(True)                          
#plt.xlim(0, 1)                  
#plt.ylim(0, 0.014)

plt.savefig('build/plot1.pdf', bbox_inches = "tight")
plt.show()
plt.clf() 

#plot2

v30 = abs(v30)
#plt.plot(v30, f(v30)/np.cos((70.53*np.pi)/180), 'xr', markersize=6 , label = 'Messdaten')
plt.plot(v30, f(v30), 'xr', markersize=6 , label = 'Messdaten')

para, pcov = curve_fit(g, v30, f(v30))
a, b = para
fa, fb = np.sqrt(np.diag(pcov))
ua = ufloat(a, fa) 
ub = ufloat(b, fb)
# print('Ausgleichsgerade2:')
# print(ua)
# print(ub)
xx = np.linspace(0, 1, 10**4)
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)

plt.xlabel(r'$v \, / \, \mathrm{ms^{-1}}$')
plt.ylabel(r'$\frac{\Delta \nu}{\cos (\alpha)}$')
plt.legend(loc="best")                  
plt.grid(True)                          
#plt.xlim(0, 1)                  
#plt.ylim(0, 0.018)

plt.savefig('build/plot2.pdf', bbox_inches = "tight")
plt.clf() 

#plot3

v60 = abs(v60)
#plt.plot(v60, f(v60)/np.cos((54.74*np.pi)/180), 'xr', markersize=6 , label = 'Messdaten')
plt.plot(v60, f(v60), 'xr', markersize=6 , label = 'Messdaten')

para, pcov = curve_fit(g, v60, f(v60))
a, b = para
fa, fb = np.sqrt(np.diag(pcov))
ua = ufloat(a, fa) 
ub = ufloat(b, fb)
# print('Ausgleichsgerade3:')
# print(v60)
# print(ua)
# print(ub)
xx = np.linspace(0, 1, 10**4)
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)

plt.xlabel(r'$v \, / \, \mathrm{ms^{-1}}$')
plt.ylabel(r'$\frac{\Delta \nu}{\cos (\alpha)}$')
plt.legend(loc="best")                  
plt.grid(True)                          
#plt.xlim(0, 1)                  
#plt.ylim(0, 0.015)

plt.savefig('build/plot3.pdf', bbox_inches = "tight")
#plt.show()
plt.clf() 

#Strömungsprofil

speed1=pd.read_csv('data/speed1.txt',sep=' ', header=None, names=['d','s','sig'])
#print(speed1.to_latex(index=False, column_format="c c c"))

speed2=pd.read_csv('data/speed2.txt',sep=' ', header=None, names=['d','s','sig'])
#print(speed2.to_latex(index=False, column_format="c c c"))

#daten
d1, s1, sig1 = np.genfromtxt('data/speed1.txt', unpack=True, skip_header=1)
d2, s2, sig2 = np.genfromtxt('data/speed2.txt', unpack=True, skip_header=1)

#print(d1)
d1 = (6/4) * d1 

plt.plot(d1, s1, 'xr', markersize=6 , label = 'Momentangeschwindigkeit für P = 45%')

plt.xlabel(r'$x \, / \, \mathrm{mm}$')
plt.ylabel(r'$v \, / \, \mathrm{cms^{-1}}$')
plt.legend(loc="best")                  
plt.grid(True)
plt.savefig('build/plot4.pdf',bbox_inches = "tight")
plt.clf()

plt.plot(d1, sig1, 'xr', markersize=6 , label = 'Streuintensität für P = 45%')

plt.xlabel(r'$x \, / \, \mathrm{mm}$')
plt.ylabel(r'$v \, / \, \mathrm{cms^{-1}}$')
plt.legend(loc="best")                  
plt.grid(True)
plt.savefig('build/plot5.pdf',bbox_inches = "tight")
plt.clf()

d2 = (6/4) * d2 
plt.plot(d2, s2, 'xr', markersize=6 , label = 'Momentangeschwindigkeit für P = 70%')

plt.xlabel(r'$x \, / \, \mathrm{mm}$')
plt.ylabel(r'$v \, / \, \mathrm{cms^{-1}}$')
plt.legend(loc="best")                  
plt.grid(True)
plt.savefig('build/plot6.pdf',bbox_inches = "tight")
plt.clf()

plt.plot(d2, sig2, 'xr', markersize=6 , label = 'Streuintensität für P = 70%')

plt.xlabel(r'$x \, / \, \mathrm{mm}$')
plt.ylabel(r'$v \, / \, \mathrm{cms^{-1}}$')
plt.legend(loc="best")                  
plt.grid(True)
plt.savefig('build/plot7.pdf',bbox_inches = "tight")
plt.clf()

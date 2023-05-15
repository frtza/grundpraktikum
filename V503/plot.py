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

teil, q_korr, fehl = np.genfromtxt('data/ladung.txt', unpack = True, skip_header =1)
xx = np.linspace(0, 10, 100)
#plt.plot(teil,q_korr*10**(-19),'x',linewidth = 1, label = 'Korigierte Ladungen')
errY = fehl
plt.errorbar(teil, (q_korr)*10**(-19), yerr = errY * 10**(-19), fmt='o' )
plt.xlabel(r'Messung')
plt.ylabel(r'$q [C]$')
plt.legend(loc="best")                  # legend position
plt.grid(True) 
plt.savefig('build/plot1.pdf')

#Stufen mitteln

def mean(x,n):
    return sum(x)/n

def std(x,mittel,n):
    return unp.sqrt(n)*unp.sqrt((1/(n-1))*sum(x - mittel)**2)

#Stufe1
stufe1 = ufloat(q_korr[0],fehl[0])*10**(-19)
print(stufe1)
#stufe2
q = q_korr[1:4]
fehler = fehl[1:4]
korr = uarray(q,fehler)
n = 3
stufe2 = mean(korr,3)*10**(-19)
#std2 = std(korr,m2,n) 
#stufe2 = uarray(m2,std)
print(stufe2)
#stufe3
q3 = q_korr[4:9]
fehler3 = fehl[4:9]
korr3 = uarray(q3,fehler3)
n = 5
stufe3 = mean(korr3,5)*10**(-19)
print(stufe3)
#stufe4
q4 = q_korr[9:13]
fehler4 = fehl[9:13]
korr4 = uarray(q4,fehler4)
n = 4
stufe4 = mean(korr4,4)*10**(-19)
print(stufe4)

#Differenzen
#1 und 2
d1 = (stufe1 - stufe2)/2
print(d1)
#2 und 3
d2 = stufe2 - stufe3
print(d2)
#3 und 4
d3 = stufe3 -stufe4
print(d3)

e = (d2+d3)/2
et = const.e
abweichung = (e -et) / et
print(et)
print(abweichung)

F = ufloat(96485.3399, 0.0024)
a = F/e 
at = 6.02214086 * 10**(23)
abweichung2 = (a -at) / at
print(abweichung2)

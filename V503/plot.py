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
#Bedingung für die Gültigkeit der Messwerte prüfen
#Fubntion der Bedingung 
def bed(v_0,v_ab,v_auf):
    return ((2 * v_0) - (v_ab - v_auf)) / (v_ab - v_auf)

#strecke
s = ufloat(0.5, 0.1) #mm

#daten einlesen
tab, tauf, t0 = np.genfromtxt('data/spannung1/1.txt', unpack=True, skip_header=1)
#mitteln der Daten
print('t_auf:')
mean_auf = np.mean(tauf)
std_auf = np.std(tauf)
t_mittel_auf = ufloat(mean_auf,std_auf)
print(t_mittel_auf)

print('t_ab:')
mean_ab = np.mean(tab)
std_ab = np.std(tab)
t_mittel_ab = ufloat(mean_ab,std_ab)
print(t_mittel_ab)
#Berechnung von v
# v= s/t
def v(z,d):
    return z/d 
vauf = v(s,t_mittel_auf)
vab = v(s,t_mittel_ab)
v0 = v(s,t0)

print('Bedingung prüfen:')
#ver für verhältnis
ver = bed(v0,vab,vauf)
print(ver)

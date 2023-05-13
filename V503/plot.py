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

g = const.g
d = ufloat(7.6250, 0.0051)*10**(-3)  
p = 1*10**(5)   
B = 8.226 * 10**3
#rholuft = 1.
rhoOel = 886

#Bedingung für die Gültigkeit der Messwerte prüfen
#Fubntion der Bedingung 
def bed(v_0,v_ab,v_auf):
    return ((2 * v_0) - (v_ab - v_auf)) / (v_ab - v_auf)

### Vorschlag:
def bedt(v_0, v_ab, v_auf):
	return abs((2 * v_0) / (v_ab - v_auf) - 1)
### Gibt relative Abweichung zurueck

#strecke
s = ufloat(0.0005, 0.0001) #m

#daten einlesen
tauf, tab, t0 = np.genfromtxt('data/spannung1/2.txt', unpack=True, skip_header=1)
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
def v(s,t):
    return s/t 

vauf = v(s,t_mittel_auf)
vab = v(s,t_mittel_ab)
v0 = v(s,t0[0])
print('Bedingung prüfen:')
#ver für verhältnis
ver = bedt(v0,vab,vauf) 
print(ver)


#ladung bestimmen 
tauf1, tab1, t01 = np.genfromtxt('data/spannung1/1.txt', unpack=True, skip_header=1)
tauf3, tab3, t03 = np.genfromtxt('data/spannung1/3.txt', unpack=True, skip_header=1)
tauf4, tab4, t04 = np.genfromtxt('data/spannung1/4.txt', unpack=True, skip_header=1)
tauf5, tab5, t05 = np.genfromtxt('data/spannung1/5.txt', unpack=True, skip_header=1)
tauf6, tab6, t06 = np.genfromtxt('data/spannung1/6.txt', unpack=True, skip_header=1)
tauf7, tab7, t07 = np.genfromtxt('data/spannung1/7.txt', unpack=True, skip_header=1)
tauf8, tab8, t08 = np.genfromtxt('data/spannung2/8.txt', unpack=True, skip_header=1)
tauf10, tab10, t010 = np.genfromtxt('data/spannung2/10.txt', unpack=True, skip_header=1)
tauf12, tab12, t012 = np.genfromtxt('data/spannung2/12.txt', unpack=True, skip_header=1)
tauf13, tab13, t013 = np.genfromtxt('data/spannung2/13.txt', unpack=True, skip_header=1)
tauf14, tab14, t014 = np.genfromtxt('data/spannung2/14.txt', unpack=True, skip_header=1)
tauf16, tab16, t016 = np.genfromtxt('data/spannung2/16.txt', unpack=True, skip_header=1)
tauf17, tab17, t017 = np.genfromtxt('data/spannung2/17.txt', unpack=True, skip_header=1)

mean_auf1 = np.mean(tauf1)
std_auf1 = np.std(tauf1)
t_mittel_auf1 = ufloat(mean_auf1,std_auf1)

mean_ab1 = np.mean(tab1)
std_ab1 = np.std(tab1)
t_mittel_ab1 = ufloat(mean_ab1,std_ab1)

vauf1 = v(s,t_mittel_auf1)
vab1 = v(s,t_mittel_ab1)
v01 = v(s,t01[0])

mean_auf3 = np.mean(tauf3)
std_auf3 = np.std(tauf3)
t_mittel_auf3 = ufloat(mean_auf3,std_auf3)

mean_ab3 = np.mean(tab3)
std_ab3 = np.std(tab3)
t_mittel_ab3 = ufloat(mean_ab3,std_ab3)

vauf3 = v(s,t_mittel_auf3)
vab3 = v(s,t_mittel_ab3)
v03 = v(s,t03[0])

mean_auf4 = np.mean(tauf4)
std_auf4 = np.std(tauf4)
t_mittel_auf4 = ufloat(mean_auf4,std_auf4)

mean_ab4 = np.mean(tab4)
std_ab4 = np.std(tab4)
t_mittel_ab4 = ufloat(mean_ab4,std_ab4)

vauf4 = v(s,t_mittel_auf4)
vab4 = v(s,t_mittel_ab4)
v04 = v(s,t04[0])

mean_auf5 = np.mean(tauf5)
std_auf5 = np.std(tauf5)
t_mittel_auf5 = ufloat(mean_auf5,std_auf5)

mean_ab5 = np.mean(tab5)
std_ab5 = np.std(tab5)
t_mittel_ab5 = ufloat(mean_ab5,std_ab5)

vauf5 = v(s,t_mittel_auf5)
vab5 = v(s,t_mittel_ab5)
v05 = v(s,t05[0])

mean_auf6 = np.mean(tauf6)
std_auf6 = np.std(tauf6)
t_mittel_auf6 = ufloat(mean_auf6,std_auf6)

mean_ab6 = np.mean(tab6)
std_ab6 = np.std(tab6)
t_mittel_ab6 = ufloat(mean_ab6,std_ab6)

vauf6 = v(s,t_mittel_auf6)
vab6 = v(s,t_mittel_ab6)
v06 = v(s,t06[0])

mean_auf7 = np.mean(tauf7)
std_auf7 = np.std(tauf7)
t_mittel_auf7 = ufloat(mean_auf7,std_auf7)

mean_ab7 = np.mean(tab7)
std_ab7 = np.std(tab7)
t_mittel_ab7 = ufloat(mean_ab7,std_ab7)

vauf7 = v(s,t_mittel_auf7)
vab7 = v(s,t_mittel_ab7)
v07 = v(s,t07[0])

mean_auf8 = np.mean(tauf8)
std_auf8 = np.std(tauf8)
t_mittel_auf8 = ufloat(mean_auf8,std_auf8)

mean_ab8 = np.mean(tab8)
std_ab8 = np.std(tab8)
t_mittel_ab8 = ufloat(mean_ab8,std_ab8)

vauf8 = v(s,t_mittel_auf8)
vab8 = v(s,t_mittel_ab8)
v08 = v(s,t08[0])

mean_auf10 = np.mean(tauf10)
std_auf10 = np.std(tauf10)
t_mittel_auf10 = ufloat(mean_auf10,std_auf10)

mean_ab10 = np.mean(tab10)
std_ab10 = np.std(tab10)
t_mittel_ab10 = ufloat(mean_ab10,std_ab10)

vauf10 = v(s,t_mittel_auf10)
vab10 = v(s,t_mittel_ab10)
v010 = v(s,t010[0])

mean_auf12 = np.mean(tauf12)
std_auf12 = np.std(tauf12)
t_mittel_auf12 = ufloat(mean_auf12,std_auf12)

mean_ab12 = np.mean(tab12)
std_ab12 = np.std(tab12)
t_mittel_ab12 = ufloat(mean_ab12,std_ab12)

vauf12 = v(s,t_mittel_auf12)
vab12 = v(s,t_mittel_ab12)
v012 = v(s,t012[0])

mean_auf13 = np.mean(tauf13)
std_auf13 = np.std(tauf13)
t_mittel_auf13 = ufloat(mean_auf13,std_auf13)

mean_ab13 = np.mean(tab13)
std_ab13 = np.std(tab13)
t_mittel_ab13 = ufloat(mean_ab13,std_ab13)

vauf13 = v(s,t_mittel_auf13)
vab13 = v(s,t_mittel_ab13)
v013 = v(s,t013[0])

mean_auf14 = np.mean(tauf14)
std_auf14 = np.std(tauf14)
t_mittel_auf14 = ufloat(mean_auf14,std_auf14)

mean_ab14 = np.mean(tab14)
std_ab14 = np.std(tab14)
t_mittel_ab14 = ufloat(mean_ab14,std_ab14)

vauf14 = v(s,t_mittel_auf14)
vab14 = v(s,t_mittel_ab14)
v014 = v(s,t014[0])

mean_auf16 = np.mean(tauf16)
std_auf16 = np.std(tauf16)
t_mittel_auf16 = ufloat(mean_auf16,std_auf16)

mean_ab16 = np.mean(tab16)
std_ab16 = np.std(tab16)
t_mittel_ab16 = ufloat(mean_ab16,std_ab16)

vauf16 = v(s,t_mittel_auf16)
vab16 = v(s,t_mittel_ab16)
v016 = v(s,t016[0])

mean_auf17 = np.mean(tauf17)
std_auf17 = np.std(tauf17)
t_mittel_auf17 = ufloat(mean_auf17,std_auf17)

mean_ab17 = np.mean(tab17)
std_ab17 = np.std(tab17)
t_mittel_ab17 = ufloat(mean_ab17,std_ab17)

vauf17 = v(s,t_mittel_auf17)
vab17 = v(s,t_mittel_ab17)
v017 = v(s,t017[0])

#Funktionen:

#feldstärke des plattenkondensators
def E(U):
    return U / d

E1 = E(t01[2])
E3 = E(t03[2])
E4 = E(t04[2])
E5 = E(t05[2])
E6 = E(t06[2])
E7 = E(t07[2])
E8 = E(t08[2])
E10 = E(t010[2])
E12 = E(t012[2])
E13 = E(t013[2])
E14 = E(t014[2])
E16 = E(t016[2])
E17 = E(t017[2])

#viskosität

visko1 = 1.8325 * 10**(-5)
visko3 = 1.8375 * 10**(-5)
visko4 = 1.8375 * 10**(-5)
visko5 = 1.8420 * 10**(-5)
visko6 = 1.8420 * 10**(-5)
visko7 = 1.8420 * 10**(-5)
visko8 = 1.8420 * 10**(-5)
visko10 = 1.8475 * 10**(-5)
visko12 = 1.8475 * 10**(-5)
visko13 = 1.8475 * 10**(-5)
visko14 = 1.8420 * 10**(-5)
visko16 = 1.8475 * 10**(-5)
visko17 = 1.8475 * 10**(-5)

#Radius der Öltröpfchen
def r(vab, vauf,visko):
    if vab-vauf<0:
        return 0
    else:
    return unp.sqrt((9/4)*(visko/g)*(vab-vauf)/(rhoOel))


#unkorrigierte ladung
def q(vab, vauf, E, r):
    if r==0:
        return 0
    else:
        return (3/4)*np.pi*visko*r*((vab+vauf)/(E))
#korrigierte ladung

def qkorr(q, r):
    if q==0:
        return 0
    else:
        return q*(1+(B/(p*r)))**(3/2)
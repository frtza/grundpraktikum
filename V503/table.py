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
B = 8.226 * 10**(-3)
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
def q(vab, vauf, E, r, visko):
    if r==0:
        return 0
    else:
        return (3)*np.pi*visko*r*((vab+vauf)/(E))
#korrigierte ladung

def qkorr(q, r):
    if q==0:
        return 0
    else:
        return q*(1+(B/(p*r)))**(-3/2)

r1 = r(vab1, vauf1, visko1)
q1 = q(vab1, vauf1, E1, r1, visko1)
qkorr1 = qkorr(q1, r1)

r3 = r(vab3, vauf3, visko3)
q3 = q(vab3, vauf3, E3, r3, visko3)
qkorr3 = qkorr(q3, r3)

r4 = r(vab4, vauf4, visko4)
q4 = q(vab4, vauf4, E4, r4, visko4)
qkorr4 = qkorr(q4, r4)

r5 = r(vab5, vauf5, visko5)
q5 = q(vab5, vauf5, E5, r5, visko5)
qkorr5 = qkorr(q5, r5)

r6 = r(vab6, vauf6, visko6)
q6 = q(vab6, vauf6, E6, r6, visko6)
qkorr6 = qkorr(q6, r6)

r7 = r(vab7, vauf7, visko7)
q7 = q(vab7, vauf7, E7, r7, visko7)
qkorr7 = qkorr(q7, r7)

r8 = r(vab8, vauf8, visko8)
q8 = q(vab8, vauf8, E8, r8, visko8)
qkorr8 = qkorr(q8, r8)

r10 = r(vab10, vauf10, visko10)
q10 = q(vab10, vauf10, E10, r10, visko10)
qkorr10 = qkorr(q10, r10)

r12 = r(vab12, vauf12, visko12)
q12 = q(vab12, vauf12, E12, r12, visko12)
qkorr12 = qkorr(q12, r12)

r13 = r(vab13, vauf13, visko13)
q13 = q(vab13, vauf13, E13, r13, visko13)
qkorr13 = qkorr(q13, r13)

r14 = r(vab14, vauf14, visko14)
q14 = q(vab14, vauf14, E14, r14, visko14)
qkorr14 = qkorr(q14, r14)

r16 = r(vab16, vauf16, visko16)
q16 = q(vab16, vauf16, E16, r16, visko16)
qkorr16 = qkorr(q16, r16)

r17 = r(vab17, vauf17, visko17)
q17 = q(vab17, vauf17, E17, r17, visko17)
qkorr17 = qkorr(q17, r17)


print('korrigierte Ladung')
print('r:\n', r1, '\nq:\n', q1, '\nqkorr1:\n', qkorr1)
print('r:\n', r3, '\nq:\n', q3, '\nqkorr3:\n', qkorr3)
print('r:\n', r4, '\nq:\n', q4, '\nqkorr4:\n', qkorr4)
print('r:\n', r5, '\nq:\n', q5, '\nqkorr5:\n', qkorr5)
print('r:\n', r6, '\nq:\n', q6, '\nqkorr6:\n', qkorr6)
print('r:\n', r7, '\nq:\n', q7, '\nqkorr7:\n', qkorr7)
print('r:\n', r8, '\nq:\n', q8, '\nqkorr8:\n', qkorr8)
print('r:\n', r10, '\nq:\n', q10, '\nqkorr10:\n', qkorr10)
print('r:\n', r12, '\nq:\n', q12, '\nqkorr12:\n', qkorr12)
print('r:\n', r13, '\nq:\n', q13, '\nqkorr13:\n', qkorr13)
print('r:\n', r14, '\nq:\n', q14, '\nqkorr14:\n', qkorr14)
print('r:\n', r16, '\nq:\n', q16, '\nqkorr16:\n', qkorr16)
print('r:\n', r17, '\nq:\n', q17, '\nqkorr17:\n', qkorr17)

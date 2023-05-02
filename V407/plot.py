# Bibliotheken importieren

import matplotlib.pyplot as plt

import numpy as np

import scipy.constants as const
from scipy.optimize import curve_fit

import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds
from uncertainties import ufloat


# Daten einlesen

ang_s, I_s, scale_s = np.genfromtxt('data/s_pol.txt', unpack=True) # Winkel in deg, Strom in uA, Skala in uA
ang_p, I_p, scale_p = np.genfromtxt('data/p_pol.txt', unpack=True) # Winkel in deg, Strom in uA, Skala in uA

I_s = unp.uarray(I_s, 0.02*scale_s) # Fehler der Stromstaerke in uA entspricht zwei Prozent der Skala
I_p = unp.uarray(I_p, 0.02*scale_p) # Fehler der Stromstaerke in uA entspricht zwei Prozent der Skala

lam = ufloat(681, 3) # Wellenlaenge des Lasers, Peak bei 680 nm, Bereich von 678 nm bis 684 nm

I_g = ufloat(490, 20) # Grundintensitaet in uA

I_d = ufloat(4.7e-3, 0.2e-3) # Dunkelstrom in uA

print(ang_s, I_s, I_p, lam, I_g, I_d)



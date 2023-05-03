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
I_s = I_s - I_d # Dunkelstromkorrektur in uA
I_p = I_p - I_d # Dunkelstromkorrektur in uA

ang_min = 49 # Intensitaetsminimum bei ang in deg
I_min = ufloat(43e-3, 2e-3) # Intensitaetsminimum in uA


# Fit-Funktionen definieren und Parameter optimieren
def amp_s(ang, n):
	ang = ang * np.pi / 180
	return ((np.sqrt(n**2 - (np.sin(ang))**2) - np.cos(ang))**2 / (n**2 - 1))**2
def amp_p(ang, n):
	ang = ang * np.pi / 180
	return ((n**2 * np.cos(ang) - np.sqrt(n**2 - (np.sin(ang))**2)) / (n**2 * np.cos(ang) + np.sqrt(n**2 - (np.sin(ang))**2)))**2

par_s, cov_s = curve_fit(amp_s, ang_s, noms(I_s/I_g), sigma=stds(I_s/I_g), p0=(2))
err_s = np.sqrt(np.diag(cov_s))
par_p, cov_p = curve_fit(amp_p, ang_p, noms(I_p/I_g), sigma=stds(I_p/I_g), p0=(2))
err_p = np.sqrt(np.diag(cov_p))

# Daten und Fits visualisieren
xx = np.linspace(4, 88, 10000)

plt.plot(xx, np.sqrt(amp_s(xx, par_s))*3/7, label='Regression')
plt.errorbar(ang_s, noms(I_s/I_g), yerr=stds(I_s/I_g), fmt='.', ms=4, color='black', ecolor='gray', capsize=1.5, label='Messdaten')
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$I / I_0$')
plt.legend()
plt.savefig('build/plot_s.pdf')
plt.close()

plt.plot(xx, amp_p(xx, par_p)/3, label='Regression')
plt.errorbar(ang_p[:31], noms(I_p[:31]/I_g), yerr=stds(I_p[:31]/I_g), fmt='.', ms=4, color='black', ecolor='gray', capsize=1.5, label='Messdaten')
plt.plot(ang_p[31:], noms(I_p[31:]/I_g), '.k', ms=4)
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$I / I_0$')
plt.legend()
plt.savefig('build/plot_p.pdf')
plt.close()


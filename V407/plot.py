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

ang_min = 49 # Intensitaetsminimumswinkel in deg
I_min = ufloat(43e-3, 2e-3) # Intensitaetsminimum in uA


# Implizite Gleichungen des Brechungsindex n
def imp_p(ang, n):
	ang = ang * np.pi / 180
	return (n**2 * np.cos(ang) - np.sqrt(n**2 - np.sin(ang)**2)) / (n**2 * np.cos(ang) + np.sqrt(n**2 - np.sin(ang)**2))
def imp_s(ang, n):
	ang = ang * np.pi / 180
	return (np.sqrt(n**2 - np.sin(ang)**2) - np.cos(ang))**2 / (n**2 - 1)

# Loesungsfaelle definieren
def n_p_1(ang, V):
	ang, sV = ang * np.pi / 180, np.sqrt(V)
	def fac(ang, sV):
		return 1/np.cos(ang) * (1 + sV) / (1 - sV)
	return np.sqrt(1/2 * fac(ang, sV)**2 - np.sqrt(1/4 * fac(ang, sV)**4 - np.sin(ang)**2 * fac(ang, sV)**2))
def n_p_2(ang, V):
	ang, sV = ang * np.pi / 180, np.sqrt(V)
	def fac(ang, sV):
		return 1/np.cos(ang) * (1 - sV) / (1 + sV)
	return np.sqrt(1/2 * fac(ang, sV)**2 - np.sqrt(1/4 * fac(ang, sV)**4 - np.sin(ang)**2 * fac(ang, sV)**2))
def n_p_3(ang, V):
	ang, sV = ang * np.pi / 180, np.sqrt(V)
	def fac(ang, sV):
		return 1/np.cos(ang) * (1 + sV) / (1 - sV)
	return np.sqrt(1/2 * fac(ang, sV)**2 + np.sqrt(1/4 * fac(ang, sV)**4 - np.sin(ang)**2 * fac(ang, sV)**2))
def n_p_4(ang, V):
	ang, sV = ang * np.pi / 180, np.sqrt(V)
	def fac(ang, sV):
		return 1/np.cos(ang) * (1 - sV) / (1 + sV)
	return np.sqrt(1/2 * fac(ang, sV)**2 + np.sqrt(1/4 * fac(ang, sV)**4 - np.sin(ang)**2 * fac(ang, sV)**2))

def n_s_1(ang, V):
	ang, sV = ang * np.pi / 180, np.sqrt(V)
	return np.sqrt(1 - sV * ((2 * np.cos(ang)) / (1 + sV))**2)
def n_s_2(ang, V):
	ang, sV = ang * np.pi / 180, np.sqrt(V)
	return np.sqrt(1 + sV * ((2 * np.cos(ang)) / (1 - sV))**2)


# Intensitaetsverhaeltnis spezifizieren und Ergebnisse darstellen

V = 0.0225

# Parallele Polarisation

xx, yy = np.arange(0, 85, 0.002), np.arange(0.225, 5.2, 0.002)
XX, YY = np.meshgrid(xx, yy)
plt.contour(XX, YY, imp_p(XX, YY), [-np.sqrt(V), np.sqrt(V)], linestyles='solid', linewidths=7.5, alpha=0.15, colors='#ff7f0e')

plt.plot(100, 100, linewidth=5, alpha=0.15, c='#ff7f0e', label='Konturen')

xx = np.linspace(12.25, 85.25, 10000)
plt.plot(xx, n_p_1(xx, V), c='#d62728', label='Lösung 1a')
xx = np.linspace(12.25, 85.25, 500000)
plt.plot(xx, n_p_2(xx, V), c='#d62728', alpha=0.5, label='Lösung 1b')

xx = np.linspace(-0.5, 75.4, 10000)
plt.plot(xx, n_p_3(xx, V), c='#ff7f0e', label='Lösung 2a')
xx = np.linspace(-0.5, 82.075, 500000)
plt.plot(xx, n_p_4(xx, V), c='#ff7f0e', alpha=0.5, label='Lösung 2b')

plt.xlim(-3.5, 87.5)
plt.ylim(-0.275, 5.5)
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$n$')
leg = plt.legend(handlelength=1.5, borderpad=0.75, loc='best', edgecolor='k', facecolor='none')
leg.get_frame().set_linewidth(0.25)
plt.savefig('build/plot_ip.pdf')
plt.close()

# Senkrechte Polarisation

xx, yy = np.arange(-6, 86, 0.002), np.arange(0.5, 1.7, 0.002)
XX, YY = np.meshgrid(xx, yy)
plt.contour(XX, YY, imp_s(XX, YY), [-np.sqrt(V), np.sqrt(V)], linestyles='solid', linewidths=7.5, alpha=0.15, colors='#ff7f0e')

plt.plot(100, 100, linewidth=5, alpha=0.15, c='#ff7f0e', label='Konturen')

xx = np.linspace(-6.3, 86.3, 10000)
plt.plot(xx, n_s_1(xx, V), c='#d62728', label='Lösung 1')
plt.plot(xx, n_s_2(xx, V), c='#ff7f0e', label='Lösung 2')

plt.ylim(0.5, 1.7)
plt.xlim(-10, 90)
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$n$')
leg = plt.legend(handlelength=1.5, borderpad=0.75, loc='best', edgecolor='k', facecolor='none')
leg.get_frame().set_linewidth(0.25)
plt.savefig('build/plot_is.pdf')
plt.close()


# Fit-Funktionen definieren und Parameter optimieren
def amp_s(ang, n, s):
	ang = ang * np.pi / 180
	return s * ((np.sqrt(n**2 - (np.sin(ang))**2) - np.cos(ang))**2 / (n**2 - 1))**2
def amp_p(ang, n, s):
	ang = ang * np.pi / 180
	return s * ((n**2 * np.cos(ang) - np.sqrt(n**2 - (np.sin(ang))**2)) / (n**2 * np.cos(ang) + np.sqrt(n**2 - (np.sin(ang))**2)))**2

# Parameter mit Skalierung als Freiheitsgrad
par_s, cov_s = curve_fit(amp_s, ang_s, noms(I_s/I_g), sigma=stds(I_s/I_g), p0=(2, 2))
err_s = np.sqrt(np.diag(cov_s))
par_p, cov_p = curve_fit(amp_p, ang_p, noms(I_p/I_g), sigma=stds(I_p/I_g), p0=(2, 2))
err_p = np.sqrt(np.diag(cov_p))

# Parameter der Formel ohne Skalierung
f_par_s, f_cov_s = curve_fit(lambda ang, n: amp_s(ang, n, 1), ang_s, noms(I_s/I_g), sigma=stds(I_s/I_g), p0=(2))
f_err_s = np.sqrt(np.diag(f_cov_s))
f_par_p, f_cov_p = curve_fit(lambda ang, n: amp_p(ang, n, 1), ang_p, noms(I_p/I_g), sigma=stds(I_p/I_g), p0=(2))
f_err_p = np.sqrt(np.diag(f_cov_p))


# Daten und Fits visualisieren
xx = np.linspace(4, 88, 10000)

plt.plot(xx, amp_s(xx, *par_s), label='Regression', c='olivedrab', zorder=1)
plt.errorbar(ang_s, noms(I_s/I_g), yerr=stds(I_s/I_g), fmt='.', ms=4, color='black', capsize=1.5, label='Messdaten', zorder=10)
plt.plot(xx, amp_s(xx, *par_s), c='olivedrab', alpha=0.5, zorder=100)
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$I / I_0$')
plt.xticks(np.arange(5, 95, 10))
leg = plt.legend(borderpad=0.75, loc='best', edgecolor='k', facecolor='none')
leg.get_frame().set_linewidth(0.25)
plt.savefig('build/plot_s.pdf')
plt.close()

plt.plot(xx, amp_p(xx, *par_p), label='Regression', c='olivedrab', zorder=1)
plt.errorbar(ang_p[:31], noms(I_p[:31]/I_g), yerr=stds(I_p[:31]/I_g), fmt='.', ms=4, color='black', capsize=1.5, label='Messdaten', zorder=10)
plt.plot(ang_p[31:], noms(I_p[31:]/I_g), '.k', ms=4, zorder=10)
plt.plot(xx, amp_p(xx, *par_p), c='olivedrab', alpha=0.5, zorder=100)
plt.ylim(-0.01, 0.17)
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$I / I_0$')
plt.xticks(np.arange(5, 95, 10))
plt.yticks(np.arange(0, 0.18, 0.04))
leg = plt.legend(borderpad=0.75, loc='best', edgecolor='k', facecolor='none')
leg.get_frame().set_linewidth(0.25)
plt.savefig('build/plot_p.pdf')
plt.close()


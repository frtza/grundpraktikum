import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds, uarray as uar

# Daten einlesen
a_ref, b_ref = np.genfromtxt('data/reflexion.txt', unpack=True)
a_bre, b_bre = np.genfromtxt('data/brechung.txt', unpack=True)
a_dis, g_dis, r_dis = np.genfromtxt('data/dispersion.txt', unpack=True)
k_beu, g_beu_1, g_beu_2, r_beu_1, r_beu_2 = np.genfromtxt('data/beugung.txt', unpack=True)

# Ablesefehler einfügen
a_ref, b_ref, a_dis, g_dis, r_dis, g_beu_1, g_beu_2, r_beu_1, r_beu_2 = uar(a_ref, 1), uar(b_ref, 1), uar(a_dis, 1), uar(g_dis, 1), uar(r_dis, 1), uar(g_beu_1, 1), uar(g_beu_2, 1), uar(r_beu_1, 1), uar(r_beu_2, 1)
a_bre, b_bre = uar(a_bre, 0.5), uar(b_bre, 0.5)

# Messreihen trennen
k_beu_600, g_beu_600_1, g_beu_600_2, r_beu_600_1, r_beu_600_2 = k_beu[0:2], g_beu_1[0:2], g_beu_2[0:2], r_beu_1[0:2], r_beu_2[0:2]
k_beu_300, g_beu_300_1, g_beu_300_2, r_beu_300_1, r_beu_300_2 = k_beu[2:5], g_beu_1[2:5], g_beu_2[2:5], r_beu_1[2:5], r_beu_2[2:5]
k_beu_100, g_beu_100_1, g_beu_100_2, r_beu_100_1, r_beu_100_2 = k_beu[5:12], g_beu_1[5:12], g_beu_2[5:12], r_beu_1[5:12], r_beu_2[5:12]

# Gerade definieren
def lin(x, a, b):
	return a * x + b

# Sinus definieren
def sin(w):
	w = np.pi * w / 180
	return unp.sin(w)

### Reflexionsgesetz:

# Regression
par, cov = np.polyfit(noms(a_ref), noms(b_ref), deg=1, cov=True)
err = np.sqrt(np.diag(cov))
fit = uar(par, err)

# Plot
x = np.array([10, 80])
plt.plot(x, lin(x, *par), c='olivedrab', label='Regression')
plt.errorbar(noms(a_ref), noms(b_ref), xerr=stds(a_ref), yerr=stds(b_ref), fmt='.k', ms=0, label='Messdaten')
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$\beta \mathbin{/} \unit{\degree}$')
plt.legend()
plt.savefig('build/reflexion.pdf')
plt.close()

### Brechungsgesetz:

# Regression
par, cov = np.polyfit(noms(sin(a_bre)), noms(sin(b_bre)), deg=1, cov=True)
err = np.sqrt(np.diag(cov))
fit = uar(par, err)

# Plot
x = np.array([0.5, 1.0])
plt.plot(x, lin(x, *par), c='olivedrab', label='Regression')
plt.errorbar(noms(sin(a_bre)), noms(sin(b_bre)), xerr=stds(sin(a_bre)), yerr=stds(sin(b_bre)), fmt='.k', ms=0, label='Messdaten')
plt.xlabel(r'$\sin(\alpha)$')
plt.ylabel(r'$\sin(\beta)$')
plt.legend()
plt.savefig('build/brechung.pdf')
plt.close()

### Planparallele Platten:


### Prisma:


### Beugung am Gitter:

# Gitterkostanten
d = 1e-3 / np.array([600, 600, 300, 300, 300, 100, 100, 100, 100, 100, 100, 100])

# Mittelwerte
g_beu = (g_beu_1 + g_beu_2) / 2
r_beu = (r_beu_1 + r_beu_2) / 2

# Regression
par_g, cov_g = np.polyfit(k_beu, noms(d * sin(g_beu)), deg=1, cov=True)
err_g = np.sqrt(np.diag(cov_g))
fit_g = uar(par_g, err_g)
par_r, cov_r = np.polyfit(k_beu, noms(d * sin(r_beu)), deg=1, cov=True)
err_r = np.sqrt(np.diag(cov_r))
fit_r = uar(par_r, err_r)

# Plot
x = np.array([-0.5, 6.5])
plt.plot(x, lin(x, *par_r), c='#ff3900', label='Regression')
plt.errorbar(k_beu, noms(d * sin(r_beu)), yerr=stds(d * sin(r_beu)), fmt='o', c='#b32700', mfc='none', ms=2, label='Messdaten')
plt.xlabel(r'$k$')
plt.ylabel(r'$d\sin(\varphi) \mathbin{/} \unit{\meter}$')
plt.legend()
plt.savefig('build/beugung_rot.pdf')
plt.close()
plt.plot(x, lin(x, *par_g), c='#65ff00', label='Regression')
plt.errorbar(k_beu, noms(d * sin(g_beu)), yerr=stds(d * sin(g_beu)), fmt='o', c='#47b300', mfc='none', ms=2, label='Messdaten')
plt.xlabel(r'$k$')
plt.ylabel(r'$d\sin(\varphi) \mathbin{/} \unit{\meter}$')
plt.legend()
plt.savefig('build/beugung_gruen.pdf')
plt.close()


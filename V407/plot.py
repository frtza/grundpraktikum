# Bibliotheken importieren

import matplotlib.pyplot as plt

import numpy as np

import scipy.constants as const
from scipy.optimize import curve_fit

import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds
from uncertainties import ufloat

import warnings

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
I_min = ufloat(43e-3, 2e-3) # Korrigiertes Intensitaetsminimum in uA
I_min = I_min - I_d

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


def n_p(ang, V): # Wahl von n_p_3 fuer Unsicherheiten
	ang, sV = ang * np.pi / 180, unp.sqrt(V)
	def fac(ang, sV):
		return 1/unp.cos(ang) * (1 + sV) / (1 - sV)
	return unp.sqrt(1/2 * fac(ang, sV)**2 + unp.sqrt(1/4 * fac(ang, sV)**4 - unp.sin(ang)**2 * fac(ang, sV)**2))
def n_s(ang, V): # Wahl von n_s_2 fuer Unsicherheiten
	ang, sV = ang * np.pi / 180, unp.sqrt(V)
	return unp.sqrt(1 + sV * ((2 * unp.cos(ang)) / (1 - sV))**2)

# Intensitaetsverhaeltnis spezifizieren und Ergebnisse darstellen

V = 0.0225

with warnings.catch_warnings():
	warnings.simplefilter('ignore')

	# Parallele Polarisation

	xx, yy = np.arange(0, 85, 0.003), np.arange(0.225, 5.2, 0.003)
	XX, YY = np.meshgrid(xx, yy)
	plt.contour(XX, YY, imp_p(XX, YY), [-np.sqrt(V), np.sqrt(V)], linestyles='solid', linewidths=7.5, alpha=0.15, colors='#ff7f0e')

	plt.plot(100, 100, linewidth=5, alpha=0.15, c='#ff7f0e', label='Konturen')

	xx = np.linspace(12.25, 85.25, 10000)
	plt.plot(xx, n_p_1(xx, V), c='#d62728', label='Lösung 3a')
	xx = np.linspace(12.25, 85.25, 500000)
	plt.plot(xx, n_p_2(xx, V), c='#d62728', alpha=0.5, label='Lösung 3b')

	xx = np.linspace(-0.5, 75.4, 10000)
	plt.plot(xx, n_p_3(xx, V), c='#ff7f0e', label='Lösung 4a')
	xx = np.linspace(-0.5, 82.075, 500000)
	plt.plot(xx, n_p_4(xx, V), c='#ff7f0e', alpha=0.5, label='Lösung 4b')

	plt.xlim(-3.5, 87.5)
	plt.ylim(-0.275, 5.5)
	plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
	plt.ylabel(r'$n$')
	leg = plt.legend(handlelength=1.5, borderpad=0.75, loc='best', edgecolor='k', facecolor='none')
	leg.get_frame().set_linewidth(0.25)
	plt.savefig('build/plot_ip.pdf')
	plt.close()

	# Senkrechte Polarisation

	xx, yy = np.arange(-6, 86, 0.01), np.arange(0.5, 1.7, 0.01)
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
plt.errorbar(ang_s, noms(I_s/I_g), yerr=stds(I_s/I_g), fmt='.', ms=4, color='steelblue', capsize=1.5, label='Messdaten', zorder=10)
plt.plot(xx, amp_s(xx, *par_s), c='olivedrab', alpha=0.25, zorder=100)
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$I / I_0$')
plt.xticks(np.arange(5, 95, 10))
leg = plt.legend(borderpad=0.75, loc='best', edgecolor='k', facecolor='none')
leg.get_frame().set_linewidth(0.25)
plt.savefig('build/plot_s.pdf')
plt.close()

plt.plot(xx, amp_p(xx, *par_p), label='Regression', c='olivedrab', zorder=1)
plt.errorbar(ang_p[:31], noms(I_p[:31]/I_g), yerr=stds(I_p[:31]/I_g), fmt='.', ms=4, color='steelblue', capsize=1.5, label='Messdaten', zorder=10)
plt.plot(ang_p[31:], noms(I_p[31:]/I_g), '.', c='steelblue', ms=4, zorder=10)
plt.plot(xx, amp_p(xx, *par_p), c='olivedrab', alpha=0.25, zorder=100)
plt.ylim(-0.01, 0.17)
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$I / I_0$')
plt.xticks(np.arange(5, 95, 10))
plt.yticks(np.arange(0, 0.18, 0.04))
leg = plt.legend(borderpad=0.75, loc='best', edgecolor='k', facecolor='none')
leg.get_frame().set_linewidth(0.25)
plt.savefig('build/plot_p.pdf')
plt.close()

# Tabellen schreiben

table_footer = r''' 		\bottomrule
	\end{tabular}
'''
table_header = r'''
 	\begin{tabular}
		{S[table-format=2.0]
		 S[table-format=2.0]
		 @{${}\pm{}$}
		 S[table-format=1.0]
		 S[table-format=1.3]
		 @{${}\pm{}$}
		 S[table-format=1.3]
		 S[table-format=1.3]
		 @{${}\pm{}$}
		 S[table-format=1.3]
		 S[table-format=2.0]
		 S[table-format=3.0]
		 @{${}\pm{}$}
		 S[table-format=2.0]
		 S[table-format=1.3]
		 @{${}\pm{}$}
		 S[table-format=1.3]
		 S[table-format=1.3]
		 @{${}\pm{}$}
		 S[table-format=1.3]}
		\toprule
		{$\alpha \mathbin{/} \unit{\degree}$} &
		\multicolumn{2}{c}{$I \mathbin{/} \unit{\micro\ampere}$} &
		\multicolumn{2}{c}{$I / I_0$} &
		\multicolumn{2}{c}{$n$} &
		{$\alpha \mathbin{/} \unit{\degree}$} &
		\multicolumn{2}{c}{$I \mathbin{/} \unit{\micro\ampere}$} &
		\multicolumn{2}{c}{$I / I_0$} &
		\multicolumn{2}{c}{$n$} \\
		\midrule
'''
row_template = r'		{0:2.0f} & {1:3.0f} & {2:2.0f} & {3:1.3f} & {4:1.3f} & {5:1.3f} & {6:1.3f} & {7:2.0f} & {8:3.0f} & {9:2.0f} & {10:1.3f} & {11:1.3f} & {12:1.3f} & {13:1.3f} \\'
row_template_ = r'		\multicolumn{{7}}{{c}}{{$ $}} & {0:2.0f} & {1:3.0f} & {2:2.0f} & {3:1.3f} & {4:1.3f} & {5:1.3f} & {6:1.3f} \\'
with open('build/table_s.tex', 'w') as f:
	f.write(table_header)
	for row in zip(ang_s[:19], noms(I_s[:19]), stds(I_s[:19]), noms(I_s[:19] / I_g), stds(I_s[:19] / I_g),
				   noms(n_s(ang_s[:19], I_s[:19] / I_g)), stds(n_s(ang_s[:19], I_s[:19] / I_g)),
				   ang_s[19:], noms(I_s[19:]), stds(I_s[19:]), noms(I_s[19:] / I_g), stds(I_s[19:] / I_g),
				   noms(n_s(ang_s[19:], I_s[19:] / I_g)), stds(n_s(ang_s[19:], I_s[19:] / I_g))):
		f.write(row_template.format(*row))
		f.write('\n')
	for row in zip(ang_s[38:41], noms(I_s[38:41]), stds(I_s[38:41]), noms(I_s[38:41] / I_g), stds(I_s[38:41] / I_g),
				   noms(n_s(ang_s[38:41], I_s[38:41] / I_g)), stds(n_s(ang_s[38:41], I_s[38:41] / I_g))):
		f.write(row_template_.format(*row))
		f.write('\n')
	f.write(table_footer)
table_header = r'''
 	\begin{tabular}
		{S[table-format=2.0]
		 S[table-format=2.1]
		 @{${}\pm{}$}
		 S[table-format=1.1]
		 S[table-format=1.4]
		 @{${}\pm{}$}
		 S[table-format=1.4]
		 S[table-format=1.3]
		 @{${}\pm{}$}
		 S[table-format=1.3]
		 S[table-format=2.0]
		 S[table-format=2.1]
		 @{${}\pm{}$}
		 S[table-format=1.2]
		 S[table-format=1.4]
		 @{${}\pm{}$}
		 S[table-format=1.4]
		 S[table-format=2.3]
		 @{${}\pm{}$}
		 S[table-format=1.3]}
		\toprule
		{$\alpha \mathbin{/} \unit{\degree}$} &
		\multicolumn{2}{c}{$I \mathbin{/} \unit{\micro\ampere}$} &
		\multicolumn{2}{c}{$I / I_0$} &
		\multicolumn{2}{c}{$n$} &
		{$\alpha \mathbin{/} \unit{\degree}$} &
		\multicolumn{2}{c}{$I \mathbin{/} \unit{\micro\ampere}$} &
		\multicolumn{2}{c}{$I / I_0$} &
		\multicolumn{2}{c}{$n$} \\
		\midrule
'''
row_template = r'		{0:2.0f} & {1:2.1f} & {2:1.1f} & {3:1.4f} & {4:1.4f} & {5:1.3f} & {6:1.3f} & {7:2.0f} & {8:2.1f} & {9:} & {10:1.4f} & {11:1.4f} & {12:1.3f} & {13:1.3f} \\'
row_template_ = r'		{0:2.0f} & {1:2.1f} & {2:1.1f} & {3:1.4f} & {4:1.4f} & {5:1.3f} & {6:1.3f} & \multicolumn{{7}}{{c}}{{$ $}} \\'
with open('build/table_p.tex', 'w') as f:
	f.write(table_header)
	for row in zip(ang_p[:20], noms(I_p[:20]), 0.02*scale_p[:20], noms(I_p[:20] / I_g), stds(I_p[:20] / I_g),
				   noms(n_p(ang_p[:20], I_p[:20] / I_g)), stds(n_p(ang_p[:20], I_p[:20] / I_g)),
				   ang_p[21:], noms(I_p[21:]), 0.02*scale_p[21:], noms(I_p[21:] / I_g), stds(I_p[21:] / I_g),
				   noms(n_p(ang_p[21:], I_p[21:] / I_g)), stds(n_p(ang_p[21:], I_p[21:] / I_g))):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(row_template_.format(*[ang_p[20], noms(I_p[20]), 0.02*scale_p[20], noms(I_p[20] / I_g), stds(I_p[20] / I_g),
				   noms(n_p(ang_p[20], I_p[20] / I_g)), stds(n_p(ang_p[20], I_p[20] / I_g))]))
	f.write('\n')
	f.write(table_footer)


# Varianz-Gewichtete Mittelwerte

n, dn = 0, 0
for m in n_s(ang_s, I_s / I_g):
	n += m / m.s**2
	dn += 1 / m.s**2
nn_s = n/dn
with open('build/nn_s.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{nn_s.n:1.3f}({nn_s.s:1.3f})')
	f.write(r'}')
n, dn = 0, 0
for m in n_p(ang_p, I_p / I_g):
	n += m / m.s**2
	dn += 1 / m.s**2
nn_p = n/dn
with open('build/nn_p.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{nn_p.n:1.3f}({nn_p.s:1.3f})')
	f.write(r'}')

nnn_s = ufloat(par_s[0], err_s[0])
with open('build/nnn_s.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{nnn_s.n:1.3f}({nnn_s.s:1.3f})')
	f.write(r'}')
nnn_p = ufloat(par_p[0], err_p[0])
with open('build/nnn_p.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{nnn_p.n:1.3f}({nnn_p.s:1.3f})')
	f.write(r'}')

with open('build/s_s.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_s[1]:1.3f}({err_s[1]:1.3f})')
	f.write(r'}')
with open('build/s_p.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_p[1]:1.3f}({err_p[1]:1.3f})')
	f.write(r'}')

f_nnn_s = ufloat(f_par_s[0], f_err_s[0])
with open('build/f_nnn_s.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{f_nnn_s.n:1.3f}({f_nnn_s.s:1.3f})')
	f.write(r'}')
f_nnn_p = ufloat(f_par_p[0], f_err_p[0])
with open('build/f_nnn_p.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{f_nnn_p.n:1.3f}({f_nnn_p.s:1.3f})')
	f.write(r'}')

n = (nn_s / nn_s.s**2) + (nn_p / nn_p.s**2) + (nnn_s / nnn_s.s**2) + (nnn_p / nnn_p.s**2) + (f_nnn_s / f_nnn_s.s**2) + (f_nnn_p / f_nnn_p.s**2)
dn = (1 / nn_s.s**2) + (1 / nn_p.s**2) + (1 / nnn_s.s**2) + (1 / nnn_p.s**2) + (1 / f_nnn_s.s**2) + (1 / f_nnn_p.s**2)

nn = n / dn
with open('build/nn.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{nn.n:1.3f}({nn.s:1.3f})')
	f.write(r'}')


# Vergleichsplot
xx = np.linspace(4, 88, 10000)

plt.plot(xx, amp_s(xx, nnn_s.n, par_s[1]), c='olivedrab', label='Skalierter Fit')
plt.plot(xx, amp_s(xx, f_nnn_s.n, 1), c='steelblue', label='Einfacher Fit')
plt.plot(xx, amp_s(xx, nn_s.n, 1), c='indianred', label='Theoriekurve')
plt.plot(xx, amp_s(xx, nnn_s.n, par_s[1]), c='olivedrab', alpha=0.25)
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$I / I_0$')
leg = plt.legend(borderpad=0.75, loc='best', edgecolor='k', facecolor='none')
leg.get_frame().set_linewidth(0.25)
plt.savefig('build/plot_comp_s')
plt.close()

plt.plot(xx, amp_p(xx, nnn_p.n, par_p[1]), c='olivedrab', label='Skalierter Fit')
plt.plot(xx, amp_p(xx, f_nnn_p.n, 1), c='steelblue', label='Einfacher Fit')
plt.plot(xx, amp_p(xx, nn_p.n, 1), c='indianred', label='Theoriekurve')
plt.plot(xx, amp_p(xx, f_nnn_p.n, 1), c='steelblue', alpha=0.25)
plt.plot(xx, amp_p(xx, nnn_p.n, par_p[1]), c='olivedrab', alpha=0.25)
plt.xlabel(r'$\alpha \mathbin{/} \unit{\degree}$')
plt.ylabel(r'$I / I_0$')
leg = plt.legend(borderpad=0.75, loc='best', edgecolor='k', facecolor='none')
leg.get_frame().set_linewidth(0.25)
plt.savefig('build/plot_comp_p')
plt.close()


# Parameter schreiben 

with open('build/lam.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{lam.n:.0f}({lam.s:.0f})')
	f.write(r'}{\nano\meter}')
with open('build/I_g.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{I_g.n:.0f}({I_g.s:.0f})')
	f.write(r'}{\micro\ampere}')
with open('build/I_d.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{1000*I_d.n:.1f}({1000*I_d.s:.1f})')
	f.write(r'}{\nano\ampere}')
with open('build/ang_min.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{ang_min:.0f}')
	f.write(r'}{\degree}')
with open('build/I_min.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{1000*I_min.n:.0f}({1000*I_min.s:.0f})')
	f.write(r'}{\nano\ampere}')


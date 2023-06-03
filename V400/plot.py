import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
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

# Sinus und Cosinus definieren
def sin(w):
	w = np.pi * w / 180
	return unp.sin(w)
def cos(w):
	w = np.pi * w / 180
	return unp.cos(w)
def arcsin(v):
	return 180 * unp.arcsin(v) / np.pi
def arccos(v):
	return 180 * unp.arccos(v) / np.pi

# Gewichteten Mittelwert definieren
def weigh(x):
	return sum(noms(x) / stds(x)) / sum(1 / stds(x))

# Gewichtete Standardabweichung definieren
def dev(x):
	return np.sqrt(1 / sum(1 / stds(x)))
#	return np.sqrt(abs(weigh(x**2) - weigh(x)**2))


# Statistik definieren
def stat(x):
	return uar(weigh(x), dev(x))

# Format von ufloats/uarrays
def r(s, p):
	if p == 0:
		return np.char.add(np.char.add(noms(s).round(p).astype(int).astype(str), '+-'), stds(s).round(p).astype(int).astype(str))
	return np.char.add(np.char.add(noms(s).round(p).astype(str), '+-'), stds(s).round(p).astype(str))	


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
plt.savefig('build/fig_reflexion.pdf')
plt.close()

# Tabelle
table_header = r'''	\sisetup{table-parse-only}
	\begin{tabular}{S S S}
		\toprule
		{$\alpha \mathbin{/} \unit{\degree}$} &
		{$\beta \mathbin{/} \unit{\degree}$} &
		{$\beta / \alpha$} \\
		\midrule
'''
table_footer = r'''		\bottomrule
	\end{tabular}
'''
row_template = r'		{0:} & {1:} & {2:} \\'
with open('build/tab_reflexion.tex', 'w') as f:
	f.write(table_header)
	for row in zip(r(a_ref, 0), r(b_ref, 0), r(b_ref / a_ref, 3)):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)

# Ergebnisse
with open('build/fit_ref_0.tex', 'w') as f:
	f.write(f'\\num{{{r(fit[0], 3)}}}')
with open('build/fit_ref_1.tex', 'w') as f:
	f.write(f'\\num{{{r(fit[1], 3)}}}')
with open('build/stat_ref.tex', 'w') as f:
	f.write(f'\\num{{{r(stat(b_ref / a_ref), 3)}}}')


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
plt.savefig('build/fig_brechung.pdf')
plt.close()

# Tabelle
table_header = r'''	\sisetup{table-parse-only}
	\begin{tabular}{S S S}
		\toprule
		{$\alpha \mathbin{/} \unit{\degree}$} &
		{$\beta \mathbin{/} \unit{\degree}$} &
		{$n$} \\
		\midrule
'''
table_footer = r'''		\bottomrule
	\end{tabular}
'''
row_template = r'		{0:} & {1:} & {2:} \\'
with open('build/tab_brechung.tex', 'w') as f:
	f.write(table_header)
	for row in zip(r(a_bre, 0), r(b_bre, 1), r(sin(a_bre) / sin(b_bre), 3)):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)

# Ergebnisse
with open('build/n_bre.tex', 'w') as f:
	f.write(f'\\num{{{r(1 / fit[0], 3)}}}')
with open('build/fit_bre_0.tex', 'w') as f:
	f.write(f'\\num{{{r(fit[0], 3)}}}')
with open('build/fit_bre_1.tex', 'w') as f:
	f.write(f'\\num{{{r(fit[1], 3)}}}')
with open('build/stat_bre.tex', 'w') as f:
	f.write(f'\\num{{{r(stat(sin(a_bre) / sin(b_bre)), 3)}}}')
with open('build/c.tex', 'w') as f:
	f.write(f'\\qty{{{1e-8 * const.c:.3f}e8}}{{\\meter\\per\\second}}')
with open('build/c_n.tex', 'w') as f:
	f.write(f'\\qty{{{r(1e-8 * const.c / (1 / fit[0]), 3)}e8}}{{\\meter\\per\\second}}')
with open('build/c_stat.tex', 'w') as f:
	f.write(f'\\qty{{{r(1e-8 * const.c / stat(sin(a_bre) / sin(b_bre)), 3)}e8}}{{\\meter\\per\\second}}')


### Planparallele Platten:

# Verwendete Konstanten
d = 5.85
with open('build/d_plan.tex', 'w') as f:
	f.write(f'\\qty{{{d:.2f}}}{{\\centi\\meter}}')
n = stat(uar([noms(1 / fit[0]), noms(stat(sin(a_bre) / sin(b_bre)))], [stds(1 / fit[0]), stds(stat(sin(a_bre) / sin(b_bre)))]))
with open('build/n_plan.tex', 'w') as f:
	f.write(f'\\num{{{r(n, 3)}}}')


# Formeln zum Strahlversatz
def s(d, a, b):
	return d * sin(a - b) / cos(b)
def b(a, n):
	return arcsin(sin(a) / n)

# Tabelle
table_header = r'''	\sisetup{table-parse-only}
	\begin{tabular}{S S S S S}
		\toprule
		{$\alpha \mathbin{/} \unit{\degree}$} &
		{$\beta \mathbin{/} \unit{\degree}$} &
		{$\hat{\beta} \mathbin{/} \unit{\degree}$} &
		{$s \mathbin{/} \unit{\meter}$} &
		{$\hat{s} \mathbin{/} \unit{\meter}$} \\
		\midrule
'''
table_footer = r'''		\bottomrule
	\end{tabular}
'''
row_template = r'		{0:} & {1:} & {2:} & {3:} & {4:} \\'
with open('build/tab_plan.tex', 'w') as f:
	f.write(table_header)
	for row in zip(r(a_bre, 0), r(b_bre, 1), r(b(a_bre, n), 1), r(s(d, a_bre, b_bre), 2), r(s(d, a_bre, b(a_bre, n)), 2)):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)


### Prisma:

# Verwendete Konstante
n = 1.51
with open('build/n_pris.tex', 'w') as f:
	f.write(f'\\num{{{n:.3f}}}')

# Formel zur Ablenkung
def d(a_1, a_2, b_1, b_2):
	return (a_1 + a_2) - (b_1 + b_2)

# Tabelle
table_header = r'''	\sisetup{table-parse-only}
	\begin{tabular}{S S S S S S S S S S}
		\toprule
		& &
		\multicolumn{4}{c}{Grünes Licht} &
		\multicolumn{4}{c}{Rotes Licht} \\
		\cmidrule(lr){3-6}\cmidrule(lr){7-10}
		{$\alpha_1 \mathbin{/} \unit{\degree}$} &
		{$\beta_1 \mathbin{/} \unit{\degree}$} &
		{$\alpha_2 \mathbin{/} \unit{\degree}$} &
		{$\beta_2 \mathbin{/} \unit{\degree}$} &
		{$\beta_1 + \beta_2 \mathbin{/} \unit{\degree}$} &
		{$\delta_G \mathbin{/} \unit{\degree}$} &
		{$\alpha_2 \mathbin{/} \unit{\degree}$} &
		{$\beta_2 \mathbin{/} \unit{\degree}$} &
		{$\beta_1 + \beta_2 \mathbin{/} \unit{\degree}$} &
		{$\delta_R \mathbin{/} \unit{\degree}$} \\
		\midrule
'''
table_footer = r'''		\bottomrule
	\end{tabular}
'''
row_template = r'		{0:} & {1:} & {2:} & {3:} & {4:} & {5:} & {6:} & {7:} & {8:} & {9:} \\'
with open('build/tab_pris.tex', 'w') as f:
	f.write(table_header)
	for row in zip(r(a_dis, 0), r(b(a_dis, n), 1), r(g_dis, 0), r(b(g_dis, n), 1), r(b(a_dis, n) + b(g_dis, n), 1), r(d(a_dis, g_dis, b(a_dis, n), b(g_dis, n)), 1), r(r_dis, 0), r(b(r_dis, n), 1), r(b(a_dis, n) + b(r_dis, n), 1), r(d(a_dis, r_dis, b(a_dis, n), b(r_dis, n)), 1)):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)

# Ergebnisse
with open('build/d_pris_g.tex', 'w') as f:
	f.write(f'\\qty{{{r(stat(d(a_dis, g_dis, b(a_dis, n), b(g_dis, n))), 1)}}}{{\\degree}}')
with open('build/d_pris_r.tex', 'w') as f:
	f.write(f'\\qty{{{r(stat(d(a_dis, r_dis, b(a_dis, n), b(r_dis, n))), 1)}}}{{\\degree}}')


### Beugung am Gitter:

# Gitterkostanten
d = 1e3 / np.array([600, 600, 300, 300, 300, 100, 100, 100, 100, 100, 100, 100])

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
x = np.array([-0.25, 6.25])
plt.plot(x - 0.1, lin(x, *par_g), c='#65ff00', zorder=-100)
plt.errorbar(k_beu - 0.1, noms(d * sin(g_beu)), yerr=stds(d * sin(g_beu)), fmt='.', c='#47b300', ms=3.3, zorder=-10)
plt.plot(x + 0.1, lin(x, *par_r), c='#ff3900', zorder=10)
plt.errorbar(k_beu + 0.1, noms(d * sin(r_beu)), yerr=stds(d * sin(r_beu)), fmt='.', c='#b32700', ms=3.3, zorder=100)
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([100,101], [100,101], c='gray', label='Regression')
plt.errorbar([100,101], [100,101], yerr=[1,1], fmt='.', c='gray', ms=3.3, label='Messdaten')
plt.xlabel(r'$k$')
plt.ylabel(r'$d\sin(\varphi) \mathbin{/} \unit{\micro\meter}$')
plt.legend()
plt.savefig('build/fig_beugung.pdf')
plt.close()

print(k_beu)

# Formel zur Beugung
def l(d, p, k):
	return d * sin(p) / k

# Tabelle
table_header = r'''	\sisetup{table-parse-only}
	\begin{tabular}{S S S S S S S S S S}
		\toprule
		& &
		\multicolumn{4}{c}{Grünes Licht} &
		\multicolumn{4}{c}{Rotes Licht} \\
		\cmidrule(lr){3-6}\cmidrule(lr){7-10}
		{$d \mathbin{/} \unit{\micro\meter}$} & {$k$} &
		\multicolumn{2}{c}{$\varphi \mathbin{/} \unit{\degree}$} &
		\multicolumn{2}{c}{$\lambda \mathbin{/} \unit{\nano\meter}$} &
		\multicolumn{2}{c}{$\varphi \mathbin{/} \unit{\degree}$} &
		\multicolumn{2}{c}{$\lambda \mathbin{/} \unit{\nano\meter}$} \\
		\midrule
'''
table_footer = r'''		\bottomrule
	\end{tabular}
'''
row_template = r'		{0:} & {1:} & {2:} & {3:} & {4:} & {5:} & {6:} & {7:} & {8:} & {9:} \\'
with open('build/tab_beugung.tex', 'w') as f:
	f.write(table_header)
	for row in zip(d, k_beu, r(g_beu_1, 0), r(g_beu_2, 0), l(d, g_beu_1, k_beu), l(d, g_beu_2, k_beu), r(r_beu_1, 0), r(r_beu_2, 0), l(d, r_beu_1, k_beu), l(d, r_beu_2, k_beu)):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)

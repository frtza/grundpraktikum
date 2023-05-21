import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from scipy.constants import convert_temperature
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds
from uncertainties import ufloat

# Bilder einlesen

im1 = plt.imread('content/messung/auswertung/1.png')
im2 = plt.imread('content/messung/auswertung/2.png')
im3a = plt.imread('content/messung/auswertung/3a.png')
im3b = plt.imread('content/messung/auswertung/3b.png')
im4a = plt.imread('content/messung/auswertung/4a.png')
im4b = plt.imread('content/messung/auswertung/4b.png')
im5 = plt.imread('content/messung/auswertung/5.png')
im6 = plt.imread('content/messung/auswertung/6.png')

# Kastenzahl pro Abschnitt

n1 = unp.uarray([22, 23, 24, 23, 24, 22, 26, 24, 21, 24], 1)
n2 = unp.uarray([22, 23, 23, 22, 24, 23, 24, 23, 23, 20], 1)
n3a = unp.uarray([22, 22, 21, 21, 20, 21, 19, 23, 21, 21, 27], 1)
n3b = unp.uarray([31, 31, 28, 30, 31, 29, 28, 29], 1)
n4a = unp.uarray([22, 24, 17, 21, 22, 21, 22, 19, 21, 21, 21], 1)
n4b = unp.uarray([27, 32, 27, 31, 28, 30, 29, 32], 1)
n5 = unp.uarray([24, 19, 20, 21, 20, 18, 22, 21, 21, 20, 26], 1)
n6 = unp.uarray([20, 23, 19, 20, 22, 20, 21, 20, 20, 22, 21], 1)

# Statistik der Intervalle

def mittel(uarr):
	n = len(uarr)
	val = 0.0
	i = 0
	while i < n:
		val += uarr[i]
		i += 1
	return val / n

def fehler(uarr):
	n = len(uarr)
	return unp.sqrt(mittel(uarr**2) - mittel(uarr)**2)

def stat(uarr):
	return ufloat(noms(mittel(uarr)), np.sqrt(stds(mittel(uarr))**2 + noms(fehler(uarr))**2))

stat1 = stat(n1)
stat2 = stat(n2)
stat3a = stat(n3a)
stat3b = stat(n3b)
stat4a = stat(n4a)
stat4b = stat(n4b)
stat5 = stat(n5)
stat6 = stat(n6)

# Spannung pro Kasten in Volt

v1 = 1 / n1
v2 = 1 / n2
v3a = 5 / n3a
v3b = 5 / n3b
v4a = 5 / n4a
v4b = 5 / n4b
v5 = 5 / n5
v6 = 5 / n6

# Kasten vertikal absolut zu Null und Kasten horizontal relativ zum Abschnittsanfang mit passendem Abschnitt ablesen

ab1 = np.array([0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9])
x1 = [0, 0, 0, 0, 0, 0, 0, 12, 0, 11, 0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 2, 4, 10]
x1 = unp.uarray(x1, stds(stat1))
y1 = [158, 158, 157, 156, 153, 151, 146, 143, 138, 131, 120, 117, 114, 110, 106, 100, 89, 80, 71, 56, 44, 30, 22, 15, 10, 6, 3, 1, 0]
y1 = unp.uarray(y1, 1)

ab2 = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 9])
x2 = [0, 8, 18, 6, 16, 3, 13, 1, 11, 16, 21, 3, 8, 18, 4, 14, 0, 10, 20, 6, 16, 3, 13, 0, 10, 15]
x2 = unp.uarray(x2, stds(stat2))
y2 = [128, 117, 104, 90, 76, 61, 48, 34, 21, 17, 13, 10, 8, 8, 7, 6, 6, 5, 4, 3, 2, 1, 0, -1, -2, -2]
y2 = unp.uarray(y2, 1)

ab3a = np.array([0, 1, 2, 3, 4])
x3a = [21, 19, 19, 19, 20]
x3a = unp.uarray(x3a, stds(stat3a))
y3a = [13, 22, 29, 34, 40]
y3a = unp.uarray(y3a, 1)

ab3b = np.array([1, 2, 3, 4, 5])
x3b = [4, 1, 2, 2, 3]
x3b = unp.uarray(x3b, stds(stat3b))
y3b = [36, 56, 71, 85, 99]
y3b = unp.uarray(y3b, 1)

ab4a = np.array([1, 1, 3, 4, 5])
x4a = [1, 22, 1, 1, 3]
x4a = unp.uarray(x4a, stds(stat4a))
y4a = [15, 24, 31, 36, 42]
y4a = unp.uarray(y4a, 1)

ab4b = np.array([1, 2, 3, 4, 5])
x4b = [4, 1, 3, 3, 7]
x4b = unp.uarray(x4b, stds(stat4b))
y4b = [26, 41, 52, 62, 70]
y4b = unp.uarray(y4b, 1)

ab5 = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9])
x5 = [23, 18, 19, 18, 18, 1, 0, 1, 5]
x5 = unp.uarray(x5, stds(stat5))
y5 = [31, 51, 66, 76, 80, 79, 77, 79, 85]
y5 = unp.uarray(y5, 1)

ab6 = np.array([1, 1, 3, 4, 4, 6, 7])
x6 = [2, 21, 0, 0, 21, 0, 2]
x6 = unp.uarray(x6, stds(stat6))
y6 = [25, 41, 53, 61, 68, 71, 77]
y6 = unp.uarray(y6, 1)

# Plots ausgeben

t1 = 1 * ab1 + x1 * v1[ab1]
dt1 = t1[1:] - t1[:-1]
dy1 = y1[1:] - y1[:-1]
r1 = dy1 / dt1

plt.figure(figsize=(5.78, 3.87))
plt.imshow(im1, extent=np.array([-96, 1200-96, -23, 864-23]) * 0.00911)
plt.plot([0,5], [0,0], 'ob', mfc='none')
plt.errorbar(noms(t1), noms(y1 / 22.99), xerr=stds(t1), yerr=stds(y1 / 22.99), fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 10, 1))
plt.yticks([])
plt.xlabel(r'$U_{\hspace{-0.2ex} A} \mathbin{/} \unit{\volt}$')
plt.savefig('build/plot1.pdf', dpi=100)
plt.close()

def betrag(x, a, b, c):
	return np.abs(x - a) * b + c

par, cov = curve_fit(betrag, noms(t1[-14:-4] + dt1[-13:-3] / 2), -noms(r1[-13:-3]), p0=(8.7, -917, 300), sigma=stds(r1[-13:-3]))
err = np.sqrt(np.diag(cov))

a = ufloat(par[0], err[0])
with open('build/a.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{a.n:.3f}({a.s:.3f})')
	f.write(r'}{\volt}')
aa = np.array([a.n - 1/3, a.n, a.n + 1/3])

plt.plot(aa, betrag(aa, *par), 'r', lw=1.2)
plt.errorbar(noms(t1[:-1] + dt1 / 2), -noms(r1), xerr=stds(t1[:-1] + dt1 / 2), yerr=stds(r1) / 40, fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 10, 1))
plt.yticks([])
plt.xlabel(r'$U_{\hspace{-0.2ex} A} \mathbin{/} \unit{\volt}$')
plt.savefig('build/plot11.pdf')
plt.close()

t2 = 1 * ab2 + x2 * v2[ab2]
dt2 = t2[1:] - t2[:-1]
dy2 = y2[1:] - y2[:-1]
r2 = dy2 / dt2

plt.figure(figsize=(5.78, 3.87))
plt.imshow(im2, extent=np.array([-66, 1200-66, -17, 864-17]) * 0.00925)
plt.plot([0,5], [0,0], 'ob', mfc='none')
plt.errorbar(noms(t2), noms(y2 / 22.99), xerr=stds(t2), yerr=stds(y2 / 22.99), fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 11, 1))
plt.yticks([])
plt.xlabel(r'$U_{\hspace{-0.2ex} A} \mathbin{/} \unit{\volt}$')
plt.savefig('build/plot2.pdf', dpi=100)
plt.close()

plt.plot(noms(np.array([t2[0] + dt2[0] / 2, t2[7] + dt2[7] / 2, t2[-14] + dt2[-13] / 2, t2[-2] + dt2[-1] / 2])),
		[-noms(stat(r2[:8])), -noms(stat(r2[:8])), -noms(stat(r2[-13:])), -noms(stat(r2[-13:]))], 'r', lw=1.2)
plt.errorbar(noms(t2[:-1] + dt2 / 2), -noms(r2), xerr=stds(t2[:-1] + dt2 / 2), yerr=stds(r2) / 4, fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 11, 1))
plt.yticks([])
plt.xlabel(r'$U_{\hspace{-0.2ex} A} \mathbin{/} \unit{\volt}$')
plt.savefig('build/plot22.pdf')
plt.close()

s = t2[7] + dt2[7] / 2
with open('build/s.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{s.n:.3f}({s.s:.3f})')
	f.write(r'}{\volt}')

t3a = 5 * ab3a + x3a * v3a[ab3a]
plt.figure(figsize=(5.78, 3.87))
plt.imshow(im3a, extent=np.array([-70, 1200-70, -23, 864-23]) * 0.05)
plt.plot([0,30], [0,0], 'ob', mfc='none')
plt.errorbar(noms(t3a), noms(y3a / 4.16), xerr=stds(t3a), yerr=stds(y3a / 4.16), fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 60, 5))
plt.yticks([])
plt.xlabel(r'$U_{\hspace{-0.2ex} B} \mathbin{/} \unit{\volt}$')
plt.savefig('build/plot3a.pdf', dpi=100)
plt.close()

t3b = 5 * ab3b + x3b * v3b[ab3b]
plt.figure(figsize=(5.78, 3.87))
plt.imshow(im3b, extent=np.array([-57, 1200-57, -26, 864-26]) * 0.0355)
plt.plot([0,20], [0,0], 'ob', mfc='none')
plt.errorbar(noms(t3b), noms(y3b / 6), xerr=stds(t3b), yerr=stds(y3b / 6), fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 45, 5))
plt.yticks([])
plt.xlabel(r'$U_{\hspace{-0.2ex} B} \mathbin{/} \unit{\volt}$')
plt.savefig('build/plot3b.pdf', dpi=100)
plt.close()

t4a = 5 * ab4a + x4a * v4a[ab4a]
plt.figure(figsize=(5.78, 3.87))
plt.imshow(im4a, extent=np.array([-74, 1200-74, -23, 864-23]) * 0.0501)
plt.plot([0,30], [0,0], 'ob', mfc='none')
plt.errorbar(noms(t4a), noms(y4a / 4.2), xerr=stds(t4a), yerr=stds(y4a / 4.2), fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 60, 5))
plt.yticks([])
plt.xlabel(r'$U_{\hspace{-0.2ex} B} \mathbin{/} \unit{\volt}$')
plt.savefig('build/plot4a.pdf', dpi=100)
plt.close()

t4b = 5 * ab4b + x4b * v4b[ab4b]
plt.figure(figsize=(5.78, 3.87))
plt.imshow(im4b, extent=np.array([-85, 1200-85, -25, 864-25]) * 0.0363)
plt.plot([0,20], [0,0], 'ob', mfc='none')
plt.errorbar(noms(t4b), noms(y4b / 5.75), xerr=stds(t4b), yerr=stds(y4b / 5.75), fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 45, 5))
plt.yticks([])
plt.xlabel(r'$U_{\hspace{-0.2ex} B} \mathbin{/} \unit{\volt}$')
plt.savefig('build/plot4b.pdf', dpi=100)
plt.close()

t5 = 5 * ab5 + x5 * v5[ab5]
plt.figure(figsize=(5.78, 3.87))
plt.imshow(im5, extent=np.array([-69, 1200-69, -24, 864-24]) * 0.051)
plt.plot([0,25], [0,0], 'ob', mfc='none')
plt.errorbar(noms(t5), noms(y5 / 4.12), xerr=stds(t5), yerr=stds(y5 / 4.12), fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 60, 5))
plt.yticks([])
plt.xlabel(r'$U_{\hspace{-0.2ex} B} \mathbin{/} \unit{\volt}$')
plt.savefig('build/plot5.pdf', dpi=100)
plt.close()

t6 = 5 * ab6 + x6 * v6[ab6]
plt.figure(figsize=(5.78, 3.87))
plt.imshow(im6, extent=np.array([-72, 1200-72, -36, 864-36]) * 0.0512)
plt.plot([0,25], [0,0], 'ob', mfc='none')
plt.errorbar(noms(t6), noms(y6 / 4.08), xerr=stds(t6), yerr=stds(y6 / 4.08), fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 60, 5))
plt.yticks([])
plt.xlabel(r'$U_{\hspace{-0.2ex} B} \mathbin{/} \unit{\volt}$')
plt.savefig('build/plot6.pdf', dpi=100)
plt.close()

# Tabellen ausgeben

def p(T):
	return 5.5e7 * unp.exp(-6876 / T)

def w(T):
	return 0.01 * 0.0029 / p(T)

a = 0.01

mess = np.array([r'$1$', r'$2$', r'$3 \:\: 4$', r'$5 \:\: 6$'])
T = unp.uarray([24.3, 145, 160, 180], [0, 5, 5, 5])
T = unp.uarray(convert_temperature(noms(T), 'C', 'K'), stds(T))
ps = p(T)
ws = w(T)
rels = a / ws

table_footer = r'''		\bottomrule
	\end{tabular}
'''
table_header = r'''	\sisetup{table-parse-only, retain-zero-uncertainty=true}
	\begin{tabular}{c S S S S[retain-zero-exponent=true]}
		\toprule
		{Messung} &
		{$T \mathbin{/} \unit{\kelvin}$} &
		{$p \mathbin{/} \unit{\bar}$} &
		{$\bar{w} \mathbin{/} \unit{\meter}$} &
		{$a / \bar{w}$} \\
		\midrule
'''
with open('build/table_1.tex', 'w') as f:
	f.write(table_header)
	f.write(f'		{mess[0]:} & {T[0].n:.2f}({T[0].s:.2f}) & {1e3*ps[0].n:.2f}({1e3*ps[0].s:.2f})e-6 & {1e3*ws[0].n:.2f}({1e3*ws[0].s:.2f})e-3 & {rels[0].n:.2f}({rels[0].s:.2f})e0')
	f.write(r' \\')
	f.write('\n')
	f.write(f'		{mess[1]:} & {T[1].n:.2f}({T[1].s:.2f}) & {ps[1].n:.2f}({ps[1].s:.2f})e-3 & {1e6*ws[1].n:.2f}({1e6*ws[1].s:.2f})e-6 & {1e-3*rels[1].n:.2f}({1e-3*rels[1].s:.2f})e3')
	f.write(r' \\')
	f.write('\n')
	f.write(f'		{mess[2]:} & {T[2].n:.2f}({T[2].s:.2f}) & {ps[2].n:.2f}({ps[2].s:.2f})e-3 & {1e6*ws[2].n:.2f}({1e6*ws[2].s:.2f})e-6 & {1e-3*rels[2].n:.2f}({1e-3*rels[2].s:.2f})e3')
	f.write(r' \\')
	f.write('\n')
	f.write(f'		{mess[3]:} & {T[3].n:.2f}({T[3].s:.2f}) & {0.1*ps[3].n:.2f}({0.1*ps[3].s:.2f})e-2 & {1e6*ws[3].n:.2f}({1e6*ws[3].s:.2f})e-6 & {1e-3*rels[3].n:.2f}({1e-3*rels[3].s:.2f})e3')
	f.write(r' \\')
	f.write('\n')
	f.write(table_footer)

table_footer = r'''		\bottomrule\bottomrule
	\end{tabular}
'''
table_header_1 = r'''	\sisetup{table-parse-only, table-number-alignment=left}
	\begin{tabular}{c S S S S S S S S}
		\toprule\toprule
		\multirow{2}[2]{*}{$N$} &
		\multicolumn{2}{c}{Abbildung \ref{fig:4}} &
		\multicolumn{2}{c}{Abbildung \ref{fig:5}} &
		\multicolumn{2}{c}{Abbildung \ref{fig:6a}} &
		\multicolumn{2}{c}{Abbildung \ref{fig:6b}} \\
		\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
		& {$n$} & {$r \mathbin{/} \unit{\milli\volt}$} &
		{$n$} & {$r \mathbin{/} \unit{\milli\volt}$} &
		{$n$} & {$r \mathbin{/} \unit{\milli\volt}$} &
		{$n$} & {$r \mathbin{/} \unit{\milli\volt}$} \\
		\midrule
'''
table_header_2 = r'''		\bottomrule
		\toprule
		\multirow{2}[2]{*}{$N$} &
		\multicolumn{2}{c}{Abbildung \ref{fig:7a}} &
		\multicolumn{2}{c}{Abbildung \ref{fig:7b}} &
		\multicolumn{2}{c}{Abbildung \ref{fig:8}} &
		\multicolumn{2}{c}{Abbildung \ref{fig:9}} \\
		\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
		& {$n$} & {$r \mathbin{/} \unit{\milli\volt}$} &
		{$n$} & {$r \mathbin{/} \unit{\milli\volt}$} &
		{$n$} & {$r \mathbin{/} \unit{\milli\volt}$} &
		{$n$} & {$r \mathbin{/} \unit{\milli\volt}$} \\
		\midrule
'''
v1 *= 1e3
v2 *= 1e3
v3a *= 1e3
v3b *= 1e3
v4a *= 1e3
v4b *= 1e3
v5 *= 1e3
v6 *= 1e3
row_template_1 = r'		{0:} & {1:.0f}({2:.0f}) & {3:.0f}({4:.0f}) & {5:.0f}({6:.0f}) & {7:.0f}({8:.0f}) & {9:.0f}({10:.0f}) & {11:.0f}({12:.0f}) & {13:.0f}({14:.0f}) & {15:.0f}({16:.0f}) \\'
row_template_2 = r'		{0:} & {1:.0f}({2:.0f}) & {3:.0f}({4:.0f}) & {5:.0f}({6:.0f}) & {7:.0f}({8:.0f}) & {9:.0f}({10:.0f}) & {11:.0f}({12:.0f}) & & \\'
row_template_3 = r'		{0:} & & & & & {1:.0f}({2:.0f}) & {3:.0f}({4:.0f}) & & \\'
row_template_4 = r'		{0:} & {1:.0f}({2:.0f}) & {3:.0f}({4:.0f}) & & & {5:.0f}({6:.0f}) & {7:.0f}({8:.0f}) & {9:.0f}({10:.0f}) & {11:.0f}({12:.0f}) \\'
ind = np.arange(0, 12, 1)
with open('build/table_2.tex', 'w') as f:
	f.write(table_header_1)
	for row in zip(ind, noms(n1), stds(n1), noms(v1), stds(v1), noms(n2), stds(n2), noms(v2), stds(v2), noms(n3a), stds(n3a), noms(v3a), stds(v3a), noms(n3b), stds(n3b), noms(v3b), stds(v3b)):
		f.write(row_template_1.format(*row))
		f.write('\n')
	for row in zip(ind, noms(n1), stds(n1), noms(v1), stds(v1), noms(n2), stds(n2), noms(v2), stds(v2), noms(n3a), stds(n3a), noms(v3a), stds(v3a)):
		if row[0] > 7:
			f.write(row_template_2.format(*row))
			f.write('\n')
	for row in zip(ind, noms(n3a), stds(n3a), noms(v3a), stds(v3a)):
		if row[0] > 9:
			f.write(row_template_3.format(*row))
			f.write('\n')
	f.write(r'		\midrule')
	f.write('\n')
	f.write(r'		{$\bar{n}$} & \multicolumn{2}{c}{$\num{')
	f.write(f'{stat1.n:.1f}({stat1.s:.1f})')
	f.write(r'}$} & \multicolumn{2}{c}{$\num{')
	f.write(f'{stat2.n:.1f}({stat2.s:.1f})')
	f.write(r'}$} & \multicolumn{2}{c}{$\num{')
	f.write(f'{stat3a.n:.1f}({stat3a.s:.1f})')
	f.write(r'}$} & \multicolumn{2}{c}{$\num{')
	f.write(f'{stat3b.n:.1f}({stat3b.s:.1f})')
	f.write(r'}$} \\')
	f.write('\n')
	f.write(table_header_2)
	for row in zip(ind, noms(n4a), stds(n4a), noms(v4a), stds(v4a), noms(n4b), stds(n4b), noms(v4b), stds(v4b), noms(n5), stds(n5), noms(v5), stds(v5), noms(n6), stds(n6), noms(v6), stds(v6)):
		f.write(row_template_1.format(*row))
		f.write('\n')
	for row in zip(ind, noms(n4a), stds(n4a), noms(v4a), stds(v4a), noms(n5), stds(n5), noms(v5), stds(v5), noms(n6), stds(n6), noms(v6), stds(v6)):
		if row[0] > 7:
			f.write(row_template_4.format(*row))
			f.write('\n')
	f.write(r'		\midrule')
	f.write('\n')
	f.write(r'		{$\bar{n}$} & \multicolumn{2}{c}{$\num{')
	f.write(f'{stat4a.n:.1f}({stat4a.s:.1f})')
	f.write(r'}$} & \multicolumn{2}{c}{$\num{')
	f.write(f'{stat4b.n:.1f}({stat4b.s:.1f})')
	f.write(r'}$} & \multicolumn{2}{c}{$\num{')
	f.write(f'{stat5.n:.1f}({stat5.s:.1f})')
	f.write(r'}$} & \multicolumn{2}{c}{$\num{')
	f.write(f'{stat6.n:.1f}({stat6.s:.1f})')
	f.write(r'}$} \\')
	f.write('\n')
	f.write(table_footer)

table_footer = r'''		\bottomrule
	\end{tabular}
'''
table_header = r'''	\sisetup{table-parse-only}
	\begin{tabular}{S S[table-number-alignment=right] S S[table-number-alignment=right] S S[table-number-alignment=right] @{${}\pm{}$} S[table-number-alignment=left] S S[table-number-alignment=right] @{${}\pm{}$} S[table-number-alignment=left]}
		\toprule
		\multicolumn{4}{c}{Integrales Spektrum} & 
		\multicolumn{6}{c}{Differentielles Spektrum} \\
		\cmidrule(lr){1-4}\cmidrule(lr){5-10}
		{$U_{\hspace{-0.2ex} A} \mathbin{/} \unit{\volt}$} & {$n$} &
		{$U_{\hspace{-0.2ex} A} \mathbin{/} \unit{\volt}$} & {$n$} &
		{$U_{\hspace{-0.2ex} A} \mathbin{/} \unit{\volt}$} & \multicolumn{2}{c}{$n'$} &
		{$U_{\hspace{-0.2ex} A} \mathbin{/} \unit{\volt}$} & \multicolumn{2}{c}{$n'$} \\
		\midrule
'''
row_template_1 = r'		{0:.2f}+-{1:.2f} & {2:.0f}+-{3:.0f} & {4:.2f}+-{5:.2f} & {6:.0f}+-{7:.0f} & {8:.2f}+-{9:.2f} & {10:.1f} & {11:.1f} & {12:.2f}+-{13:.2f} & {14:.1f} & {15:.1f} \\'
row_template_2 = r'		{0:.2f}+-{1:.2f} & {2:.0f}+-{3:.0f} & {4:.2f}+-{5:.2f} & {6:.0f}+-{7:.0f} & {8:.2f}+-{9:.2f} & {10:.1f} & {11:.1f} & & \multicolumn{{2}}{{c}}{{$ $}} \\'
row_template_3 = r'		{0:.2f}+-{1:.2f} & {2:.0f}+-{3:.0f} & & & {4:.2f}+-{5:.2f} & {6:.1f} & {7:.1f} & & \multicolumn{{2}}{{c}}{{$ $}} \\'
with open('build/table_3.tex', 'w') as f:
	f.write(table_header)
	for row in zip(noms(t1[:15]), stds(t1[:15]), noms(y1[:15]), stds(y1[:15]), noms(t1[15:]), stds(t1[15:]), noms(y1[15:]), stds(y1[15:]), noms(t1[:15] + dt1[:15] / 2), stds(t1[:15] + dt1[:15] / 2), -noms(r1[:15]), stds(r1[:15]), noms(t1[15:-1] + dt1[15:] / 2), stds(t1[15:-1] + dt1[15:] / 2), -noms(r1[15:]), stds(r1[15:])):
		f.write(row_template_1.format(*row))
		f.write('\n')
	i = 0
	for row in zip(noms(t1[:15]), stds(t1[:15]), noms(y1[:15]), stds(y1[:15]), noms(t1[15:]), stds(t1[15:]), noms(y1[15:]), stds(y1[15:]), noms(t1[:15] + dt1[:15] / 2), stds(t1[:15] + dt1[:15] / 2), -noms(r1[:15]), stds(r1[:15])):
		if i > 12:
			f.write(row_template_2.format(*row))
			f.write('\n')
		i += 1
	i = 0
	for row in zip(noms(t1[:15]), stds(t1[:15]), noms(y1[:15]), stds(y1[:15]), noms(t1[:15] + dt1[:15] / 2), stds(t1[:15] + dt1[:15] / 2), -noms(r1[:15]), stds(r1[:15])):
		if i > 13:
			f.write(row_template_3.format(*row))
			f.write('\n')
		i += 1
	f.write(table_footer)
with open('build/table_4.tex', 'w') as f:
	f.write(table_header)
	for row in zip(noms(t2[:13]), stds(t2[:13]), noms(y2[:13]), stds(y2[:13]), noms(t2[13:]), stds(t2[13:]), noms(y2[13:]), stds(y2[13:]), noms(t2[:13] + dt2[:13] / 2), stds(t2[:13] + dt2[:13] / 2), -noms(r2[:13]), stds(r2[:13]), noms(t2[13:-1] + dt2[13:] / 2), stds(t2[13:-1] + dt2[13:] / 2), -noms(r2[13:]), stds(r2[13:])):
		f.write(row_template_1.format(*row))
		f.write('\n')
	i = 0
	for row in zip(noms(t2[:13]), stds(t2[:13]), noms(y2[:13]), stds(y2[:13]), noms(t2[13:]), stds(t2[13:]), noms(y2[13:]), stds(y2[13:]), noms(t2[:13] + dt2[:13] / 2), stds(t2[:13] + dt2[:13] / 2), -noms(r2[:13]), stds(r2[:13])):
		if i > 11:
			f.write(row_template_2.format(*row))
			f.write('\n')
		i += 1
	f.write(table_footer)

table_footer = r'''		\bottomrule\bottomrule
	\end{tabular}
'''
table_header_1 = r'''	\sisetup{table-parse-only, table-number-alignment=right}
	\begin{tabular}{c  S S  S S  S S  S S  S S  S S}
		\toprule\toprule
		\multirow{2}[2]{*}{$k$} &
		\multicolumn{2}{c}{Abbildung \ref{fig:6a}} &
		\multicolumn{2}{c}{Abbildung \ref{fig:6b}} \\
		\cmidrule(lr){2-3}\cmidrule(lr){4-5}
		& {$U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} &
		{$\increment U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} &
		{$U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} &
		{$\increment U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} \\
		\midrule
'''
table_header_2 = r'''		\bottomrule
		\toprule
		\multirow{2}[2]{*}{$k$} &
		\multicolumn{2}{c}{Abbildung \ref{fig:7a}} &
		\multicolumn{2}{c}{Abbildung \ref{fig:7b}} \\
		\cmidrule(lr){2-3}\cmidrule(lr){4-5}
		& {$U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} &
		{$\increment U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} &
		{$U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} &
		{$\increment U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} \\
		\midrule
'''
table_header_3 = r'''		\bottomrule
		\toprule
		\multirow{2}[2]{*}{$k$} &
		\multicolumn{2}{c}{Abbildung \ref{fig:8}}  &
		\multicolumn{2}{c}{Abbildung \ref{fig:9}} \\
		\cmidrule(lr){2-3}\cmidrule(lr){4-5}
		& {$U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} &
		{$\increment U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} &
		{$U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} &
		{$\increment U_{\hspace{-0.2ex} k} \mathbin{/} \unit{\volt}$} \\
		\midrule
'''
row_template_1 = r'		{0:} & {1:.2f}+-{2:.2f} & {3:.2f}+-{4:.2f} & {5:.2f}+-{6:.2f} & {7:.2f}+-{8:.2f} \\'
row_template_2 = r'		{0:} & {1:.2f}+-{2:.2f} & & {3:.2f}+-{4:.2f} & \\'
row_template_3 = r'		{0:} & {1:.2f}+-{2:.2f} & {3:.2f}+-{4:.2f} & {5:.2f}+-{6:.2f} & \\'
row_template_4 = r'		{0:} & {1:.2f}+-{2:.2f} & {3:.2f}+-{4:.2f} & & \\'
row_template_5 = r'		{0:} & {1:.2f}+-{2:.2f} & & & \\'
ind = np.arange(1, 10, 1)
with open('build/table_5.tex', 'w') as f:
	f.write(table_header_1)
	for row in zip(ind, noms(t3a), stds(t3a), noms(t3a[1:] - t3a[:-1]), stds(t3a[1:] - t3a[:-1]), noms(t3b), stds(t3b), noms(t3b[1:] - t3b[:-1]), stds(t3b[1:] - t3b[:-1])):
		f.write(row_template_1.format(*row))
		f.write('\n')
	for row in zip(ind, noms(t3a), stds(t3a), noms(t3b), stds(t3b)):
		if row[0] > 4:
			f.write(row_template_2.format(*row))
			f.write('\n')
	f.write(table_header_2)
	for row in zip(ind, noms(t4a), stds(t4a), noms(t4a[1:] - t4a[:-1]), stds(t4a[1:] - t4a[:-1]), noms(t4b), stds(t4b), noms(t4b[1:] - t4b[:-1]), stds(t4b[1:] - t4b[:-1])):
		f.write(row_template_1.format(*row))
		f.write('\n')
	for row in zip(ind, noms(t4a), stds(t4a), noms(t4b), stds(t4b)):
		if row[0] > 4:
			f.write(row_template_2.format(*row))
			f.write('\n')
	f.write(table_header_3)
	for row in zip(ind, noms(t5), stds(t5), noms(t5[1:] - t5[:-1]), stds(t5[1:] - t5[:-1]), noms(t6), stds(t6), noms(t6[1:] - t6[:-1]), stds(t6[1:] - t6[:-1])):
		f.write(row_template_1.format(*row))
		f.write('\n')
	for row in zip(ind, noms(t5), stds(t5), noms(t5[1:] - t5[:-1]), stds(t5[1:] - t5[:-1]), noms(t6), stds(t6)):
		if row[0] > 6:
			f.write(row_template_3.format(*row))
			f.write('\n')
	for row in zip(ind, noms(t5), stds(t5), noms(t5[1:] - t5[:-1]), stds(t5[1:] - t5[:-1])):
		if row[0] > 7:
			f.write(row_template_4.format(*row))
			f.write('\n')
	for row in zip(ind, noms(t5), stds(t5)):
		if row[0] > 8:
			f.write(row_template_5.format(*row))
			f.write('\n')
	f.write(table_footer)

dtt = np.concatenate((t3a[1:] - t3a[:-1], t3b[1:] - t3b[:-1], t4a[1:] - t4a[:-1], t4b[1:] - t4b[:-1], t5[1:] - t5[:-1], t6[1:] - t6[:-1]))

def weighted(uarr):
	return sum(uarr / stds(uarr)**2) / sum(1 / stds(uarr)**2)

dtt = ufloat(noms(weighted(dtt)), np.sqrt(stds(weighted(dtt))**2 + np.mean(stds(dtt))**2))
with open('build/d.tex', 'w') as f:
	f.write(r'\qty{')
	f.write(f'{dtt.n:.2f}+-{dtt.s:.2f}')
	f.write(r'}{\electronvolt}')

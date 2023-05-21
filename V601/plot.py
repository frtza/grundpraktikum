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

table_footer = r'''			\bottomrule
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




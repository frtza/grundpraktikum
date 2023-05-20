import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
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

ab1 = [0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9]
x1 = [0, 0, 0, 0, 0, 0, 0, 12, 0, 11, 0, 2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 2, 4, 10]
x1 = unp.uarray(x1, stds(stat1))
y1 = [158, 158, 157, 156, 153, 151, 146, 143, 138, 131, 120, 117, 114, 110, 106, 100, 89, 80, 71, 56, 44, 30, 22, 15, 10, 6, 3, 1, 0]
y1 = unp.uarray(y1, 1)

ab2 = [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 9]
x2 = [0, 8, 18, 6, 16, 3, 13, 1, 11, 16, 21, 3, 8, 18, 4, 14, 0, 10, 20, 6, 16, 3, 13, 0, 10, 15]
x2 = unp.uarray(x2, stds(stat2))
y2 = [128, 117, 104, 90, 76, 61, 48, 34, 21, 17, 13, 10, 8, 8, 7, 6, 6, 5, 4, 3, 2, 1, 0, -1, -2, -2]
y2 = unp.uarray(y2, 1)

# Plots ausgeben

t1 = 1 * ab1 + x1 * v1[ab1]
dt1 = t1[1:] - t1[:-1]
dy1 = y1[1:] - y1[:-1]
r1 = dy1 / dt1
plt.figure(figsize=(5.78, 4.1616))
plt.imshow(im1, extent=np.array([-96, 1200-96, -23, 864-23]) * 0.00911)
plt.plot([0,5], [0,0], 'ob', mfc='none')
plt.errorbar(noms(t1), noms(y1 / 22.99), xerr=stds(t1), yerr=stds(y1 / 22.99), fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 10, 1))
plt.yticks([])
plt.savefig('build/plot1.pdf', dpi=100)
plt.close()


t2 = 1 * ab2 + x2 * v2[ab2]
dt2 = t2[1:] - t2[:-1]
dy2 = y2[1:] - y2[:-1]
r2 = dy2 / dt2
plt.figure(figsize=(5.78, 4.1616))
plt.imshow(im2, extent=np.array([-66, 1200-66, -17, 864-17]) * 0.00925)
plt.plot([0,5], [0,0], 'ob', mfc='none')
plt.errorbar(noms(t2), noms(y2 / 22.99), xerr=stds(t2), yerr=stds(y2 / 22.99), fmt='none', color='b', elinewidth=0.8)
plt.xticks(np.arange(0, 11, 1))
plt.yticks([])
plt.savefig('build/plot2.pdf', dpi=100)
plt.close()

plt.figure(figsize=(5.78, 4.1616))
plt.imshow(im3a, extent=np.array([-70, 1200-70, -23, 864-23]) * 0.05)
plt.plot([0,30], [0,0], 'ob', mfc='none')
plt.xticks(np.arange(0, 60, 5))
plt.yticks([])
plt.savefig('build/plot3a.pdf', dpi=100)
plt.close()

plt.figure(figsize=(5.78, 4.1616))
plt.imshow(im3b, extent=np.array([-57, 1200-57, -26, 864-26]) * 0.0355)
plt.plot([0,20], [0,0], 'ob', mfc='none')
plt.xticks(np.arange(0, 45, 5))
plt.yticks([])
plt.savefig('build/plot3b.pdf', dpi=100)
plt.close()

plt.figure(figsize=(5.78, 4.1616))
plt.imshow(im4a, extent=np.array([-74, 1200-74, -23, 864-23]) * 0.0501)
plt.plot([0,30], [0,0], 'ob', mfc='none')
plt.xticks(np.arange(0, 60, 5))
plt.yticks([])
plt.savefig('build/plot4a.pdf', dpi=100)
plt.close()

plt.figure(figsize=(5.78, 4.1616))
plt.imshow(im4b, extent=np.array([-85, 1200-85, -25, 864-25]) * 0.0363)
plt.plot([0,20], [0,0], 'ob', mfc='none')
plt.xticks(np.arange(0, 45, 5))
plt.yticks([])
plt.savefig('build/plot4b.pdf', dpi=100)
plt.close()

plt.figure(figsize=(5.78, 4.1616))
plt.imshow(im5, extent=np.array([-69, 1200-69, -24, 864-24]) * 0.051)
plt.plot([0,25], [0,0], 'ob', mfc='none')
plt.xticks(np.arange(0, 60, 5))
plt.yticks([])
plt.savefig('build/plot5.pdf', dpi=100)
plt.close()

plt.figure(figsize=(5.78, 4.1616))
plt.imshow(im6, extent=np.array([-72, 1200-72, -36, 864-36]) * 0.0512)
plt.plot([0,25], [0,0], 'ob', mfc='none')
plt.xticks(np.arange(0, 60, 5))
plt.yticks([])
plt.savefig('build/plot6.pdf', dpi=100)
plt.close()

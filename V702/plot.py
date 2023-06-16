#!/usr/bin/env python
# coding: utf-8

# #### Test

# In[1]:


get_ipython().run_cell_magic('capture', '', "\n%config InlineBackend.figure_formats = ['svg']\n\npwd = %pwd\npwd += '/..'\n\n%env TEXINPUTS=$pwd\n%env MATPLOTLIBRC=$pwd/matplotlibrc\n\n%matplotlib inline\n\nimport matplotlib as mpl\n\nmpl.rc_file('../matplotlibrc')\n\nmpl.use('module://matplotlib_inline.backend_inline')\nmpl.style.use('default')\n\nfrom matplotlib.backends.backend_pgf import FigureCanvasPgf\nmpl.backend_bases.register_backend('pdf', FigureCanvasPgf)\n\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport uncertainties.unumpy as unp\nfrom uncertainties.unumpy import nominal_values as noms, std_devs as stds, uarray as uar\n\ns = np.array([-1e4, 1e4])")


# Nulleffekt

# In[2]:


# zentriertes gleitendes mittel
def smooth(p : int, t, N):
    return t[p-int(p/2)-1:-p+int(p/2)+p%2], np.convolve(N, np.ones(p)/p, mode='valid')


# intervall
t_u = 10

# einlesen
N_u = np.genfromtxt('data/null_10.txt', unpack=True)

# normieren
N_u /= t_u

# zeitschritte
t_u = np.arange(0, t_u * len(N_u), t_u) + t_u

# dimension
plt.figure(figsize=(9, 6))

# grenzen
plt.plot(t_u, N_u, 'k.', ms=0)
x_u = plt.xlim()
y_u = plt.ylim()
plt.xlim(x_u)
plt.ylim(y_u)

# mittelwert
mN_u = uar(np.mean(N_u), np.std(N_u))
plt.plot(s, noms(mN_u) * s / s, c='olivedrab', label='Mittelwert')
plt.fill_between(s, (noms(mN_u) - stds(mN_u)) * s / s, (noms(mN_u) + stds(mN_u)) * s / s, fc='olivedrab', alpha=0.25)

# berechnen
t_uu, N_uu = smooth(2, t_u, N_u)

# messpunkte
plt.plot(t_uu, N_uu, 'k.', ms=5, label='Messdaten')

# beschriftung
plt.xlabel(r'$t\;/\;$s')
plt.ylabel(r'$N\;/\;$s$^{-1}$')

# anzeigen
plt.legend()
plt.close()
print(mN_u)


# In[3]:


# dimensionen
fig = plt.figure(figsize=(10.5, 7))


# einteilung
ax1 = fig.add_subplot(221)

# berechnen
t_u_5, N_u_5 = smooth(5, t_u, N_u)

# grenzen
plt.plot(t_u_5, N_u_5, 'k.', ms=0)
plt.xlim(plt.xlim())
plt.ylim(y_u)

# mittelwert
plt.plot(s, noms(mN_u) * s / s, c='olivedrab', label='Mittelwert')
plt.fill_between(s, noms(mN_u) - stds(mN_u) * s/s, noms(mN_u) + stds(mN_u) * s/s, fc='olivedrab', alpha=0.2)

# messung
plt.plot(t_u_5, N_u_5, 'k.', ms=5, label='Messdaten')

# anzeige
plt.legend()
plt.xlabel('$n = 5$')


#einteilung
ax2 = fig.add_subplot(222)

# berechnen
t_u_10, N_u_10 = smooth(10, t_u, N_u)

# grenzen
plt.plot(t_u_10, N_u_10, 'k.', ms=0)
plt.xlim(plt.xlim())
plt.ylim(y_u)

# mittelwert
plt.plot(s, noms(mN_u) * s / s, c='olivedrab', label='Mittelwert')
plt.fill_between(s, noms(mN_u) - stds(mN_u) * s/s, noms(mN_u) + stds(mN_u) * s/s, fc='olivedrab', alpha=0.2)

# messung
plt.plot(t_u_10, N_u_10, 'k.', ms=5, label='Messdaten')

# anzeige
plt.legend()
plt.xlabel('$n = 10$')


# einteilung
ax3 = fig.add_subplot(223)

# berechnen
t_u_20, N_u_20 = smooth(20, t_u, N_u)

# grenzen
plt.plot(t_u_20, N_u_20, 'k.', ms=0)
plt.xlim(plt.xlim())
plt.ylim(y_u)

# mittelwert
plt.plot(s, noms(mN_u) * s / s, c='olivedrab', label='Mittelwert')
plt.fill_between(s, noms(mN_u) - stds(mN_u) * s/s, noms(mN_u) + stds(mN_u) * s/s, fc='olivedrab', alpha=0.2)

# messung
plt.plot(t_u_20, N_u_20, 'k.', ms=5, label='Messdaten')

# anzeige
plt.legend()
plt.xlabel('$n = 20$')


# einteilung
ax4 = fig.add_subplot(224)

# berechnen
t_u_40, N_u_40 = smooth(40, t_u, N_u)

# grenzen
plt.plot(t_u_40, N_u_40, 'k.', ms=0)
plt.xlim(plt.xlim())
plt.ylim(y_u)

# mittelwert
plt.plot(s, noms(mN_u) * s / s, c='olivedrab', label='Mittelwert')
plt.fill_between(s, noms(mN_u) - stds(mN_u) * s/s, noms(mN_u) + stds(mN_u) * s/s, fc='olivedrab', alpha=0.2)

# messung
plt.plot(t_u_40, N_u_40, 'k.', ms=5, label='Messdaten')

# anzeige
plt.legend()
plt.xlabel('$n = 40$')


# anzeigen
fig.supxlabel(r'$t\;/\;$s')
fig.supylabel(r'$N\;/\;$s$^{-1}$')
plt.subplots_adjust(hspace=0.25, left=0.075)
plt.show()


# Vanadium

# In[4]:


# intervall
t = 30

# einlesen
N = np.genfromtxt('data/v_30.txt', unpack=True)

# normieren
N /= t

# bereinigen
N = N - mN_u

# zeitschritte
t = np.arange(0, t * len(N), t) + t

# dimension
plt.figure(figsize=(9, 6))

# skala konfigurieren
ax1 = plt.subplot(111)
plt.plot(t, noms(N), 'k.', ms=0)
ax1.set(xscale='linear', yscale='log')
ax2 = ax1.twinx()
ax2.axis('off')

# grenzen
plt.errorbar(t, np.log(noms(N)), yerr=stds(unp.log(N)), fmt='k.', ms=0, alpha=0)
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

# regression
par, cov = np.polyfit(t, np.log(noms(N)), 1, cov=True)
err = np.sqrt(np.diag(cov))
rN = uar(par, err)
plt.plot(s, par[0] * s + par[1], c='olivedrab', label='Regression')

# messpunkte
plt.errorbar(t, np.log(noms(N)), yerr=stds(unp.log(N)), fmt='k.', ms=5, label='Messdaten')

# anzeigen
plt.legend()
ax1.set_xlabel(r'$t\;/\;$s')
ax1.set_ylabel(r'$N\;/\;$s$^{-1}$')
plt.show()
print(rN)


# In[5]:


# halbwertszeit aus zerfallskonstante
def hwz(lam):
    return unp.log(2) / lam

# zerfallsrate aus achsenabschnitt
def akt(NN):
    return unp.exp(NN)

# berechnen
lam_V = -rN[0]
tau_V = hwz(lam_V)
num_V = rN[1]
stt_V = akt(num_V)

# ausgabe
print(f'\nV-52\n\n\nHalbwertszeit:\n\nlam = {lam_V:.5f}  [1/s]\ntau = {tau_V:.2f}      [s]\n')
print(f'Ergebnis:\n{(tau_V.n - tau_V.n % 60) / 60:.0f} min {tau_V % 60:.0f} s\n')
print(f'Literatur:\n3 min 44.6 s\n\n\nStartaktivität:\n')
print(f'num = {num_V:.3f}   [ln(1/s)]\nstt = {stt_V:.3f}     [1/s]\n')


# Rhodium

# In[6]:


# intervall
t_1 = 8

# einlesen
N_1 = np.genfromtxt('data/rh_8.txt', unpack=True)

# normieren
N_1 /= t_1

# bereinigen
N_1 = N_1 - mN_u

# zeitschritte
t_1 = np.arange(0, t_1 * len(N_1), t_1) + t_1

# dimension
plt.figure(figsize=(9, 6))

# skala konfigurieren
ax1 = plt.subplot(111)
plt.plot(t_1, noms(N_1), 'k.', ms=0)
ax1.set(xscale='linear', yscale='log')
ax2 = ax1.twinx()
ax2.axis('off')

# grenzen
plt.plot(t_1, np.log(noms(N_1)), 'k.', ms=0)
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

# markierung
plt.axvspan(410, s[1], color='olivedrab', alpha=0.2)

# regression
rN_1, rt_1 = N_1[t_1 >= 410], t_1[t_1 >= 410]
par_1, cov_1 = np.polyfit(rt_1, np.log(noms(rN_1)), 1, cov=True)
err_1 = np.sqrt(np.diag(cov_1))
rN_1 = uar(par_1, err_1)
plt.plot(s, par_1[0] * s + par_1[1], c='olivedrab', label='Regression')

# messpunkte
plt.plot(t_1, np.log(noms(N_1)), 'k.', ms=5, label='Messdaten')

# anzeigen
plt.legend()
ax1.set_xlabel(r'$t\;/\;$s')
ax1.set_ylabel(r'$N\;/\;$s$^{-1}$')
plt.show()
print(rN_1)


# In[7]:


# berechnen
lam_Rh_1 = -rN_1[0]
tau_Rh_1 = hwz(lam_Rh_1)
num_Rh_1 = rN_1[1]
stt_Rh_1 = akt(num_Rh_1)

# ausgabe
print(f'\nRh-104\n\n\nHalbwertszeit:\n\nlam = {lam_Rh_1:.5f}  [1/s]\ntau = {tau_Rh_1:.2f}      [s]\n')
print(f'Ergebnis:\n{(tau_Rh_1.n - tau_Rh_1.n % 60) / 60:.0f} min {tau_Rh_1 % 60:.0f} s\n')
print(f'Literatur:\n4 min 20 s\n\n\nStartaktivität:\n')
print(f'num = {num_Rh_1:.3f}   [ln(1/s)]\nstt = {stt_Rh_1:.3f}     [1/s]\n')


# In[8]:


# bereinigen
N_1i = N_1 - unp.exp(rN_1[0] * t_1 + rN_1[1])

# dimension
plt.figure(figsize=(9, 6))

# skala konfigurieren
ax1 = plt.subplot(111)
plt.plot(t_1, noms(N_1i), 'k.', ms=0)
ax1.set(xscale='linear', yscale='log')
ax2 = ax1.twinx()
ax2.axis('off')

# grenzen
plt.plot(t_1, np.log(abs(noms(N_1i))), 'k.', ms=0)
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

# markierung
plt.axvspan(s[0], 80, color='olivedrab', alpha=0.2)

# regression
rN_1i, rt_1 = N_1i[t_1 <= 80], t_1[t_1 <= 80]
par_1i, cov_1i = np.polyfit(rt_1, np.log(noms(rN_1i)), 1, cov=True)
err_1i = np.sqrt(np.diag(cov_1i))
rN_1i = uar(par_1i, err_1i)
plt.plot(s, par_1i[0] * s + par_1i[1], c='olivedrab', label='Regression')

# messpunkte
plt.plot(t_1, np.log(abs(noms(N_1i))), 'k.', ms=5, label='Messdaten')

# anzeigen
plt.legend()
ax1.set_xlabel(r'$t\;/\;$s')
ax1.set_ylabel(r'$N\;/\;$s$^{-1}$')
plt.show()
print(rN_1i)


# In[9]:


# berechnen
lam_Rh_1i = -rN_1i[0]
tau_Rh_1i = hwz(lam_Rh_1i)
num_Rh_1i = rN_1i[1]
stt_Rh_1i = akt(num_Rh_1i)

# ausgabe
print(f'\nRh-104i\n\n\nHalbwertszeit:\n\nlam = {lam_Rh_1i:.5f}  [1/s]\ntau = {tau_Rh_1i:.2f}      [s]\n')
print(f'Ergebnis:\n{tau_Rh_1i % 60:.0f} s\n')
print(f'Literatur:\n42.3 s\n\n\nStartaktivität:\n')
print(f'num =  {num_Rh_1i:.3f}   [ln(1/s)]\nstt = {stt_Rh_1i:.3f}     [1/s]\n')


# In[10]:


# probe
tt_1 = 410
ttt_1 = 80
print(f'\nBedingung:\n\nt* = {tt_1}  [s]\nt** = {ttt_1}  [s]\n')
print(f'Ni(t*) = {unp.exp(rN_1i[0]*tt_1+rN_1i[1])}  [1/s] << N(t*) = {unp.exp(rN_1[0]*tt_1+rN_1[1])}  [1/s]\n')

# dimension
plt.figure(figsize=(9, 6))

# skala konfigurieren
ax1 = plt.subplot(111)
plt.errorbar(t_1, noms(N_1), yerr=stds(N_1), fmt='k.', ms=0, alpha=0)
ax1.set(xscale='linear', yscale='log')
ax2 = ax1.twinx()
ax2.axis('off')

# grenzen
plt.errorbar(t_1, np.log(noms(N_1)), yerr=stds(unp.log(N_1)), fmt='k.', ms=0, alpha=0)
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

# summenkurve
xx = np.linspace(-100, 800, 9000)
yy = unp.log(unp.exp(rN_1[0] * xx + rN_1[1]) + unp.exp(rN_1i[0] * xx + rN_1i[1]))
plt.plot(xx, noms(yy), c='olivedrab', label='Superposition')

# messpunkte
plt.errorbar(t_1, np.log(noms(N_1)), yerr=stds(unp.log(N_1)), fmt='k.', ms=5, label='Messpunkte')

# anzeigen
plt.legend()
ax1.set_xlabel(r'$t\;/\;$s')
ax1.set_ylabel(r'$N\;/\;$s$^{-1}$')
plt.show()


# In[11]:


# intervall
t_2 = 15

# einlesen
N_2 = np.genfromtxt('data/rh_15.txt', unpack=True)

# normieren
N_2 /= t_2

# bereinigen
N_2 = N_2 - mN_u

# zeitschritte
t_2 = np.arange(0, t_2 * len(N_2), t_2) + t_2

# dimension
plt.figure(figsize=(9, 6))

# skala konfigurieren
ax1 = plt.subplot(111)
plt.plot(t_2, noms(N_2), 'k.', ms=0)
ax1.set(xscale='linear', yscale='log')
ax2 = ax1.twinx()
ax2.axis('off')

# grenzen
plt.plot(t_2, np.log(noms(N_2)), 'k.', ms=0)
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

# markierung
plt.axvspan(420, s[1], color='olivedrab', alpha=0.2)

# regression
rN_2, rt_2 = N_2[t_2 >= 420], t_2[t_2 >= 420]
par_2, cov_2 = np.polyfit(rt_2, np.log(noms(rN_2)), 1, cov=True)
err_2 = np.sqrt(np.diag(cov_2))
rN_2 = uar(par_2, err_2)
plt.plot(s, par_2[0] * s + par_2[1], c='olivedrab', label='Regression')

# messpunkte
plt.plot(t_2, np.log(noms(N_2)), 'k.', ms=5, label='Messdaten')

# anzeigen
plt.legend()
ax1.set_xlabel(r'$t\;/\;$s')
ax1.set_ylabel(r'$N\;/\;$s$^{-1}$')
plt.show()
print(rN_2)


# In[12]:


# berechnen
lam_Rh_2 = -rN_2[0]
tau_Rh_2 = hwz(lam_Rh_2)
num_Rh_2 = rN_2[1]
stt_Rh_2 = akt(num_Rh_2)

# ausgabe
print(f'\nRh-104\n\n\nHalbwertszeit:\n\nlam = {lam_Rh_2:.5f}  [1/s]\ntau = {tau_Rh_2:.2f}      [s]\n')
print(f'Ergebnis:\n{(tau_Rh_2.n - tau_Rh_2.n % 60) / 60:.0f} min {tau_Rh_2 % 60:.0f} s\n')
print(f'Literatur:\n4 min 20 s\n\n\nStartaktivität:\n')
print(f'num = {num_Rh_2:.3f}   [ln(1/s)]\nstt = {stt_Rh_2:.3f}     [1/s]\n')


# In[13]:


# bereinigen
N_2i = N_2 - unp.exp(rN_2[0] * t_2 + rN_2[1])

# dimension
plt.figure(figsize=(9, 6))

# skala konfigurieren
ax1 = plt.subplot(111)
plt.plot(t_2, noms(N_2i), 'k.', ms=0)
ax1.set(xscale='linear', yscale='log')
ax2 = ax1.twinx()
ax2.axis('off')

# grenzen
plt.plot(t_2, np.log(abs(noms(N_2i))), 'k.', ms=0)
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

# markierung
plt.axvspan(s[0], 210, color='olivedrab', alpha=0.2)

# regression
rN_2i, rt_2 = N_2i[t_2 <= 210], t_2[t_2 <= 210]
par_2i, cov_2i = np.polyfit(rt_2, np.log(noms(rN_2i)), 1, cov=True)
err_2i = np.sqrt(np.diag(cov_2i))
rN_2i = uar(par_2i, err_2i)
plt.plot(s, par_2i[0] * s + par_2i[1], c='olivedrab', label='Regression')

# messpunkte
plt.plot(t_2, np.log(abs(noms(N_2i))), 'k.', ms=5, label='Messdaten')

# anzeigen
plt.legend()
ax1.set_xlabel(r'$t\;/\;$s')
ax1.set_ylabel(r'$N\;/\;$s$^{-1}$')
plt.show()
print(rN_2i)


# In[14]:


# berechnen
lam_Rh_2i = -rN_2i[0]
tau_Rh_2i = hwz(lam_Rh_2i)
num_Rh_2i = rN_2i[1]
stt_Rh_2i = akt(num_Rh_2i)

# ausgabe
print(f'\nRh-104i\n\n\nHalbwertszeit:\n\nlam = {lam_Rh_2i:.5f}  [1/s]\ntau = {tau_Rh_2i:.2f}      [s]\n')
print(f'Ergebnis:\n{tau_Rh_2i % 60:.0f} s\n')
print(f'Literatur:\n42.3 s\n\n\nStartaktivität:\n')
print(f'num =  {num_Rh_2i:.3f}   [ln(1/s)]\nstt = {stt_Rh_2i:.3f}     [1/s]\n')


# In[15]:


# probe
tt_2 = 420
ttt_2 = 210
print(f'\nBedingung:\n\nt*  = {tt_2}  [s]\nt** = {ttt_2}  [s]\n')
print(f'Ni(t*) = {unp.exp(rN_2i[0]*tt_2+rN_2i[1])}  [1/s] << N(t*) = {unp.exp(rN_2[0]*tt_2+rN_2[1])}  [1/s]\n')

# dimension
plt.figure(figsize=(9, 6))

# skala konfigurieren
ax1 = plt.subplot(111)
plt.errorbar(t_2, noms(N_2), yerr=stds(N_2), fmt='k.', ms=0, alpha=0)
ax1.set(xscale='linear', yscale='log')
ax2 = ax1.twinx()
ax2.axis('off')

# grenzen
plt.errorbar(t_2, np.log(noms(N_2)), yerr=stds(unp.log(N_2)), fmt='k.', ms=0, alpha=0)
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())

# summenkurve
xx = np.linspace(-100, 800, 9000)
yy = unp.log(unp.exp(rN_2[0] * xx + rN_2[1]) + unp.exp(rN_2i[0] * xx + rN_2i[1]))
plt.plot(xx, noms(yy), c='olivedrab', label='Superposition')

# messpunkte
plt.errorbar(t_2, np.log(noms(N_2)), yerr=stds(unp.log(N_2)), fmt='k.', ms=5, label='Messpunkte')

# anzeigen
plt.legend()
ax1.set_xlabel(r'$t\;/\;$s')
ax1.set_ylabel(r'$N\;/\;$s$^{-1}$')
plt.show()


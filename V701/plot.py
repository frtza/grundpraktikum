# essential libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# read columns of data from txt file
p_6, qty_6, max_6, chnl_6 = np.genfromtxt('data/stats_6cm.txt', unpack=True)
x_6 = p_6 * 6.0 / 1013.0
E_6 = chnl_6 * 4.0 / chnl_6[0]

p_4, qty_4, max_4, chnl_4 = np.genfromtxt('data/stats_4cm.txt', unpack=True)
x_4 = p_4 * 4.0 / 1013.0
E_4 = chnl_4 * 4.0 / chnl_4[0]

st = np.genfromtxt('data/qty_4cm_300mbar.txt', unpack=True)
st = np.sort(st)
mean, var = np.mean(st), np.var(st)

rng = np.random.default_rng(99)
gauss = rng.normal(loc=mean, scale=np.sqrt(var), size=500)
poisson = rng.poisson(lam=mean, size=500)

# define fit functions
def sig(x, a, b, c, d):
	return a / (1 + np.exp(b * (x - c))) + d

def lin(x, a, b):
	return a * x + b

# parameter and error arrays, covariance matrix for regression
par_6, cov_6 = curve_fit(sig, x_6, qty_6, p0=(-20000, -200, 2, 20000))
err_6 = np.sqrt(np.diag(cov_6))

par_6p, cov_6p = np.polyfit(x_6, E_6, deg=1, cov=True)
err_6p = np.sqrt(np.diag(cov_6p))

par_4, cov_4 = curve_fit(sig, x_4, qty_4, p0=(-20000, -200, 2, 20000))
err_4 = np.sqrt(np.diag(cov_4))

par_4p, cov_4p = np.polyfit(x_4, E_4, deg=1, cov=True)
err_4p = np.sqrt(np.diag(cov_4p))

# graphical representation of correlation
x = np.linspace(-0.04, 3, 10000)

plt.plot(x, sig(x, *par_6), label='Sigmoid-Fit')
plt.plot(x_6, qty_6, 'kx', ms=3.21, label='Messwerte')
plt.xlabel(r'$x \mathbin{/} \unit{\centi\meter}$')
plt.ylabel(r'$N_\text{tot}$')
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('build/plot_qty_6.pdf')
plt.close()

plt.plot(x, lin(x, *par_6p), label='Regression')
plt.plot(x_6, E_6, 'kx', ms=3.21, label='Messwerte')
plt.xlabel(r'$x \mathbin{/} \unit{\centi\meter}$')
plt.ylabel(r'$E_\text{max} \mathbin{/} \unit{\mega\electronvolt}$')
plt.legend()
plt.savefig('build/plot_E_6.pdf')
plt.close()

plt.plot(x, sig(x, *par_4), label='Sigmoid-Fit')
plt.plot(x_4, qty_4, 'kx', ms=3.21, label='Messwerte')
plt.xlabel(r'$x \mathbin{/} \unit{\centi\meter}$')
plt.ylabel(r'$N_\text{tot}$')
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('build/plot_qty_4.pdf')
plt.close()

plt.plot(x, lin(x, *par_4p), label='Regression')
plt.plot(x_4, E_4, 'kx', ms=3.21, label='Messwerte')
plt.xlabel(r'$x \mathbin{/} \unit{\centi\meter}$')
plt.ylabel(r'$E_\text{max} \mathbin{/} \unit{\mega\electronvolt}$')
plt.legend()
plt.savefig('build/plot_E_4.pdf')
plt.close()

plt.hist(st, bins=7, density=True, histtype='stepfilled', label='Messverteilung')
plt.hist(gauss, bins=7, density=True, histtype='step', lw=1.25, label='Gaußverteilung')
plt.hist(poisson, bins=7, density=True, histtype='step', lw=1.25, label='Poissonverteilung')
plt.xlabel(r'$N_\text{tot}$')
plt.ylabel(r'$n_\text{rel}$')
plt.legend()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig('build/plot_st.pdf')
plt.close()


# format and export latex table
table_footer = r''' 		\bottomrule
	\end{tabular}
'''
table_header = r'''	\sisetup{table-format=3.0}
 	\begin{tabular}
		{S S[table-format=1.2] S[table-format=5.0] S S S[table-format=1.2]}
		\toprule
		{$p \mathbin{/} \unit{\milli\bar}$} &
		{$x \mathbin{/} \unit{\centi\meter}$} &
		{$N_\text{tot}$} &
		{$N_\text{max}$} &
		{$\text{CH}$} &
		{$E_\text{max} \mathbin{/} \unit{\mega\electronvolt}$} \\
		\midrule
'''
row_template = r'		{0:3.0f} & {1:1.2f} & {2:5.0f} & {3:3.0f} & {4:3.0f} & {5:1.2f} \\'
with open('build/table_6.tex', 'w') as f:
	f.write(table_header)
	for row in zip(p_6, x_6, qty_6, max_6, chnl_6, E_6):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)
with open('build/table_4.tex', 'w') as f:
	f.write(table_header)
	for row in zip(p_4, x_4, qty_4, max_4, chnl_4, E_4):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)
table_header = r'''	\sisetup{table-format=4.0}
 	\begin{tabular}
		{S S S S S S S S S S}
		\toprule
'''
row_template = r'		{0:4.0f} & {1:4.0f} & {2:4.0f} & {3:4.0f} & {4:4.0f} & {5:4.0f} & {6:4.0f} & {7:4.0f} & {8:4.0f} & {9:4.0f} \\'
with open('build/table_st.tex', 'w') as f:
	f.write(table_header)
	for row in zip(st[0:10], st[10:20], st[20:30], st[30:40], st[40:50], st[50:60], st[60:70], st[70:80], st[80:90], st[90:100]):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)

# format and export calculated values to build directory

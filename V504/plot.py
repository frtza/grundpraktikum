# essential libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds
from uncertainties import ufloat

# read columns of data from txt file
U, I_1, I_2, I_3 = np.genfromtxt('data/raumladung.txt', unpack=True)
U_af, I_a = np.genfromtxt('data/anlauf.txt', unpack=True)

# correction for internal resistance
U_a = U_af - 1e6 * I_a * 1e-9
U_a[0] = 0

# implementation of gaussian filter for smoothing unevenly spaced data
def gaussian_sum_smooth(x_data, y_data, x_eval, sig):
	delta_x = x_eval[:, None] - x_data
	weights = np.exp((-delta_x**2)/(2*sig**2)) / (np.sqrt(2*np.pi)*sig)
	weights /= np.sum(weights, axis=1, keepdims=True)
	return np.dot(weights, y_data)

# physical constants
eps_0 = const.epsilon_0
e_0 = const.e
m_0 = const.m_e
k_B = const.k
h_0 = const.h

# functions for nonlinear regression
def anlauf(x, p, q):
	return p * np.exp(q*x)
def raumladung(x, a, b):
	return a * 4/9 * eps_0 * np.sqrt(2*e_0/m_0) * x**b
def saettigung(x, u, v, w):
	return u + v/x**w

# parameter and error arrays, covariance matrix for regression
par_11, cov_11 = curve_fit(raumladung, U[:14], I_1[:14])
err_11 = np.sqrt(np.diag(cov_11))
par_12, cov_12 = curve_fit(saettigung, U[13:], I_1[13:])
err_12 = np.sqrt(np.diag(cov_12))

par_21, cov_21 = curve_fit(raumladung, U[:17], I_2[:17])
err_21 = np.sqrt(np.diag(cov_21))
par_22, cov_22 = curve_fit(saettigung, U[16:], I_2[16:])
err_22 = np.sqrt(np.diag(cov_22))

par_31, cov_31 = curve_fit(raumladung, U[:22], I_3[:22])
err_31 = np.sqrt(np.diag(cov_31))
par_32, cov_32 = curve_fit(saettigung, U[21:], I_3[21:])
err_32 = np.sqrt(np.diag(cov_32))

par_4, cov_4 = curve_fit(anlauf, U_a[:-2], I_a[:-2])
err_4 = np.sqrt(np.diag(cov_4))

# graphical representation of correlation
plt.plot(np.linspace(20.65,252,23135), saettigung(np.linspace(20.65,252,23135), *par_12), label='Sättigungsstrom')
plt.plot(np.linspace(-2,19.75,2175), raumladung(np.linspace(-2,19.75,2175), *par_11), label='Raumladungsstrom')
plt.plot(U, I_1, 'kx', markersize=3.21, label='Messdaten')
plt.xlabel(r'$U_A \mathbin{/} \unit{\volt}$')
plt.ylabel(r'$I_A \mathbin{/} \unit{\milli\ampere}$')
plt.legend(loc='lower right')
plt.savefig('build/plot_1.pdf')
plt.close()

plt.plot(np.linspace(52,252,20000), saettigung(np.linspace(52,252,20000), *par_22), label='Sättigungsstrom')
plt.plot(np.linspace(-2,48,5000), raumladung(np.linspace(-2,48,5000), *par_21), label='Raumladungsstrom')
plt.plot(U, I_2, 'kx', markersize=3.21, label='Messdaten')
plt.xlabel(r'$U_A \mathbin{/} \unit{\volt}$')
plt.ylabel(r'$I_A \mathbin{/} \unit{\milli\ampere}$')
plt.legend(loc='lower right')
plt.savefig('build/plot_2.pdf')
plt.close()

plt.plot(np.linspace(102.5,252,14950), saettigung(np.linspace(102.5,252,14950), *par_32), label='Sättigungsstrom')
plt.plot(np.linspace(-2,97,9900), raumladung(np.linspace(-2,97,9900), *par_31), label='Raumladungsstrom')
plt.plot(U, I_3, 'kx', markersize=3.21, label='Messdaten')
plt.xlabel(r'$U_A \mathbin{/} \unit{\volt}$')
plt.ylabel(r'$I_A \mathbin{/} \unit{\milli\ampere}$')
plt.legend(loc='lower right')
plt.savefig('build/plot_3.pdf')
plt.close()

plt.plot(np.linspace(-0.004,0.91,9140), anlauf(np.linspace(-0.004,0.91,9140), *par_4), label='Regression')
plt.plot(U_a, I_a, 'kx', markersize=3.21, label='Messdaten')
plt.xlabel(r'$\hat{U}_{-A} \mathbin{/} \unit{\volt}$')
plt.ylabel(r'$I_A \mathbin{/} \unit{\nano\ampere}$')
plt.legend(loc='upper right')
plt.savefig('build/plot_4.pdf')
plt.close()

# format and export latex table
table_header = r''' 	\begin{tabular}
		{S[table-format=3.0, detect-weight, detect-shape, detect-mode]
		 S[table-format=1.3, detect-weight, detect-shape, detect-mode]
		 S[table-format=1.3, detect-weight, detect-shape, detect-mode]
		 S[table-format=1.3, detect-weight, detect-shape, detect-mode]}
		\toprule
		{$ $} &
		\multicolumn{3}{c}{$I_A \mathbin{/} \unit{\milli\ampere}$} \\
		\cmidrule[\lightrulewidth]{2-4}
		{$ $} & 
		{$I_H = \qty{2.0}{\ampere}$} & 
		{$I_H = \qty{2.2}{\ampere}$} & 
		{$I_H = \qty{2.4}{\ampere}$} \\
		{$U_A \mathbin{/} \unit{\volt}$} & 
		{$U_H = \qty{3.5}{\volt}$} & 
		{$U_H = \qty{4.5}{\volt}$} & 
		{$U_H = \qty{5.0}{\volt}$} \\
		\midrule
'''
table_footer = r''' 		\bottomrule
	\end{tabular}
'''
row_template_0 = r'		{0:3.0f} & {1:1.3f} & {2:1.3f} & {3:1.3f} \\'
row_template_1 = r'		\bfseries {0:3.0f} & \bfseries {1:1.3f} & \bfseries {2:1.3f} & \bfseries {3:1.3f} \\'
with open('build/table_1.tex', 'w') as f:
	f.write(table_header)
	for row in zip(U[0:1], I_1[0:1], I_2[0:1], I_3[0:1]):
		f.write(row_template_0.format(*row))
		f.write('\n')
	for row in zip(U[1:7], I_1[1:7], I_2[1:7], I_3[1:7]):
		f.write(row_template_1.format(*row))
		f.write('\n')
	for row in zip(U[7:8], I_1[7:8], I_2[7:8], I_3[7:8]):
		f.write(row_template_0.format(*row))
		f.write('\n')
	for row in zip(U[8:13], I_1[8:13], I_2[8:13], I_3[8:13]):
		f.write(row_template_1.format(*row))
		f.write('\n')
	for row in zip(U[13:], I_1[13:], I_2[13:], I_3[13:]):
		f.write(row_template_0.format(*row))
		f.write('\n')
	f.write(table_footer)

table_header = r'''	\begin{tabular}
		{S[table-format=1.1, detect-weight, detect-shape, detect-mode]
		 S[table-format=1.3, detect-weight, detect-shape, detect-mode]
		 S[table-format=1.6, detect-weight, detect-shape, detect-mode]
		 @{\hspace{9ex}}
		 S[table-format=1.1, detect-weight, detect-shape, detect-mode]
		 S[table-format=1.3, detect-weight, detect-shape, detect-mode]
		 S[table-format=1.6, detect-weight, detect-shape, detect-mode]}
		\toprule
		{$U_{-A} \mathbin{/} \unit{\volt}$} &
		{$I_A \mathbin{/} \unit{\nano\ampere}$} &
		{$\hat{U}_{-A} \mathbin{/} \unit{\volt}$} &
		{$U_{-A} \mathbin{/} \unit{\volt}$} &
		{$I_A \mathbin{/} \unit{\nano\ampere}$} &
		{$\hat{U}_{-A} \mathbin{/} \unit{\volt}$} \\
		\midrule
'''
row_template_0 = r'		{0:1.1f} & {1:} & {2:1.6f} & {3:1.1f} & {4:} & {5:1.6f} \\'
row_template_1 = r'		{0:1.1f} & {1:} & {2:1.6f} & \bfseries {3:1.1f} & \bfseries {4:} & \bfseries {5:1.6f} \\'
with open('build/table_2.tex', 'w') as f:
	f.write(table_header)
	for row in zip(U_af[:3], I_a[:3], U_a[:3], U_af[5:8], I_a[5:8], U_a[5:8]):
		f.write(row_template_0.format(*row))
		f.write('\n')
	for row in zip(U_af[3:5], I_a[3:5], U_a[3:5], U_af[8:10], I_a[8:10], U_a[8:10]):
		f.write(row_template_1.format(*row))
		f.write('\n')
	f.write(table_footer)

N_WL = 1.0
I_H = unp.uarray([2.0, 2.2, 2.4], [0.1, 0.1, 0.1])
U_H = unp.uarray([3.5, 4.5, 5.0], [1, 1, 1])
T_H = ((I_H*U_H - N_WL)/(0.32*0.28*5.7e-12))**(1/4)
I_S = unp.uarray([par_12[0], par_22[0], par_32[0]], [err_12[0], err_22[0], err_32[0]]) * 1e-3
W_J = -k_B * T_H * unp.log((I_S * h_0**3)/(4*np.pi * e_0 * m_0 * 0.000032 * k_B**2 * T_H**2))
W_eV = W_J / e_0
W_w = ufloat(sum(noms(W_eV)/(stds(W_eV)**2))/sum(1/(stds(W_eV)**2)), np.sqrt(1/sum(1/(stds(W_eV)**2))))
table_header = r'''	\begin{tabular}
		{S[table-format=1.1]
		 @{${}\pm{}$}
		 S[table-format=1.1]
		 S[table-format=1.1]
		 @{${}\pm{}$}
		 S[table-format=1.1]
		 S[table-format=4.0]
		 @{${}\pm{}$}
		 S[table-format=1.0]
		 S[table-format=1.3]
		 @{${}\pm{}$}
		 S[table-format=1.3]
		 S[table-format=3.1]
		 @{${}\pm{}$}
		 S[table-format=2.1, table-number-alignment=left]
		 S[table-format=1.3]
		 @{${}\pm{}$}
		 S[table-format=1.3]}
		\toprule
		\multicolumn{2}{c}{$I_H \mathbin{/} \unit{\ampere}$} &
		\multicolumn{2}{c}{$U_H \mathbin{/} \unit{\volt}$} &
		\multicolumn{2}{c}{$T \mathbin{/} \unit{\kelvin}$} &
		\multicolumn{2}{c}{$I_S \mathbin{/} \unit{\milli\ampere}$} &
		\multicolumn{2}{c}{$W \mathbin{/} \qty{e-21}{\joule}$} &
		\multicolumn{2}{c}{$W \mathbin{/} \unit{\electronvolt}$} \\
		\midrule
'''
row_template = r'		{0:1.1f} & {1:1.1f} & {2:1.1f} & {3:1.1f} & {4:4.0f} & {5:1.0} & {6:1.3f} & {7:1.3f} & {8:3.1f} & {9:2.1f} & {10:1.3f} & {11:1.3f} \\'
with open('build/table_3.tex', 'w') as f:
	f.write(table_header)
	for row in zip(noms(I_H), stds(I_H), noms(U_H), stds(U_H), noms(T_H), stds(T_H), noms(I_S) * 1e3, stds(I_S) * 1e3, noms(W_J) * 1e21, stds(W_J) * 1e21, noms(W_eV), stds(W_eV)):
		f.write(row_template.format(*row))
		f.write('\n')
	f.write(table_footer)

# format and export calculated values to build directory
with open('build/a_1.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_11[0]:.0f}({err_11[0]:.0f})')
	f.write(r'}')
with open('build/b_1.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_11[1]:.3f}({err_11[1]:.3f})')
	f.write(r'}')
with open('build/u_1.tex', 'w') as f:
	f.write(r'\qty[per-mode=reciprocal]{')
	f.write(f'{par_12[0]:.3f}({err_12[0]:.3f})')
	f.write(r'}{\milli\ampere}')
with open('build/v_1.tex', 'w') as f:
	f.write(r'\qty[per-mode=reciprocal]{')
	f.write(f'{par_12[1]:.2f}({err_12[1]:.2f})')
	f.write(r'}{\milli\ampere\volt}\,^{\num{')
	f.write(f'{par_12[2]:.2f}({err_12[2]:.2f})')
	f.write(r'}}')
with open('build/w_1.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_12[2]:.2f}({err_12[2]:.2f})')
	f.write(r'}')

with open('build/a_2.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_21[0]:.0f}({err_21[0]:.0f})')
	f.write(r'}')
with open('build/b_2.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_21[1]:.3f}({err_21[1]:.3f})')
	f.write(r'}')
with open('build/u_2.tex', 'w') as f:
	f.write(r'\qty[per-mode=reciprocal]{')
	f.write(f'{par_22[0]:.3f}({err_22[0]:.3f})')
	f.write(r'}{\milli\ampere}')
with open('build/v_2.tex', 'w') as f:
	f.write(r'\qty[per-mode=reciprocal]{')
	f.write(f'{par_22[1]:.1f}({err_22[1]:.1f})')
	f.write(r'}{\milli\ampere\volt}\,^{\num{')
	f.write(f'{par_22[2]:.2f}({err_22[2]:.2f})')
	f.write(r'}}')
with open('build/w_2.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_22[2]:.2f}({err_22[2]:.2f})')
	f.write(r'}')

with open('build/a_3.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_31[0]:.0f}({err_31[0]:.0f})')
	f.write(r'}')
with open('build/b_3.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_31[1]:.3f}({err_31[1]:.3f})')
	f.write(r'}')
with open('build/u_3.tex', 'w') as f:
	f.write(r'\qty[per-mode=reciprocal]{')
	f.write(f'{par_32[0]:.3f}({err_32[0]:.3f})')
	f.write(r'}{\milli\ampere}')
with open('build/v_3.tex', 'w') as f:
	f.write(r'\qty[per-mode=reciprocal]{')
	f.write(f'{par_32[1]:.0f}({err_32[1]:.0f})')
	f.write(r'}{\milli\ampere\volt}\,^{\num{')
	f.write(f'{par_32[2]:.2f}({err_32[2]:.2f})')
	f.write(r'}}')
with open('build/w_3.tex', 'w') as f:
	f.write(r'\num{')
	f.write(f'{par_32[2]:.2f}({err_32[2]:.2f})')
	f.write(r'}')

with open('build/p.tex', 'w') as f:
	f.write(r'\qty[per-mode=reciprocal]{')
	f.write(f'{par_4[0]:.2f}({err_4[0]:.2f})')
	f.write(r'}{\nano\ampere}')
with open('build/q.tex', 'w') as f:
	f.write(r'\qty[per-mode=reciprocal]{')
	f.write(f'{par_4[1]:.2f}({err_4[1]:.2f})')
	f.write(r'}{\per\volt}')
w = ufloat(par_4[1],err_4[1])
with open('build/T.tex', 'w') as f:
	f.write(r'\qty[per-mode=reciprocal]{')
	f.write(f'{(-e_0/(k_B*w)).n:.0f}({(-e_0/(k_B*w)).s:.0f})')
	f.write(r'}{\kelvin}')

with open('build/W.tex', 'w') as f:
	f.write(r'\qty[per-mode=reciprocal]{')
	f.write(f'{W_w.n:.3f}({W_w.s:.3f})')
	f.write(r'}{\electronvolt}')


def f(v):
    return (2 * nu0 * v)/c

plt.plot(v15, f(v15), 'xr', markersize=6 , label = 'Messdaten')

# Ausgleichsrechung
def g(x, a, b):
    return a*x + b

para, pcov = curve_fit(g, v15, f(v15))
a, b = para
fa, fb = np.sqrt(np.diag(pcov))
ua = ufloat(a, fa) 
ub = ufloat(b, fb)

xx = np.linspace(0, 1, 10**4)
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)

plt.xlabel(r'$v \, / \, \mathrm{ms^{-1}}$')
plt.ylabel(r'$\frac{\Delta \nu}{\cos (\alpha)}$')
plt.legend(loc="best")                  
plt.grid(True)                          
plt.xlim(0, 1)                  
plt.ylim(0, 0.014)
v30 = abs(v30)
plt.plot(v30, f(v30), 'xr', markersize=6 , label = 'Messdaten')

# Ausgleichsrechung
para, pcov = curve_fit(g, v30, f(v30))
a, b = para
fa, fb = np.sqrt(np.diag(pcov))
ua = ufloat(a, fa) 
ub = ufloat(b, fb)

xx = np.linspace(0, 1.4, 10**4)
plt.plot(xx, g(xx, a, b), '-b', linewidth = 1, label = 'Ausgleichsfunktion', alpha=0.5)

plt.xlabel(r'$v \, / \, \mathrm{ms^{-1}}$')
plt.ylabel(r'$\frac{\Delta \nu}{\cos (\alpha)}$')
plt.legend(loc="best")                  
plt.grid(True)                          
plt.xlim(0, 1.3)                  
plt.ylim(0, 0.018)
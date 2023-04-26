import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from numpy import sqrt
import pandas as pd
import scipy.constants as const
from scipy.optimize import curve_fit                        # Funktionsfit:     popt, pcov = curve_fit(func, xdata, ydata) 
from uncertainties import ufloat                            # Fehler:           fehlerwert =  ulfaot(x, err)
from uncertainties import unumpy as unp 
from uncertainties.unumpy import uarray                     # Array von Fehler: fehlerarray =  uarray(array, errarray)
from uncertainties.unumpy import (nominal_values as noms,   # Wert:             noms(fehlerwert) = x
                                  std_devs as stds)         # Abweichung:       stds(fehlerarray) = errarray

#latexTabelle der Messdaten zur Kennlinie des Geiger-Müller-Zählrohrs
print('Tabelle zur Krnnlinie:')
kennlinie=pd.read_csv('data/kennlinie.txt',sep=' ', header=None, names=['Spannung', 'Zählrate', 'Strom'])
print(kennlinie.to_latex(index=False, column_format="c c c"))

#Kennlinie Geiger-Müller

U, N, I = np.genfromtxt('data/kennlinie.txt', unpack=True, skip_header=1)   






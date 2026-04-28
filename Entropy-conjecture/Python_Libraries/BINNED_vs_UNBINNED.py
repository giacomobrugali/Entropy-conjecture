import numpy as np
import operazioni as op
import matplotlib.pyplot as plt
from math import floor, ceil
from iminuit import Minuit
from scipy.stats import expon, norm
from iminuit.cost import ExtendedBinnedNLL
from iminuit.cost import UnbinnedNLL
from IPython.display import display
import funzioni as func


with open ('dati_2.txt') as f :
   
   eventi = [float (x) for x in f.readlines ()]

def main () :
   #definisco le variabili da dare a iminuit per iniziare il fit
   M = ceil (max (eventi))
   m = floor (min (eventi))
   MU = op.media ( eventi)
   DEV = op.deviazione_standard (eventi)
   
   # creo un istogramma numpy per inizializzare bin_content e bin_edges    
   N_bins = floor (len(eventi) / 100)
   x_range = (m, M)
   x_val = np.linspace (-5, 15, 101)
   bin_content, bin_edges = np.histogram (eventi, bins = N_bins, range = x_range)
   N_events = sum(bin_content)
   
   #inizializzo la cost_function per il fit e lo eseguo BINNED
   cost_func = ExtendedBinnedNLL (bin_content, bin_edges, func.gaussian)
   my_minuit = Minuit (cost_func,N_signal = N_events, mu = MU, sigma = DEV) 
   my_minuit.migrad (ncall = 10000)
   
   #inizializzo la cost_function per il fit e lo eseguo UNBINNED
   cost_func_unb = UnbinnedNLL (eventi, func.gauss)
   my_minuit1 = Minuit (cost_func_unb, mu = MU, sigma = DEV) 
   my_minuit1.migrad (ncall = 10000)
   
   #stampo i valori ottenuti dal fit BINNED
   print (my_minuit.valid)
   #display (my_minuit)
   #print(my_minuit.covariance)
   #print (my_minuit.valid)
   # formatted output
   for par, val, err in zip (my_minuit.parameters, my_minuit.values, my_minuit.errors) :
       print(f'{par} = {val:.3f} +/- {err:.3f}') 
   #print(my_minuit.fval)
   
   #stampo i valori ottenuti dal fit UNBINNED
   print (my_minuit1.valid)
   #display (my_minuit)
   #print(my_minuit.covariance)
   #print (my_minuit.valid)
   # formatted output
   for par, val, err in zip (my_minuit1.parameters, my_minuit1.values, my_minuit1.errors) :
       print(f'{par} = {val:.3f} +/- {err:.3f}') 
   #print(my_minuit.fval)
   
   
   
   #estraggo i valori ottenuti dal fit BINNED
   N_signal_fit = my_minuit.values[0]
   mu_fit       = my_minuit.values[1]
   sigma_fit    = my_minuit.values[2]

   #estraggo i valori ottenuti dal fit UNBINNED
   mu_fit1       = my_minuit1.values[0]
   sigma_fit1    = my_minuit1.values[1]

   # disegno la gaussiana con i valori del fit BINNED
   bin_width = bin_edges[1] - bin_edges[0]
   y_val = func.gaussian(x_val, N_signal_fit, mu_fit, sigma_fit) * bin_width
   
   # disegno la gaussiana con i valori del fit UNBINNED
   y_val1 = func.gauss(x_val, mu_fit1, sigma_fit1) * N_events * bin_width


   # stampo l'istogramma e la funzione del modello
   fig, ax = plt.subplots()
   ax.hist(eventi, bins=bin_edges, range=(m, M), color='red', edgecolor='black', alpha=0.5,      label='Dati')
   ax.plot(x_val, y_val, color='black', lw=2, label='Fit gaussiano BINNED')
   ax.plot(x_val, y_val1, color='red', lw=1, label='Fit gaussiano UNBINNED')
   ax.legend()
   plt.show()
   
   
if __name__ == "__main__" :
   main ()   

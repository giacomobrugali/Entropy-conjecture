import numpy as np
import operazioni as op
import matplotlib.pyplot as plt
from math import floor, ceil
from iminuit import Minuit
from scipy.stats import expon, norm
from iminuit.cost import ExtendedBinnedNLL
from IPython.display import display


with open ('dati.txt') as f :
   
   eventi = [float (x) for x in f.readlines ()]
   
   
def mod_total (bin_edges, N_signal, mu, sigma, N_background, tau) :
   
   return N_signal * norm.cdf (bin_edges, mu, sigma)  + N_background * expon.cdf (bin_edges, 0, tau )

def mod_total_fit (bin_edges, N_signal, mu, sigma, N_background, tau, dx) :
   
   return N_signal * norm.pdf (bin_edges, mu, sigma) * dx + N_background * expon.pdf (bin_edges, 0, tau )* dx            
   
def main () :
   #definisco le variabili da dare a iminuit per iniziare il fit
   M = ceil (max (eventi))
   m = floor (min (eventi))
   MU = op.media ( eventi)
   DEV = op.deviazione_standard (eventi)
   
   # creo un istogramma numpy per inizializzare bin_content e bin_edges    
   N_bins = floor (len(eventi) / 100)
   x_range = (m, M)
   x_val = np.linspace (0, 20, 101)
   bin_content, bin_edges = np.histogram (eventi, bins = N_bins, range = x_range)
   N_events = sum(bin_content)
   
   #inizializzo la cost_function per il fit e lo eseguo
   cost_func = ExtendedBinnedNLL (bin_content, bin_edges, mod_total)
   my_minuit = Minuit (cost_func, N_signal = N_events, mu = MU, sigma = DEV, N_background = N_events, tau =1) #N_signal = N_events è fondamentale per eseguire il fit su ogni funzione
   my_minuit.migrad ()
   
   #stampo i valori ottenuti dal fit
   print (my_minuit.valid)
   display (my_minuit)
   print(my_minuit.covariance)
   print (my_minuit.valid)
   # formatted output
   for par, val, err in zip (my_minuit.parameters, my_minuit.values, my_minuit.errors) :
       print(f'{par} = {val:.3f} +/- {err:.3f}') 

   #estraggo i valori ottenuti dal fit
   N_signal_fit = my_minuit.values[0]
   mu_fit = my_minuit.values[1]  
   sigma_fit = my_minuit.values[2]
   N_background_fit = my_minuit.values[3] 
   tau_fit = my_minuit.values[4]
   
   #disegno la curva teorica
   dx = bin_edges[1] - bin_edges[0]
   g = mod_total_fit (x_val, N_signal_fit, mu_fit, sigma_fit, N_background_fit, tau_fit, dx)
   
   #stampo l'istogramma e la funzione del modello
   fig, ax = plt.subplots ()
   ax.plot (x_val, g, color = 'black', lw = 2)
   ax.hist (eventi, bins = bin_edges, range=(m, M), color = 'red', edgecolor = 'black', )
   plt.show ()
   
   
if __name__ == "__main__" :
   main ()

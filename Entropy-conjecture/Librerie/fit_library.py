import numpy as np
import operazioni as op  # libreria personalizzata con media e deviazione standard
import matplotlib.pyplot as plt
from math import floor, ceil
from iminuit import Minuit  # package per minimizzazione (fit)
from scipy.stats import expon, norm
from iminuit.cost import ExtendedBinnedNLL  # ML binned
from iminuit.cost import UnbinnedNLL        # ML unbinned
from IPython.display import display
from iminuit.cost import LeastSquares       # Minimi quadrati
import funzioni as func  # libreria personalizzata con gaussiane
from scipy.stats import chi2, norm


'''with open ('dati_2.txt') as f :             # apro il file contenente i dati
   eventi = [float (x) for x in f.readlines ()]  # leggo tutti i valori e li converto in float
   
M = ceil (max (eventi))                      # massimo dei dati arrotondato per eccesso
m = floor (min (eventi))                     # minimo dei dati arrotondato per difetto
MU = op.media(eventi)                         # media dei dati
DEV = op.deviazione_standard(eventi)         # deviazione standard dei dati
N_bins = floor(len(eventi) / 100)            # numero di bin per ML binned
x_range = (m, M)                             # range per l'istogramma
x_values = np.linspace(-5, 15, 20)           # vettore di riferimento (per LS o grafici)
x_i = []                                     # placeholder per i valori X
sigma_y = np.ones(20)                         # errori iniziali per LS'''


def BINNED_gauss(eventi):
   M = ceil (max (eventi))                      # massimo dei dati arrotondato per eccesso
   m = floor (min (eventi))                     # minimo dei dati arrotondato per difetto
   MU = op.media(eventi)                         # media dei dati
   DEV = op.deviazione_standard(eventi)         # deviazione standard dei dati
   N_bins = floor(len(eventi) / 100)            # numero di bin per ML binned
   x_range = (m, M)                             # range per l'istogramma
   x_values = np.linspace(-5, 15, 20)           # vettore di riferimento (per LS o grafici)
   x_i = []                                     # placeholder per i valori X
   sigma_y = np.ones(20)                         # errori iniziali per LS                                                               
   bin_content, bin_edges = np.histogram(eventi, bins=N_bins, range=x_range)  # histogram dei dati
   N_events = sum(bin_content)                                                  # numero totale eventi
   cost_func = ExtendedBinnedNLL(bin_content, bin_edges, func.gauss_binned)     # funzione di costo ML binned
   my_minuit = Minuit(cost_func, N_signal=N_events, mu=MU, sigma=DEV)          # inizializzo il fit
   my_minuit.migrad(ncall=10000)                                               # minimizzazione
   mu_fit = my_minuit.values[1]                                                # parametro mu stimato
   sigma_fit = my_minuit.values[2]                                             # parametro sigma stimato
   mu_sig = my_minuit.errors[1]                                                # errore su mu
   sigma_sig = my_minuit.errors[2]                                             # errore su sigma
   return my_minuit.valid, mu_fit, sigma_fit, mu_sig, sigma_sig                 # restituisce tuple di risultati
   
def UNBINNED_gauss(eventi):                                                             
   cost_func_unb = UnbinnedNLL(eventi, func.gauss)       # funzione di costo ML unbinned
   my_minuit1 = Minuit(cost_func_unb, mu=MU, sigma=DEV)  # inizializzazione fit
   my_minuit1.migrad(ncall=10000)                        # minimizzazione
   mu_fit1 = my_minuit1.values[0]                        # parametro mu stimato
   sigma_fit1 = my_minuit1.values[1]                     # parametro sigma stimato
   mu_sig1 = my_minuit1.errors[0]                        # errore su mu
   sigma_sig1 = my_minuit1.errors[1]                     # errore su sigma
   return my_minuit1.valid, mu_fit1, sigma_fit1, mu_sig1, sigma_sig1                 

def LeastSquares_gauss(eventi):
   N_bins = 50                                                     # scelgo numero di bin per LS
   bin_content, bin_edges = np.histogram(eventi, bins=N_bins, range=(min(eventi), max(eventi)))  # histogram
   x_i = 0.5 * (bin_edges[:-1] + bin_edges[1:])                   # centro dei bin
   y_i = bin_content                                               # valori Y
   sigma_y = np.sqrt(bin_content)                                   # errore sqrt(N) per LS
   ls = LeastSquares(x_i, y_i, sigma_y, func.gauss_binned)         # funzione di costo LS
   my_minuit = Minuit(ls, N_signal=sum(y_i), mu=np.mean(eventi), sigma=np.std(eventi))  # inizializzazione
   my_minuit.migrad()                                              # minimizzazione
   my_minuit.hesse()                                               # stima delle incertezze
   mu_fit = my_minuit.values['mu']                                  # mu stimato
   sigma_fit = my_minuit.values['sigma']                            # sigma stimato
   mu_sig = my_minuit.errors['mu']                                  # errore su mu
   sigma_sig = my_minuit.errors['sigma']                            # errore su sigma
   return my_minuit.valid, mu_fit, sigma_fit, mu_sig, sigma_sig      # ritorna i risultati

def main():
   N_eventi = np.logspace(np.log10(20), np.log10(10000), 20, dtype=int)  # array di dimensioni del campione
   mu_B, mu_UB, mu_LS = [], [], []                                        # liste per mu
   sigma_B, sigma_UB, sigma_LS = [], [], []                                # liste per sigma
   
   for N in N_eventi:
      sample = eventi[:N]                                 # prendo i primi N eventi
      B = BINNED_gauss(sample)                           # fit binned
      UB = UNBINNED_gauss(sample)                        # fit unbinned
      LS = LeastSquares_gauss(sample)                     # LS su dati binned
      mu_B.append(B[1])                                 # estraggo mu
      mu_UB.append(UB[1])
      mu_LS.append(LS[1])
      sigma_B.append(B[2])                               # estraggo sigma
      sigma_UB.append(UB[2])
      sigma_LS.append(LS[2])
   
   fig, ax = plt.subplots(1, 3, figsize=(15, 6))          # crea figura con 3 subplot
   
   # Grafico BINNED
   ax[0].plot(N_eventi, mu_B, color='black', lw=2, label=f'mu BINNED = {mu_B[-1]:.3f}')                  #:.3f formatta il numero con 3 cifre decimali.
   ax[0].plot(N_eventi, mu_B, color='black', lw=2, )                                                      #[-1] prende l’ultimo valore della lista (cioè l’ultimo fit eseguito).
   ax[0].plot(N_eventi, sigma_B, color='red', lw=2, label=f'sigma BINNED = {sigma_B[-1]:.3f}')
   # Grafico UNBINNED
   ax[1].plot(N_eventi, mu_UB, color='black', lw=2, label=f'mu UNBINNED = {mu_UB[-1]:.3f}')
   ax[1].plot(N_eventi, sigma_UB, color='red', lw=2, label=f'sigma UNBINNED = {sigma_UB[-1]:.3f}')
   # Grafico LeastSquares
   ax[2].plot(N_eventi, mu_LS, color='black', lw=2, label=f'mu LeastSquares = {mu_LS[-1]:.3f}')
   ax[2].plot(N_eventi, sigma_LS, color='red', lw=2, label=f'sigma LeastSquares = {sigma_LS[-1]:.3f}')
   
   ax[0].legend(loc='best')
   ax[1].legend(loc='best')
   ax[2].legend(loc='best')
   
   plt.show()

class Fitter:
    @staticmethod
    def gauss_pdf(x, N, mu, sigma):
        """Distribuzione Gaussiana Estesa per il fit."""
        return N * norm.pdf(x, mu, sigma)

    @staticmethod
    def generic_fit(data, model_func, initial_guesses, bins=50, x_range=None):
        """Esegue il fit e restituisce oggetto Minuit e statistiche."""
        counts, bin_edges = np.histogram(data, bins=bins, range=x_range)
        cost_func = ExtendedBinnedNLL(counts, bin_edges, model_func)
        
        my_minuit = Minuit(cost_func, **initial_guesses)
        my_minuit.migrad()
        my_minuit.hesse()
        
        # Calcolo statistiche di bontà del fit (Goodness of Fit)
        stats = Fitter.compute_stats(my_minuit, counts, bin_edges, model_func)
        
        return my_minuit, stats

    @staticmethod
    def compute_stats(minuit_obj, counts, bin_edges, model_func):
        """Calcola Chi-quadro, gradi di libertà e p-value."""
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # Valori attesi dal modello nei centri dei bin
        expected = model_func(bin_centers, *minuit_obj.values)
        
        # Chi-quadro di Pearson (usiamo solo bin con conteggi > 0)
        mask = expected > 0
        chi2_val = np.sum(((counts[mask] - expected[mask])**2) / expected[mask])
        
        # Gradi di libertà: Numero di bin - Numero di parametri fittati
        dof = len(counts[mask]) - len(minuit_obj.values)
        chi2_red = chi2_val / dof if dof > 0 else 0
        
        # P-value: probabilità di ottenere un chi2 maggiore o uguale a quello osservato
        p_value = chi2.sf(chi2_val, dof)
        
        return {
            "chi2": chi2_val,
            "dof": dof,
            "chi2_red": chi2_red,
            "p_value": p_value
        }
    







   
if __name__ == "__main__":
   main()


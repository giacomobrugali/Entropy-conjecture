import numpy as np
import operazioni as op  # libreria personalizzata con media e deviazione standard
from math import floor, ceil
from iminuit import Minuit  # package per minimizzazione (fit)rm
from iminuit.cost import ExtendedBinnedNLL  # ML binned
from iminuit.cost import UnbinnedNLL        # ML unbinned
from iminuit.cost import LeastSquares       # Minimi quadrati
import funzioni as func  # libreria personalizzata con gaussiane


def BINNED_gauss(eventi):        # UNRELIABLE
   MU = op.media(eventi)                               # media dei dati, valore iniziale per il fit
   DEV = op.deviazione_standard(eventi)               # deviazione standard dei dati, punto di partenza
   M = ceil(max(eventi))                               # limite superiore dell'intervallo dati
   m = floor(min(eventi))                              # limite inferiore dell'intervallo dati
   x_range = (m, M)                                   # range per l'istogramma
   N_bins = 50                 # numero di bin (circa 1 bin ogni 100 eventi)
   bin_content, bin_edges = np.histogram(eventi, bins=N_bins, range=x_range)  # istogramma dei dati
   N_events = sum(bin_content)                         # numero totale di eventi
   cost_func = ExtendedBinnedNLL(bin_content, bin_edges, func.gauss_binned)  # funzione di costo ML binned
   my_minuit = Minuit(cost_func, N_signal=N_events, mu=MU, sigma=DEV)       # inizializza Minuit
   my_minuit.migrad(ncall=10000)                       # esegue la minimizzazione
   mu_fit = my_minuit.values[1]                        # stima di mu
   sigma_fit = my_minuit.values[2]                     # stima di sigma
   mu_sig = my_minuit.errors[1]                        # errore su mu
   sigma_sig = my_minuit.errors[2]                     # errore su sigma
   Q_2 = my_minuit.fval
   return my_minuit.valid, mu_fit, sigma_fit, mu_sig, sigma_sig, Q_2  # restituisce tuple di risultati

def UNBINNED_gauss(eventi):   
   MU = op.media(eventi)                               # media dei dati
   DEV = op.deviazione_standard(eventi)               # deviazione standard dei dati
   cost_func_unb = UnbinnedNLL(eventi, func.gauss)    # funzione di costo ML non binnata
   my_minuit1 = Minuit(cost_func_unb, mu=MU, sigma=DEV)  # inizializzazione fit
   my_minuit1.migrad(ncall=10000)                     # minimizzazione
   mu_fit1 = my_minuit1.values[0]                     # stima mu
   sigma_fit1 = my_minuit1.values[1]                  # stima sigma
   mu_sig1 = my_minuit1.errors[0]                     # errore su mu
   sigma_sig1 = my_minuit1.errors[1]                  # errore su sigma
   Q_2 = my_minuit1.fval
   return my_minuit1.valid, mu_fit1, sigma_fit1, mu_sig1, sigma_sig1, Q_2  # tuple risultati

def LeastSquares_gauss(eventi):       # UNRELIABLE
   N_bins = 50
   bin_content, bin_edges = np.histogram(eventi, bins=N_bins, range=(min(eventi), max(eventi)))
   x_i = 0.5 * (bin_edges[:-1] + bin_edges[1:])   # centri bin
   y_i = bin_content
   sigma_y = np.sqrt(bin_content)
   sigma_y[sigma_y == 0] = 1.0   # evita divisioni per zero

   ls = LeastSquares(x_i, y_i, sigma_y, func.gauss_binned)
   my_minuit = Minuit(ls, N_signal=sum(y_i), mu=np.mean(eventi), sigma=np.std(eventi))
   my_minuit.migrad()
   my_minuit.hesse()

   mu_fit = my_minuit.values[0]
   sigma_fit = my_minuit.values[1]
   mu_sig = my_minuit.errors[0]
   sigma_sig = my_minuit.errors[1]
   Q_2 = my_minuit.fval
   return my_minuit.valid, mu_fit, sigma_fit, mu_sig, sigma_sig, Q_2

   
def BINNED (eventi, funzione):        # UNRELIABLE
   MU = op.media(eventi)                               # media dei dati, valore iniziale per il fit
   DEV = op.deviazione_standard(eventi)               # deviazione standard dei dati, punto di partenza
   M = ceil(max(eventi))                               # limite superiore dell'intervallo dati
   m = floor(min(eventi))                              # limite inferiore dell'intervallo dati
   x_range = (m, M)                                   # range per l'istogramma
   N_bins = 50                 # numero di bin (circa 1 bin ogni 100 eventi)
   bin_content, bin_edges = np.histogram(eventi, bins=N_bins, range=x_range)  # istogramma dei dati
   N_events = sum(bin_content)                         # numero totale di eventi
   cost_func = ExtendedBinnedNLL(bin_content, bin_edges, func.gauss_binned)  # funzione di costo ML binned
   my_minuit = Minuit(cost_func, N_signal=N_events, mu=MU, sigma=DEV)       # inizializza Minuit
   my_minuit.migrad(ncall=10000)                       # esegue la minimizzazione
   mu_fit = my_minuit.values[1]                        # stima di mu
   sigma_fit = my_minuit.values[2]                     # stima di sigma
   mu_sig = my_minuit.errors[1]                        # errore su mu
   sigma_sig = my_minuit.errors[2]                     # errore su sigma
   Q_2 = my_minuit.fval
   return my_minuit.valid, mu_fit, sigma_fit, mu_sig, sigma_sig, Q_2  # restituisce tuple di risultati 
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
  


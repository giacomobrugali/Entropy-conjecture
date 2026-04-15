# esame 01_09
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import operazioni as op
import PDFs as PDF
from iminuit import Minuit
from iminuit.cost import LeastSquares
from iminuit.cost import ExtendedBinnedNLL  # ML binned
from iminuit.cost import UnbinnedNLL        # ML unbinned
from IPython.display import display
from iminuit.cost import LeastSquares       # Minimi quadrati
import funzioni as func  # libreria personalizzata con gaussiane
from math import floor, ceil



# 1-definizione della doppia gaussiana
def double_gauss (x,y, m_1, m_2, s_1, s_2) :
    return (1/(s_1*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-m_1)/s_1)**2) *(1/(s_2*np.sqrt(2*np.pi))) * np.exp(-0.5*((y-m_2)/s_2)**2)

# 2-definizione dell'algoritmo di metropolis
def metropolis (N_punti) :
    m_1 = 1
    m_2 = 2
    s_1 = 0.5
    s_2 = 1.2
    X_0 = 0
    Y_0 = 0
    delta = 2
    X_val = []
    Y_val = []
    for i in range(N_punti):
       X_1 = np.random.uniform(X_0 - delta, X_0 + delta)
       Y_1 = np.random.uniform(Y_0 - delta, Y_0 + delta)
       alpha = double_gauss (X_1,Y_1, m_1, m_2, s_1, s_2) / double_gauss (X_0,Y_0, m_1, m_2, s_1, s_2)
       u = np.random.uniform (0,1)
       if alpha > u :
           X_0 = X_1
           Y_0 = Y_1
           X_val.append(X_1)
           Y_val.append(Y_1)
       else :
           X_val.append(X_0)
           Y_val.append(Y_0)
    return X_val, Y_val

# 3 istogramma  
def hist (X, mu, sigma) :
    X_m = np.linspace(min(X), max(X), 10000)
    Y_m = PDF.pdf_gauss_arr(X_m, mu, sigma)
    N_bins = op.sturges ( len(X))
    x_range = (min(X), max(X))
    bin_content, bin_edges = np.histogram (X, bins = N_bins, range = x_range)
    fig, ax = plt.subplots ()
    ax.plot (X_m, Y_m, color = 'blue', lw = 2)
    ax.hist (X, bins = bin_edges, range=x_range, color = 'red', edgecolor = 'black', density=True)
    plt.show ()

#4 fit gaussiano dei dati
def fit_gauss (X) :
    M = ceil (max (X))                      # massimo dei dati arrotondato per eccesso
    m = floor (min (X))                     # minimo dei dati arrotondato per difetto
    MU = op.media(X)                         # media dei dati
    DEV = op.deviazione_standard(X)         # deviazione standard dei dati
    N_bins = op.sturges(len(X))            # numero di bin per ML binned
    x_range = (m, M)                             # range per l'istogramma
    x_values = np.linspace(-5, 15, 20)           # vettore di riferimento (per LS o grafici)
    x_i = []                                     # placeholder per i valori X
    sigma_y = np.ones(len(X))                         # errori iniziali per LS
    bin_content, bin_edges = np.histogram(X, bins=N_bins, range=x_range)  # histogram dei dati
    N_events = sum(bin_content)                                                  # numero totale eventi
    cost_func = ExtendedBinnedNLL(bin_content, bin_edges, func.gauss_binned)     # funzione di costo ML binned
    my_minuit = Minuit(cost_func, N_signal=N_events, mu=MU, sigma=DEV)          # inizializzo il fit
    my_minuit.migrad(ncall=10000)                                               # minimizzazione
    mu_fit = my_minuit.values[1]                                                # parametro mu stimato
    sigma_fit = my_minuit.values[2]                                             # parametro sigma stimato
    mu_sig = my_minuit.errors[1]                                                # errore su mu
    sigma_sig = my_minuit.errors[2]                                             # errore su sigma
    return my_minuit.valid, mu_fit, sigma_fit, mu_sig, sigma_sig,my_minuit.covariance                 # restituisce tuple di risultati
  
#4-bis compatibilità dati
def compat (x_1, x_2, s):
    t = abs (x_1 - x_2) /s
    if t < 3 :
        return True
    else :
        return False

def main () :
    mu_x = 1
    mu_y = 2
    sigma_x = 0.5
    sigma_y = 1.2
    X_val, Y_val = metropolis (10000)
    hist (X_val, 1, 0.5)
    hist (Y_val, 2, 1.2)
    F= fit_gauss (X_val)
    F_1= fit_gauss (Y_val)
    print ('media x fit = ', F[1], 'sigma x fit =', F[2])
    print ('media y fit = ', F_1[1], 'sigma y fit =', F_1[2])
    if compat (mu_x, F[1],F[3]) == True :
        print('mu_x è compatibile con il valore tabulato')
    else :
        print ( 'mu_x NON è compatibile con il valore tabulato')    

    if compat (mu_y, F_1[1],F_1[3]) == True :
        print('mu_y è compatibile con il valore tabulato')
    else :
        print ( 'mu_y NON è compatibile con il valore tabulato')    

    if compat (sigma_x, F[2],F[4]) == True :
        print('sigma_x è compatibile con il valore tabulato')
    else :
        print ( 'sigma_x NON è compatibile con il valore tabulato')    

    if compat (sigma_y, F_1[2],F_1[4]) == True :
        print('sigma_y è compatibile con il valore tabulato')
    else :
        print ( 'sigma_y NON è compatibile con il valore tabulato')  
    print (F[5])
    print (F_1[5])

if __name__ == "__main__":
    main()










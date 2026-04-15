import numpy as np
import matplotlib.pyplot as plt
import integrazione as it
import myrand as mr
import random 
import operazioni as op
import PDFs as pdf
from math import floor, ceil
from iminuit import Minuit  
from scipy.stats import expon, norm
from iminuit.cost import ExtendedBinnedNLL 
import funzioni as func  

def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)



def generate_uniform (seed = 0.) :
    
    if seed != 0. : random.seed (float (seed))
    return random.random ()


def rand_gauss_TAC (mean, sigma, xMin, xMax) :
    
    x = rand_range (xMin, xMax)
    yMax = 1/(sigma * np.sqrt(2 * np.pi))
    y = rand_range (0, yMax)
    while (y > pdf.pdf_gauss(x, mean,sigma)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x


def generate_gauss_TAC (mean, sigma, xMin, xMax, N, seed = 0.) :
    
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_gauss_TAC (mean, sigma, xMin, xMax))
    return randlist



def pdf_fondo (x, min,max) :
    if x <= max and x >= min :
        return 0.5 * np.sin (x)
    else :
        return 0

def pdf_fondo_vett (X, min,max) :
    valori = []
    for x in X : 
        if x <= max and x >= min :
            v = 0.5 * np.sin (x)
            valori.append(v)
        else :
            valori.append(0)
    return valori

def inverse_pdf_fondo (N) :
    values = []
    for n in range (N) :
        u = random.random ()
        x = np.arccos(1 - (2 * u))
        values.append (x)
    return values


def BINNED_gauss(eventi,max,min):
   M = max                      # massimo dei dati arrotondato per eccesso
   m = min                     # minimo dei dati arrotondato per difetto
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








def main () :
    
# 2 - grafico e integrale HoM
    min = 0
    max = np.pi
    x_val = np.linspace (-1,4,1000)
    y_val = pdf_fondo_vett (x_val, min,max)

    fig, ax = plt.subplots ()
    ax.plot (x_val, y_val, color = 'green', lw = 2)
    plt.show ()


    funzione = lambda x : pdf_fondo(x,0,np.pi)
    I, I_sigma = it.integrale_HoM(funzione, 3.2, 0, 0.5, 0, 1000)
    print ('il valore dell integrale è :', I,'±', I_sigma)

# 3 - inverse function
    rand_values = inverse_pdf_fondo (2000)
    x_val_1 = np.linspace (0,1,2000)
    
    N_punti = len(rand_values)
    
    #n_bins = int(np.sqrt(N_punti))                                                      
    n_bins = op.sturges(len(rand_values))                                              
    x_range = (np.min(rand_values), np.max(rand_values))                                              
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conteggi, bordi, patches = ax.hist(rand_values, 
                                       bins=n_bins, 
                                       range=x_range, 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.7,
                                       density=True, 
                                       label=f'Dati ($N$={N_punti})') 

    
    ax.set_title('Distribuzione dei dati generati')
    ax.set_xlabel('Valore')
    ax.set_ylabel('Densità di probabilità') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)                                                          
    ax.grid(axis='x', alpha=0.3)                                                           
    plt.show()
    
# 4 - TaC
    mean = (np.pi) / 4
    sigma = (np.pi) / 20
    xMin = mean -3 * sigma
    xMax = mean + 3 * sigma
    values_2 = generate_gauss_TAC (mean, sigma, xMin, xMax, 1000, seed = 0.)
    
    valori_1 = np.asarray(rand_values)
    valori_2 = np.asarray(values_2)
    values_tot = np.concatenate((valori_1, valori_2))

    N_punti = len(values_tot)
    
    n_bins = int(np.sqrt(N_punti))                                                      
    #n_bins = op.sturges(len(values_tot))                                              
    x_range = (np.min(values_tot), np.max(values_tot))                                              
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conteggi, bordi, patches = ax.hist(values_tot, 
                                       bins=n_bins, 
                                       range=x_range, 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.7,
                                       density=True, 
                                       label=f'Dati ($N$={N_punti})') 

    
    ax.set_title('Distribuzione dei dati generati')
    ax.set_xlabel('Valore')
    ax.set_ylabel('Densità di probabilità') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)                                                          
    ax.grid(axis='x', alpha=0.3)                                                           
    plt.show()

# 5 - toy experiment e bias
    n_toy = 10
    mu = []

    for n in range (n_toy) :
        mean = (np.pi) / 4
        sigma = (np.pi) / 20
        xMin = mean -3 * sigma
        xMax = mean + 3 * sigma
        rand_values = inverse_pdf_fondo (2000)
        values_2 = generate_gauss_TAC (mean, sigma, xMin, xMax, 1000, seed = 0.)
    
        valori_1 = np.asarray(rand_values)
        valori_2 = np.asarray(values_2)
        values_tot = np.concatenate((valori_1, valori_2))
        valid, mu_fit, sigma_fit, mu_sig, sigma_sig = BINNED_gauss(values_tot, (np.pi / 2), 0)
        if valid:
            mu.append(mu_fit)
    
    media_exp = op.media (mu)
    bias = abs (media_exp - (np.pi / 4))

    print ('il bias è', bias)
    


















if __name__ == "__main__" :
    main ()


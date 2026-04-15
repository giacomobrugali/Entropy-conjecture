# 12_01_2026

import numpy as np
import matplotlib.pyplot as plt
import random 
import operazioni as op
import pandas as pd


def inv_exp (y, lamb = 1) :
   
    return -1 * np.log (1-y) / lamb


def rand_exp (tau) :
    
    lamb = 1. / tau
    return inv_exp (random.random (), lamb)


def generate_exp (tau, N, seed = 0.) :
    
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_exp (tau))
    return randlist


def cammino_medio (E,tau) :
    L = []
    while E > 0 :
        l = rand_exp (tau)
        if E < 0.01 :
            E -= 15 * l
        elif 0.01 <= E <= 0.5 :
            E -= (0.15 / E) * l
        elif E > 0.5 :
            E -= 0.3 * l

        L.append (l)

    l_tot = np.sum(L)
    return l_tot

def cammino_spessore_vett (tau,spessore,N) :
    survivors = 0
    energy = []
    for n in range (N) :
        L = 0
        E = 1
        while L <= spessore and E > 0 :
            l = rand_exp (tau)
            if L + l > spessore:
                l_eff = spessore - L 
            else:
                l_eff = l             
            if E < 0.01 :
                E -= 15 * l
            elif 0.01 <= E <= 0.5 :
                E -= (0.15/E) *l
            elif E > 0.5 :
                E -= 0.3 * l
            L += l
        if L > spessore and E > 0:
            survivors += 1
            energy.append (E)

    return survivors, energy


def cammino_medio_vett(e, tau, N):
    L_TOT = []
    for i in range(N):
        l_tot = 0
        E = e
        while E > 0:
            l = rand_exp(tau)
            
            # 1. Calcoliamo PRIMA quanto perderebbe in questo passo
            loss = 0
            if E < 0.01:
                loss = 15 * l
            elif 0.01 <= E <= 0.5:
                loss = (0.15 / E) * l
            elif E > 0.5:
                loss = 0.3 * l
            
            # 2. Controllo: Ha abbastanza energia per fare tutto il passo?
            if loss > E:
                # NO: Si ferma durante il passo. 
                # Calcoliamo quanta frazione di 'l' riesce a percorrere.
                frazione_percorsa = E / loss
                l_tot += l * frazione_percorsa
                E = 0 # L'energia è finita
            else:
                # SI: Completa il passo e sottrae l'energia
                E -= loss
                l_tot += l
                
        L_TOT.append(l_tot)

    return L_TOT
def cammino_medio_numpy(e, tau, N):
    # 1. PRE-CALCOLO: Contiamo quanti passi servono a una particella per fermarsi
    # Dato che la perdita di energia è fissa, lo facciamo una volta sola.
    E_temp = e
    num_passi = 0
    
    while E_temp > 0:
        num_passi += 1
        # La tua logica di perdita energia
        if E_temp < 0.01:
            E_temp -= 15
        elif 0.01 <= E_temp <= 0.5:
            E_temp -= 0.15
        elif E_temp > 0.5:
            E_temp -= 0.3
            
    # 2. GENERAZIONE VETTORIALE
    # Creiamo una matrice con N righe (particelle) e 'num_passi' colonne
    # Ogni cella contiene un passo casuale estratto dall'esponenziale
    passi = np.random.exponential(scale=tau, size=(N, num_passi))
    
    # 3. SOMMA SUGLI ASSI
    # Sommiamo lungo l'asse 1 (le colonne) per ottenere il totale per ogni riga (particella)
    L_TOT = np.sum(passi, axis=1)
    
    return L_TOT




def istogramma(dati):
    N_punti = len(dati)
    
    # 1. Calcolo parametri
    n_bins = int(np.sqrt(N_punti))                                                       # Regola della radice
    '''n_bins = op.sturges(len(dati))'''                                             # regola di sturges
    x_range = (np.min(dati), np.max(dati))                                               # Range automatico sui dati
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 2. Costruzione istogramma
    # density=True normalizza l'area a 1 (utile per confrontare con la PDF)
    # edgecolor definisce i bordi delle colonne per una migliore leggibilità
    conteggi, bordi, patches = ax.hist(dati, 
                                       bins=n_bins, 
                                       range=x_range, 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.7, 
                                       label=f'Dati ($N$={N_punti})') 

    # 3. Formattazione
    ax.set_title('Distribuzione dei dati generati')
    ax.set_xlabel('Valore')
    ax.set_ylabel('Densità di probabilità') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)                                                           # mette una griglia perpendicolare all'asse Y
    ax.grid(axis='x', alpha=0.3)                                                           # mette una griglia perpendicolare all'asse X
    plt.show()
    return conteggi, bordi
    

def main () :

#1 - 10.000 cammini medi
    tau = 10**(-3)
    E = 1
    N = 10000
    L_T = []    
    l = cammino_medio_vett (E,tau,N)
    #hist = istogramma(l)

#2 - distanza di un elettrone
    L_e = cammino_medio (E,tau)
    print ('la distanza percorsa dall elettrone è :' ,L_e)

# 3 - media, sigma, skewness, kurtosis
    L_1 = cammino_medio_vett (E,tau,N)
    Mean = op.media (L_1)
    Dev = op.deviazione_standard(L_1)
    Skewness = op.skewness(L_1)
    Kurtosis = op.kurtosis(L_1)

    '''print ('la media della distribuzione è :', Mean)
    print ('la deviazione standard della distribuzione è :', Dev)
    print ('la asimmetria della distribuzione è :', Skewness)
    print ('la curtosi della distribuzione è :', Kurtosis)'''

    print("-" * 30) # Stampa una linea separatrice
    print(f"{'STATISTICA':<20} | {'VALORE':>10}")
    print("-" * 30)
    print(f"{'Media':<20} | {Mean:>10.4f}")
    print(f"{'Deviazione Std':<20} | {Dev:>10.4f}")
    print(f"{'Skewness':<20} | {Skewness:>10.4f}")
    print(f"{'Kurtosis':<20} | {Kurtosis:>10.4f}")
    print("-" * 30)

# 4 - profondità
    spessore = Mean
    N = 10000
    survivors, energy = cammino_spessore_vett (tau,spessore,N)
    fraction = survivors / N
    istogramma (energy)

# 5 - left energy profile
    N_e = 100
    valori_mu = np.linspace (0,5,100)
    E_rilasciata = []
    for mu in valori_mu :
        survivors_1, energy_1 = cammino_spessore_vett (tau,mu,N_e)
        e_left = (N_e - survivors_1) + (100 - np.sum(energy_1))
        E_rilasciata.append (e_left)

    istogramma(E_rilasciata)
        


    








if __name__ == "__main__" :
    main ()

        
        





import numpy as np
import matplotlib.pyplot as plt
import myrand as mr
import operazioni as op

def generate_gauss_bm () :
    x1 = mr.generate_uniform (1)
    x2 = mr.generate_uniform (1)
    g1 = np.sqrt((-2)* np.log(x1)) * np.cos(2 * np.pi * x2)
    g2 = np.sqrt((-2)* np.log(x1)) * np.sin(2 * np.pi * x2)
    return g1, g2


def generate_gauss_bm_2 () :
    x1 = mr.generate_uniform (1)
    x2 = mr.generate_uniform (1)
    g1 = (np.sqrt((-2)* np.log(x1)) * np.cos(2 * np.pi * x2) ) + 5
    g2 = (np.sqrt((-2)* np.log(x1)) * np.sin(2 * np.pi * x2) ) + 5
    return g1, g2


def generate_gauss_bm_vett (N) :
    x1 = mr.generate_uniform (N)
    x2 = mr.generate_uniform (N)
    X1 = np.asarray(x1)
    X2 = np.asarray(x2)
    g1 = np.sqrt((-2)* np.log(X1)) * np.cos(2 * np.pi * X2)
    g2 = np.sqrt((-2)* np.log(X1)) * np.sin(2 * np.pi * X2)
    return g1, g2


def generate_gauss_bm_vett_2 (N, mu, sigma) :
    x1 = mr.generate_uniform (N)
    x2 = mr.generate_uniform (N)
    X1 = np.asarray(x1)
    X2 = np.asarray(x2)
    g1 = ((np.sqrt((-2)* np.log(X1)) * np.cos(2 * np.pi * X2) ) * sigma) + mu
    g2 = ((np.sqrt((-2)* np.log(X1)) * np.sin(2 * np.pi * X2) ) * sigma) + mu
    return g1, g2



def istogramma(dati):
    N_punti = len(dati)
    
    # 1. Calcolo parametri
    n_bins = int(np.sqrt(N_punti))                                                       # Regola della radice
    '''n_bins = op.sturges(len(dati))'''                                                 # regola di sturges
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

# 2 - generazione 1000 dati
    N = 500
    mu = 5
    sigma = 2
    g1, g2 = generate_gauss_bm_vett_2 (N, mu, sigma)
    g_tot = np.concatenate((g1 , g2))
    #istogramma ( g_tot)

# 3 - media, varianza e errori
    Mean = op.media (g_tot)
    Variance = op.varianza(g_tot)
    err_mean = op.dev_stndrd_mean (g_tot)
    err_variance = op.errore_varianza(g_tot)
    Dev = op.deviazione_standard(g_tot)

    print ('il valore della media è :', Mean, '±',err_mean)
    print ('il valore della deviazione standard è :', Dev, '±',err_mean)
    print ('il valore della varianza è :', Variance, '±',err_variance)
    

# 4 - variazione 
    '''N_1 = np.linspace (100,1000, 100)
    sigma_1 = []
    err_mean_1 = []
    for n_1 in N_1 :
        n1 = int(n_1)
        g1, g2 = generate_gauss_bm_vett (n1)
        g_tot = np.concatenate((g1 , g2))
        sig_1 = op.deviazione_standard (g_tot)
        err_M_1 = op.dev_stndrd_mean (g_tot)
        sigma_1.append (sig_1)
        err_mean_1.append(err_M_1)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))                                          

                  
    ax[0].plot(N_1 , sigma_1, label='Sigma', color='red')                                 

    ax[0].set_title('variazione della deviazione standard')                                          

    ax[0].legend()                                                                                

    ax[1].plot(N_1, err_mean_1, label='errore sulla media', color='blue')  

    ax[1].set_title('variazione dell errore sulla media')   

    ax[1].legend()                            

    plt.tight_layout()                                                                                 
    plt.show()'''

# 5 - variazione della funzione
    

   









if __name__ == "__main__" :
    main ()

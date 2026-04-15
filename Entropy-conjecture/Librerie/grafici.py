import matplotlib.pyplot as plt
import numpy as np


# ESEMPIO DI UNA FIGURA CON 4 GRAFICI

'''
# --- Plotting ---
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))                                          # creo una figura con 4 grafici inserendo 2 righe e 2 colorre
    
    # Grafico Media
                  
    ax[0,0].plot(DATI , func(DATI), label='Log-L Media', color='red')                                  # [0,0] è il grafico in alto a sx

    ax[1,0].plot(DATI , func(DATI), label='LLR Media', color='blue')                                   # [1,0] è il grafico in basso a sx

    ax[0,0].axvline(x = X, color='darkred', linestyle='--', label=f'Max MLE: {mu_fit:.2f}')            # creo una linea verticale in corrispondenza del valore mu_fit

    ax[1,0].scatter([X_1, X_2], [Y1, Y_2], color='black', marker='o', zorder=5)                        # disegno 2 punti in nero, devo passare a destra le coord. X di tutti i punti e poi le Y

    ax[1,0].axhline(y=-0.5, color='gray', linestyle='--', label=r'$L_{max} - 0.5$')                    # creo una linea orizzontale in corrispondenza del valore 0.5

    ax[0,0].set_title(r'Scansione e Fit della Media ($\\mu$)')                                          # inserisco il titolo, inserisce automaticamente il valore di mu

    ax[1,0].set_xlim(x_min, x_max)                                                                     # limito la visualizzazione del grafico tra x_1 = mu_fit - 1 e x_2 = mu_fit + 1

    ax[1,0].set_ylim(y_min , y_max)                                                                    # limito la visualizzazione del grafico tra y_1 = - 1 e y_2 = + 1

    ax[0,0].legend()                                                                                   # visualizzo la legenda del grafico [0,0]

    ax[1,0].legend()                                                                                   # visualizzo la legenda del grafico [0,0]

    ax[0,1].plot(sigma_possibili, LS, label='Log-L Sigma', color='red')                                # [0,1] è il grafico in alto a dx

    ax[1,1].plot(sigma_possibili, llr_sigma, label='LLR Sigma', color='blue')                          # [0,0] è il grafico in basso a dx

    
    plt.tight_layout()                                                                                 # comando per organizzare meglio i grafici 
    plt.show()'''

    



# ESEMPIO DI UN GRAFICO CON BARRE DI ERRORE
'''
    
    ax.errorbar(x_coord, y_coord, yerr=sigma_y, fmt='o', label='Dati simulati', color='black')      # grafico di un set di punti con le relative barre di incertezze 
    
    ax.plot(x_coord, func(x_coord, m_fit, q_fit), label='Fit risultante', color='red', linewidth=2) # grafico della funzione interpolante dei punti'''


    


# ESEMPIO ISTOGRAMMA

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






#COME FARE UN FIT DELL'HISTOGRAMMA

'''
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares


counts, edges = istogramma(dati, bins=n_bins, range=x_range)                          # 1. Genera i conteggi e i bordi (e ne disegnare il grafico)


x_centers = (edges[:-1] + edges[1:]) / 2.                                             # 2. Calcola i centri dei bin (le nostre xi)


y_values = counts                                                                     # 3. Definisci le yi e le loro incertezze (statistica di Poisson)
y_errors = np.sqrt(counts)


mask = y_values > 0                                                                   # ATTENZIONE: Se un bin ha 0 conteggi, l'errore sqrt(0)=0 darà errore nel fit.
x_fit = x_centers[mask]                                                               # Bisogna filtrare solo i bin con almeno un evento.
y_fit = y_values[mask]                                                                # questa tecnica è il BOOLEAN MASKING
err_fit = y_errors[mask]

bin_width = edges[1] - edges[0]                                                       # calcolo la larghezza dei bin, fattore di scala per il fit

F = lambda x,A,B,m,s,t : bin_width * modello_funzione

least_squares = LeastSquares(x_fit, y_fit, err_fit, F)                                # 4. Applica il LeastSquares
my_minuit = Minuit(least_squares, ...)
my_minuit.migrad()

DoF = my_minuit.values['dof']
print(DoF)

x_q = np.linspace(1,1000,len(Q2))
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.plot(x_fit, y_fit, label='chi squared', color='blue', linewidth=2)                 # disegno inizialmente la funzione chi2 che passa per i punti dell'istogramma
plt.show ()

N_punti = len(Q2) 
x_sovrapp = np.linspace (np.min(Q2), np.max(Q2), len(Q2))                             # creo le X per la sovrapposizione teorica della funzione al grafico dell'istogramma
y_sovrapp = bin_width * modello_funzione(x_sovrapp, DoF)                              # devo scalare anche le y per la larghezza di un bin
fig, ax = plt.subplots(figsize=(8, 5))
   
ax.hist(Q2, 
                                       bins=int(np.sqrt(N_punti)), 
                                       range=(np.min(Q2), np.max(Q2)), 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.7, 
                                       label=f'Dati ($N$={N_punti})',
                                       density=True) 

ax.plot (x_sovrapp, y_sovrapp, color = 'red')

ax.set_title('Distribuzione dei dati generati')
ax.set_xlabel('Valore')
ax.set_ylabel('Densità di probabilità') 
ax.legend()
ax.grid(axis='y', alpha=0.3)  
plt.show()
'''









# COME FORMATTARE UNA STAMPA A SCHERMO

'metodo 1 TABELLA'
'''
    print("-" * 30) # Stampa una linea separatrice
    print(f"{'STATISTICA':<20} | {'VALORE':>10}")
    print("-" * 30)
    print(f"{'Media':<20} | {Mean:>10.4f}")
    print(f"{'Deviazione Std':<20} | {Dev:>10.4f}")
    print(f"{'Skewness':<20} | {Skewness:>10.4f}")
    print(f"{'Kurtosis':<20} | {Kurtosis:>10.4f}")
    print("-" * 30)'''


'metodo 2 DIZIONARIO'
'''
    stats = {
    "Media": Mean,
    "Deviazione Std": Dev,
    "Skewness": Skewness,
    "Kurtosis": Kurtosis
}

    for chiave, valore in stats.items():
        print(f"{chiave}: {valore:.4f}")'''


'metodo 3 PANDAS'
'''
    import pandas as pd

# Creiamo un DataFrame (una tabella)
    df_stats = pd.DataFrame({
    'Metrica': ['Media', 'Deviazione Std', 'Skewness', 'Kurtosis'],
    'Valore': [Mean, Dev, Skewness, Kurtosis]
    })

# Stampiamo senza l'indice numerico a sinistra per pulizia
    print(df_stats.to_string(index=False))'''





# COME PLOTTARE PIU CURVE SULLO STESSO GRAFICO
'''# Estendiamo x a 25 per vedere bene la coda delle curve con lambda alti
    x_val = np.arange(0, 25, 1)
    
    # Definiamo i lambda da testare (da 0 a 10)
    lambdas = np.arange(0, 11, 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- TRUCCO PER I COLORI ---
    # Generiamo automaticamente 11 colori diversi dalla mappa 'jet' (o 'viridis')
    colors = plt.cm.jet(np.linspace(0, 1, len(lambdas)))

    # --- IL CICLO FOR ---
    # enumerate ci dà sia l'indice (i) per il colore, sia il valore (lam)
    for i, lam in enumerate(lambdas):
        y_val = poisson_vett(x_val, lam)
        
        # Plot automatico
        ax.plot(x_val, y_val, color=colors[i], label=f'λ = {lam}')

    # Aggiungiamo la legenda per capire quale colore corrisponde a quale lambda
    ax.legend()
    ax.set_title("Distribuzione di Poisson al variare di Lambda")
    ax.grid(True, alpha=0.3)

    plt.show()'''





# CALCOLO BINS OTTIMIZZATO PER DISCRETI
'''# Creiamo bin centrati sugli interi: [-0.5, 0.5, 1.5, 2.5 ...]
# In questo modo la barra dello "0" sta tra -0.5 e 0.5, quella dell'"1" tra 0.5 e 1.5, ecc.
min_val = np.min(N_events)
max_val = np.max(N_events)
bins_discreti = np.arange(min_val - 0.5, max_val + 1.5, 1)'''
import matplotlib.pyplot as plt
import numpy as np
import operazioni as op

'''ESEMPIO DI UNA FIGURA CON 4 GRAFICI


# --- Plotting ---
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))                                          # creo una figura con 4 grafici inserendo 2 righe e 2 colorre
    
    # Grafico Media
                  
    ax[0,0].plot(DATI , func(DATI), label='Log-L Media', color='red')                                  # [0,0] è il grafico in alto a sx

    ax[1,0].plot(DATI , func(DATI), label='LLR Media', color='blue')                                   # [1,0] è il grafico in basso a sx

    ax[0,0].axvline(x = X, color='darkred', linestyle='--', label=f'Max MLE: {mu_fit:.2f}')            # creo una linea verticale in corrispondenza del valore mu_fit

    ax[1,0].scatter([X_1, X_2], [Y1, Y_2], color='black', marker='o', zorder=5)                        # disegno 2 punti in nero, devo passare a destra le coord. X di tutti i punti e poi le Y

    ax[1,0].axhline(y=-0.5, color='gray', linestyle='--', label=r'$L_{max} - 0.5$')                    # creo una linea orizzontale in corrispondenza del valore 0.5

    ax[0,0].set_title(r'Scansione e Fit della Media ($\mu$)')                                          # inserisco il titolo, inserisce automaticamente il valore di mu

    ax[1,0].set_xlim(x_min, x_max)                                                                     # limito la visualizzazione del grafico tra x_1 = mu_fit - 1 e x_2 = mu_fit + 1

    ax[1,0].set_ylim(y_min , y_max)                                                                    # limito la visualizzazione del grafico tra y_1 = - 1 e y_2 = + 1

    ax[0,0].legend()                                                                                   # visualizzo la legenda del grafico [0,0]

    ax[1,0].legend()                                                                                   # visualizzo la legenda del grafico [0,0]

    ax[0,1].plot(sigma_possibili, LS, label='Log-L Sigma', color='red')                                # [0,1] è il grafico in alto a dx

    ax[1,1].plot(sigma_possibili, llr_sigma, label='LLR Sigma', color='blue')                          # [0,0] è il grafico in basso a dx

    
    plt.tight_layout()                                                                                 # comando per organizzare meglio i grafici 
    plt.show()

    
ESEMPIO DI UN GRAFICO CON BARRE DI ERRORE
     
    ax.errorbar(x_coord, y_coord, yerr=sigma_y, fmt='o', label='Dati simulati', color='black')      # grafico di un set di punti con le relative barre di incertezze 
    
    ax.plot(x_coord, func(x_coord, m_fit, q_fit), label='Fit risultante', color='red', linewidth=2) # grafico della funzione interpolante dei punti


ESEMPIO ISTOGRAMMA'''

def istogramma(dati):
    N_punti = len(dati)
    
    # 1. Calcolo parametri
    n_bins = op.sturges(len(dati))                          
    x_range = (np.min(dati), np.max(dati))                        # Range automatico sui dati
    
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
                                       label=f'Dati ($N$={N_punti})',
                                       density=True) 

    # 3. Formattazione
    ax.set_title('Distribuzione dei dati generati')
    ax.set_xlabel('Valore')
    ax.set_ylabel('Densità di probabilità') # Se density=True
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.show()
    return conteggi, bordi
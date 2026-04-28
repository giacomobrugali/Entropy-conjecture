import numpy as np
import myrand as mr
import operazioni as op

   
def integrale_HoM(funzione, X_max, X_min, Y_max, Y_min, N_eve):
    x_val = mr.generate_range(X_min, X_max, N_eve)            # numeri casuali x uniformi tra X_min e X_max
    y_val = mr.generate_range(Y_min, Y_max, N_eve)            # numeri casuali y uniformi tra Y_min e Y_max
    count = 0                                                 # contatore punti sotto la curva
    for x, y in zip(x_val, y_val):                            # scorre coppie (x,y)
        if funzione(x) > y:                                   # verifica se il punto è sotto la curva
            count = count + 1                                 # incrementa il contatore se sotto la curva
    A = (X_max - X_min) * (Y_max - Y_min)                    # area del rettangolo contenente la funzione
    n = float(count) / N_eve                                  # frazione di punti sotto la curva
    I = A * n                                                 # stima dell’integrale
    I_sigma = A * np.sqrt(n * (1 - n) / N_eve)               # errore statistico dell’integrale
    return I, I_sigma                                         # restituisce integrale e incertezza


def integral_crude_MC(funzione, M, m, N_eve):
    x_val = mr.generate_range(m, M, N_eve)                   # numeri casuali x uniformi tra m e M
    G = []                                                    # lista valori funzione
    for x in x_val:                                           # scorre tutti i punti x
        gx = funzione(x)                                      # calcola funzione in x
        G.append(gx)                                          # aggiunge valore alla lista
    mean = op.media(G)                                        # media dei valori della funzione
    dvm = op.dev_stndrd_mean(G)                               # deviazione standard della media
    I = (M - m) * mean                                        # stima dell’integrale
    I_sigma = (M - m) * dvm                                   # errore statistico dell’integrale
    return I, I_sigma                                         # restituisce integrale e incertezza

import numpy as np

def integral_HoM(funzione, X_max, X_min, Y_max, Y_min, N_eve):
    # Generiamo i numeri direttamente con numpy (più veloce del tuo modulo mr)
    x_val = np.random.uniform(X_min, X_max, N_eve)
    y_val = np.random.uniform(Y_min, Y_max, N_eve)
    
    # VETTORIZZAZIONE:
    # Calcoliamo la funzione su tutte le x in un colpo solo
    y_teorici = funzione(x_val)
    
    # Creiamo una maschera booleana (True se sotto, False se sopra)
    # Nota: y < f(x) è la condizione standard per l'area positiva
    punti_sotto = y_val < y_teorici 
    
    # Contiamo i True
    count = np.sum(punti_sotto)
    
    # Calcoli finali 
    A = (X_max - X_min) * (Y_max - Y_min)
    n = count / N_eve
    I = A * n
    I_sigma = A * np.sqrt(n * (1 - n) / N_eve)
    
    return I, I_sigma, x_val, y_val, punti_sotto # Restituisco anche i punti per il grafico




def grafico_integrazione(x_val, y_val, punti_sotto, funzione, x_min, x_max):
    plt.figure(figsize=(10, 6))
    
    # A. Separiamo i punti usando la maschera booleana
    # punti_sotto è un array di True/False. 
    # Usando ~punti_sotto (la tilde inverte) otteniamo i punti sopra.
    
    x_in = x_val[punti_sotto]      # Punti SOTTO (Hit)
    y_in = y_val[punti_sotto]
    
    x_out = x_val[~punti_sotto]    # Punti SOPRA (Miss)
    y_out = y_val[~punti_sotto]
    
    # B. Disegniamo i punti (Scatter plot)
    # s=1 rende i punti piccoli, alpha=0.5 li rende trasparenti per vedere la densità
    plt.scatter(x_out, y_out, color='red', s=1, alpha=0.5, label='Punti sopra')
    plt.scatter(x_in, y_in, color='green', s=1, alpha=0.5, label='Punti sotto')
    
    # C. Disegniamo la linea della funzione
    # Creiamo un array denso di x per avere una linea liscia
    x_linea = np.linspace(x_min, x_max, 1000)
    y_linea = funzione(x_linea)
    
    plt.plot(x_linea, y_linea, color='black', linewidth=2, label='Funzione f(x)')
    
    # D. Abbellimenti
    plt.legend(loc='upper right')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Integrazione Monte Carlo Hit-or-Miss')
    plt.grid(True, alpha=0.3)
    plt.show()

# funzione quad in scipy per l'integrazione 
'''from scipy.integrate import quad
# definition of a polinomial function
def polin(x): return x**2 + x + 1

area = quad (polin, 0., 4.)
print ('area = ', area[0])
print ('absolute error estimate = ', area[1])'''
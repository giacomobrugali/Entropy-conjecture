import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2


def rand_gauss_TCL (mean, sigma, N_sum = 10) :
    y = 0.
    delta = np.sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + np.random.uniform(xMin, xMax)
    return y / N_sum

def parabola (x,a,b,c) :
    return a * (x**2) + b*x + c

def parabola_vett (X,a,b,c) :
    x1 = np.asarray(X)
    return a * (x1**2) + b*x1 + c

def phi (x) :
    return 0.5 * (x**2) + 1

def phi_vett (X) :
    X1 = np.asarray (X)
    return 0.5 * (X1**2) + 1

def funzione_vett (X) :
    values = []
    sigma = []
    for x in X :
        eps = rand_gauss_TCL(0,1.5)
        ph = phi(x)
        sigma.append (abs(eps))
        values.append (ph + eps)
    return values, sigma


def psi (x,a,b,c) :
    return a * (np.e**(b*x)) + c


def model(x, a, b, c):
    # Calcoliamo l'argomento dell'esponente
    arg = b * x
    
    # 1. Protezione anti-overflow: 
    # Se arg > 700, np.exp esplode. Lo blocchiamo a 700.
    # Se arg < -700, np.exp diventa 0. Lo blocchiamo a -700.
    arg = np.clip(arg, -700, 700)
    
    # 2. Usa np.exp invece di np.e ** (è più stabile e veloce)
    return a * np.exp(arg) + c



def main () :

# 1 - generazione punti
    x_val = np.linspace (0,10,10)
    y_val, sigma  = funzione_vett(x_val)

# 2 - disegno e fit
    fig,ax = plt.subplots()
    ax.scatter(x_val, y_val,color = 'blue', marker = 'o')
    plt.show ()

    least_squares = LeastSquares(x_val, y_val, sigma, parabola)                                # Inizializzazione della funzione di costo (Q-squared)
    

    my_minuit = Minuit(least_squares, a = 0, b = 0, c = 0)                                                  # Creazione dell'oggetto Minuit con valori iniziali m=0, q=0
    
    my_minuit.migrad()                                                                           # Esegue la minimizzazione
    my_minuit.hesse()                                                                            # Calcola le incertezze

    
    print(f"Successo del fit: {my_minuit.valid}")                                                # --- Analisi del Risultato ---
    print(f"Valore del Q2 minimo: {my_minuit.fval:.3f}")
    print(f"Gradi di libertà (N-k): {my_minuit.ndof}")
    
    
    p_value = 1. - chi2.cdf(my_minuit.fval, df=my_minuit.ndof)                                   # Calcolo del p-value
    print(f"p-value associato: {p_value:.4f}")

    
    print("\nRisultati dei parametri:")                                                          # Stampa parametri e incertezze
    for par, val, err in zip(my_minuit.parameters, my_minuit.values, my_minuit.errors):
        print(f'{par} = {val:.3f} +/- {err:.3f}')

    
    print("\nMatrice di Correlazione:")                                                          # Matrice di Correlazione
    print(my_minuit.covariance.correlation())

    fig, ax = plt.subplots(figsize=(8, 6))                                                       # --- Visualizzazione ---
    ax.set_title('Fit Least-Squares di un modello parabolico')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Rappresentazione dei dati con barre d'errore
    ax.errorbar(x_val, y_val, yerr=sigma, fmt='o', label='Dati simulati', color='black')

    # Rappresentazione del modello fittato
    a_fit = my_minuit.values['a']
    b_fit = my_minuit.values['b']
    c_fit = my_minuit.values['c']
    ax.plot(x_val, parabola_vett(x_val,a_fit,b_fit,c_fit), label='Fit risultante', color='red', linewidth=2)

    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# toy exp e Q^2
    N_toy = 1000
    Q2_parabolic = []
    Q2_psi = []

    for n in range (N_toy) :
        y_val_1, sigma_1  = funzione_vett(x_val)
        least_squares = LeastSquares(x_val, y_val_1, sigma_1, parabola)                                
        my_minuit = Minuit(least_squares, a = 0, b = 0, c = 0)                                           
        my_minuit.migrad()                                                                     
        my_minuit.hesse()                                      
        q2_pa = my_minuit.fval                 
        Q2_parabolic.append (q2_pa)            

    for n1 in range (N_toy) :
        y_val_2, sigma_2  = funzione_vett(x_val)
        least_squares = LeastSquares(x_val, y_val_2, sigma_2, psi)                                
        my_minuit = Minuit(least_squares, a = 0.1, b = 1, c = 0.1)   
        my_minuit.limits["b"] = (None, 2.0)                                        
        my_minuit.migrad()                                                                     
        my_minuit.hesse()                                      
        q2_psi = my_minuit.fval                 
        Q2_psi.append (q2_psi)    

    N_punti = len(Q2_parabolic)

    n_bins = int(np.sqrt(N_punti))                                                                                              
    x_range = (np.min(Q2_parabolic), np.max(Q2_parabolic))                                          
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conteggi, bordi, patches = ax.hist(Q2_parabolic, 
                                       bins=n_bins, 
                                       range=x_range, 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.7, 
                                       
                                       label=f'Dati ($N$={N_punti})') 

    ax.set_title('Distribuzione del Q^2 da fit parabolico')
    ax.set_xlabel('x')
    ax.set_ylabel('Q^2') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)                                                        
    ax.grid(axis='x', alpha=0.3)                                                           
    plt.show() 

    N_punti_1 = len(Q2_psi)

    n_bins_1 = int(np.sqrt(N_punti))                                                                                              
    x_range_1 = (np.min(Q2_psi), np.max(Q2_psi))                                          
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conteggi, bordi, patches = ax.hist(Q2_psi, 
                                       bins=n_bins_1, 
                                       range=x_range_1, 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.7, 
                                       
                                       label=f'Dati ($N$={N_punti})') 

    ax.set_title('Distribuzione del Q^2 da fit esponenziale')
    ax.set_xlabel('x')
    ax.set_ylabel('Q^2') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)                                                        
    ax.grid(axis='x', alpha=0.3)                                                           
    plt.show() 

# 5 - soglia di rigetto
    Q0 = np.linspace (0, max( Q2_parabolic),1000)
    Q_prob_par = [] # Probabilità (frazione)
    Q_prob_psi = []

    # Trasformiamo le liste in array numpy per velocità e comodità
    arr_par = np.array(Q2_parabolic)
    arr_psi = np.array(Q2_psi)
    N_tot = len(arr_par) # Numero totale di esperimenti (es. 1000)

    # 2. Calcolo (Vettorizzato)
    for q in Q0:
        # np.sum(arr < q) conta quanti elementi soddisfano la condizione
        # Dividiamo per N_tot per ottenere la PROBABILITÀ (0-1)
        prob_par = np.sum(arr_par < q) / N_tot
        prob_psi = np.sum(arr_psi < q) / N_tot
        
        Q_prob_par.append(prob_par)
        Q_prob_psi.append(prob_psi)

    # 3. Grafico
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Q0, Q_prob_par, label='Modello Corretto (Parabola)', color='blue')
    ax.plot(Q0, Q_prob_psi, label='Modello Sbagliato (Exp)', color='red', linestyle='--')
    
    ax.set_title('Probabilità di accettazione vs Soglia di Rigetto $Q^2_0$')
    ax.set_xlabel('Soglia $Q^2_0$')
    ax.set_ylabel('Probabilità (Frazione di Toy Exp accettati)')
    ax.set_ylim(-0.05, 1.05) # Fissa l'asse Y tra 0 e 1
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
        
    



    
   


        



if __name__ == "__main__" :
    main ()
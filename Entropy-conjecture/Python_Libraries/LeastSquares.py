import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

# 1. Definizione del modello deterministico (una retta)
def func(x, m, q):
    return m * x + q

def main():
    
    m_true = 2.5                                                                                 # --- Generazione dei dati ---
    q_true = 1.2
    epsilon_sigma = 1.5                                                                          # Incertezza sui punti y
    N_punti = 15

    
    x_coord = np.linspace(0, 10, N_punti)                                                        # Generazione coordinate x e rumore gaussiano epsilon
    epsilons = np.random.normal(0, epsilon_sigma, N_punti)
    
   
    y_coord = func(x_coord, m_true, q_true) + epsilons                                           # Generazione y basata sul modello + rumore
    sigma_y = epsilon_sigma * np.ones(len(y_coord))

  
    least_squares = LeastSquares(x_coord, y_coord, sigma_y, func)                                # Inizializzazione della funzione di costo (Q-squared)
    

    my_minuit = Minuit(least_squares, m=0, q=0)                                                  # Creazione dell'oggetto Minuit con valori iniziali m=0, q=0
    
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
    ax.set_title('Fit Least-Squares di un modello lineare')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Rappresentazione dei dati con barre d'errore
    ax.errorbar(x_coord, y_coord, yerr=sigma_y, fmt='o', label='Dati simulati', color='black')

    # Rappresentazione del modello fittato
    m_fit = my_minuit.values['m']
    q_fit = my_minuit.values['q']
    ax.plot(x_coord, func(x_coord, m_fit, q_fit), label='Fit risultante', color='red', linewidth=2)

    ax.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


    # Accesso ai valori dei parametri
    m_val = my_minuit.values['m']
    q_val = my_minuit.values['q']

    # Accesso alle incertezze (errori)
    m_err = my_minuit.errors['m']
    q_err = my_minuit.errors['q']

    q2 = my_minuit.fval        # Valore del chi-quadro al minimo
    ndof = my_minuit.ndof      # Numero di gradi di libertà (N - k)


if __name__ == "__main__":
    main()
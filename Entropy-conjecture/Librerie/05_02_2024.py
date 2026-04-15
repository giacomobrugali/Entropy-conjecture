import numpy as np
import matplotlib.pyplot as plt
import myrand as mr
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
import grafici as grf
import funzioni as fun

# 1 - parabolic function

def parabola (x , a, b, c) :
    return a + (b*x) + (c * (x)**2)

def parabola_vett(X, a, b, c):
    # Trasforma X in un array numpy (se non lo è già)
    X = np.asarray(X)
    
    # Calcola la parabola su tutti gli elementi contemporaneamente
    # y = a + bx + cx^2
    return a + (b * X) + (c * X**2)

# 2 - random generator
def random_gen (a,b,c,mu,sigma,X) :
    X = np.asarray (X)
    rumore = np.array([mr.rand_gauss_TCL(mu, sigma) for _ in range(len(X))])
    return (a + (b * X) + (c * X**2)) + rumore


def random_gen_unif (a,b,c,L,X) :
    X = np.asarray (X)
    rumore = np.array([mr.rand_range (- L, L) for _ in range(len(X))])
    return (a + (b * X) + (c * X**2)) + rumore

def main () :
# parabola
    a = 3
    b = 2
    c = 1
    mu = 0
    sigma = 10
    x_val = np.linspace (0,10,10000)
    x_rand = np.linspace (0,10,10)
    y_val = parabola_vett (x_val,a,b,c)
    y_rand = random_gen (a,b,c,mu,sigma,x_rand)
    sigma_y = 10 * np.ones(len(x_rand))

    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.plot(x_val, y_val, label='parabola', color='red', linewidth=2)
    ax.errorbar(x_rand, y_rand, yerr=sigma_y, fmt='o', label='Dati simulati', color='black')
    
    plt.legend ()
    plt.show ()
# campione
    mu = 0
    sigma = 10
    x_rand = mr.generate_range (0, 10, 10)
    y_rand = random_gen (a,b,c,mu,sigma, x_rand)    
    '''print (x_rand)
    print (y_rand)'''
# fit
    least_squares = LeastSquares(x_rand, y_rand, sigma_y, parabola)
    my_minuit = Minuit(least_squares, a=0, b=0, c=0)                                                  
    my_minuit.migrad()                                                                           
    my_minuit.hesse()
    
    print(f"Successo del fit: {my_minuit.valid}")                                                
    print(f"Valore del Q2 minimo: {my_minuit.fval:.3f}")
    print(f"Gradi di libertà (N-k): {my_minuit.ndof}")
    
    
    p_value = 1. - chi2.cdf(my_minuit.fval, df=my_minuit.ndof)                                   
    print(f"p-value associato: {p_value:.4f}")

    
    print("\nRisultati dei parametri:")                                                          
    for par, val, err in zip(my_minuit.parameters, my_minuit.values, my_minuit.errors):
        print(f'{par} = {val:.3f} +/- {err:.3f}')

    
    print("\nMatrice di Correlazione:")                                                          
    print(my_minuit.covariance.correlation())

# distribuzione Q2
    N_exp = 1000
    Q2 = []

    for n in range (N_exp) :
        y_rand_1 = random_gen (a,b,c,mu,sigma,x_rand)
        least_squares = LeastSquares(x_rand, y_rand_1, sigma_y, parabola)
        my_minuit = Minuit(least_squares, a=0, b=0, c=0)                                                  
        my_minuit.migrad()       
        q2 = my_minuit.fval
        Q2.append (q2)         

    counts, edges = grf.istogramma(Q2)
    
    x_centers = (edges[:-1] + edges[1:]) / 2.
    y_values = counts
    y_errors = np.sqrt(counts)
    mask = y_values > 0                                                                   
    x_fit = x_centers[mask]                                                               
    y_fit = y_values[mask]         
    err_fit = y_errors[mask]
    least_squares = LeastSquares(x_fit, y_fit, err_fit, fun.modello_chi2)                 
    my_minuit = Minuit(least_squares, dof = 10)
    my_minuit.migrad()
    DoF = my_minuit.values['dof']
    print(DoF)

    x_q = np.linspace(1,1000,len(Q2))
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.plot(x_fit, y_fit, label='chi squared', color='blue', linewidth=2)
    plt.show ()

    N_punti = len(Q2)
    x_sovrapp = np.linspace (np.min(Q2), np.max(Q2), len(Q2))
    y_sovrapp = fun.modello_chi2_vett(x_sovrapp, DoF)
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

# toy experiments con distribuzione uniforme
    L = 10 * np.sqrt(3)
    x_rand_1 = np.linspace (0,10,10)
    y_rand_1 = random_gen_unif (a,b,c,L,x_rand_1)

    N_exp = 1000
    Q2_1 = []

    for n in range (N_exp) :
        y_rand_1 = random_gen_unif (a,b,c,L,x_rand_1)
        least_squares = LeastSquares(x_rand_1, y_rand_1, sigma_y, parabola)
        my_minuit = Minuit(least_squares, a=0, b=0, c=0)                                                  
        my_minuit.migrad()       
        q2 = my_minuit.fval
        Q2_1.append (q2)        

    N_punti_1 = len(Q2_1)
    
    # 1. Calcolo parametri
    n_bins_1 = int(np.sqrt(N_punti))                                                       # Regola della radice
    '''n_bins = op.sturges(len(dati))'''                                                 # regola di sturges
    x_range_1 = (np.min(Q2_1), np.max(Q2))
    n_bins = int(np.sqrt(N_punti))                                                       # Regola della radice
    '''n_bins = op.sturges(len(dati))'''                                                 # regola di sturges
    x_range = (np.min(Q2_1), np.max(Q2))

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 2. Costruzione istogramma
    # density=True normalizza l'area a 1 (utile per confrontare con la PDF)
    # edgecolor definisce i bordi delle colonne per una migliore leggibilità
    ax.hist(Q2, 
                                       bins=n_bins, 
                                       range=x_range, 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.4, 
                                       label=f'Dati gaussiana($N$={N_punti})',
                                       density=True) 
    ax.hist(Q2_1, 
                                       bins=n_bins_1, 
                                       range=x_range_1, 
                                       color='red', 
                                       edgecolor='black', 
                                       alpha=0.4, 
                                       label=f'Dati uniforme ($N$={N_punti})',
                                       density=True)
                                        

    # 3. Formattazione
    ax.set_title('Distribuzione dei dati generati')
    ax.set_xlabel('Valore')
    ax.set_ylabel('Densità di probabilità') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)  
    ax.grid(axis='x', alpha=0.3) 
    plt.show()

# valore di rigetto
    Q2_ordinati = np.sort(Q2)
    # Trovo l'indice che corrisponde al 90% dei dati
    # (10000 esperimenti * 0.90 = 9000)
    indice_soglia = int(0.90 * len(Q2_ordinati))

    soglia_empirica = Q2_ordinati[indice_soglia]
    print(f"Soglia calcolata dai dati: {soglia_empirica:.2f}")
    







if __name__ == "__main__" :
    main ()
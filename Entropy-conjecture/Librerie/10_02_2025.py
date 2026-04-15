import numpy as np
import matplotlib.pyplot as plt
import math
import random


def factorial(arr):
    def _calcola_singolo(n):
        if n == 0:
            return 1
        return np.prod(np.arange(1, int(n) + 1))
    vettorizzata = np.vectorize(_calcola_singolo)
    
    return vettorizzata(arr)


def pdf_poisson (x, lam) :
    x = np.asarray(x) 
    if x.ndim == 0:
        return np.exp(x * np.log(lam) - lam - math.lgamma(int(x)+1))
    else:
        return np.exp(x * np.log(lam) - lam - np.array([math.lgamma(int(i)+1) for i in x]))
    
def poisson (x,lam) :
    return ((np.e ** (-lam)) * (lam **(x))) / (factorial(x))

def poisson_vett (X,lam) :
    X1 = np.asarray (X)
    return ((np.e ** (-lam)) * (lam **(X1))) / (factorial(X1))


def inv_exp (y, lamb = 1) :
    
    return -1 * np.log (1-y) / lamb


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_exp (tau) :
    
    lamb = 1. / tau
    return inv_exp (random.random (), lamb)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_exp (tau, N, seed = 0.) :
    
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_exp (tau))
    return randlist


def pdf_exp_np(x, t0):
    x = np.array(x)
    t0 = np.array(t0)
    return (1/t0) * np.exp(-x / t0)   
       

def count (dt,rho) :
    time = 0
    count = 0
    while time <= dt :
        time += rand_exp (1 / rho)
        if time <= dt :
            count += 1
    return count

def log_likelihood_2 (theta, pdf, sample):
    """
    Calcola la log-likelihood per un campione di variabili indipendenti
    identicamente distribuite secondo la pdf con parametro theta.
    Funziona anche se sample è un array NumPy.
    """
    p = pdf(sample, theta)          # valutiamo la pdf su tutti i dati
    p = np.where(p > 0, p, 1e-300) # evita log(0) sostituendo valori <=0 con un numero piccolo
    return np.sum(np.log(p))


def GR_iter_max ( g, x_min, x_max, prec) :
   
   x_1 = 0.
   x_2 = 0.
   r = 0.618	
   L = abs(x_min - x_max)
   iter_count = 0
   while (L > prec) :
      iter_count += 1
      x_1 = x_min + (r * (x_max - x_min))
      x_2 = x_min + ((1. - r) * (x_max - x_min))
   
      if ( g(x_2) < g(x_1) ) :         
         x_min = x_2
      
      else : 
         x_max = x_1 
               
      L = abs(x_min - x_max)
   return (x_min + x_max) / 2  



def LLR(theta_values, pdf, log_likelihood_func, sample, theta_hat):
    """
    Calcola il log-likelihood ratio LLR(theta) = log(L(theta)/L_max)
    usando la differenza delle log-likelihood per stabilità numerica.
    
    Parameters:
    - theta_values: array di valori del parametro theta (per lo scan)
    - pdf: la funzione densità di probabilità
    - log_likelihood_func: funzione che calcola la log-likelihood (es. log_likelihood_2)
    - sample: dati osservati
    - theta_hat: il valore di massimo calcolato con precisione (es. via Sezione Aurea)
    
    Returns:
    - LLR_values: array dei valori del log-likelihood ratio (massimo a 0)
    """
    # 1. Calcola la log-likelihood per ogni valore di theta nel range fornito
    # Usiamo direttamente la tua log_likelihood_2 che restituisce già il logaritmo
    ll_values = np.array([log_likelihood_func(t, pdf, sample) for t in theta_values])
    
    # 2. Calcola la log-likelihood nel punto di massimo esatto (theta_hat)
    ll_max = log_likelihood_func(theta_hat, pdf, sample)
    
    # 3. Il logaritmo del rapporto è la differenza delle log-likelihood
    # Questo garantisce che il valore massimo nell'array sia esattamente 0
    return ll_values - ll_max

def intersect_LLR(
    g,              # funzione di cui trovare lo zero
    pdf,            # probability density function of the events
    sample,         # sample of the events
    xMin,           # minimo dell'intervallo          
    xMax,           # massimo dell'intervallo 
    ylevel,         # value of the horizontal intersection    
    prec = 0.0001   # precisione della funzione        
):
    """
    Funzione che calcola zeri con il metodo della bisezione
    """

    def gprime(x):
        return g(pdf, sample, x) - ylevel  # x sostituisce theta

    xAve = xMin
    while (xMax - xMin) > prec:
        xAve = 0.5 * (xMax + xMin)
        if gprime(xAve) * gprime(xMin) > 0.:
            xMin = xAve
        else:
            xMax = xAve
    return xAve

    
def main () :
# 1 - plot poisson

    x_val = np.arange(0, 25, 1)
    lambdas = np.arange(0, 11, 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.jet(np.linspace(0, 1, len(lambdas)))

    for i, lam in enumerate(lambdas):
        y_val = poisson_vett(x_val, lam)
        
        ax.plot(x_val, y_val, color=colors[i], label=f'λ = {lam}')

    ax.legend()
    ax.set_title("Distribuzione di Poisson al variare di Lambda")
    ax.grid(True, alpha=0.3)

    plt.show()

# 2 - istogramma esponenziale
    lam = 10
    tau = 1 / lam
    x_exp = generate_exp (tau, 1000)
    x_pdf = np.linspace (np.min(x_exp), np.max(x_exp),1000)
    y_pdf = pdf_exp_np(x_pdf, tau)

    N_punti = len(x_exp)
    
    n_bins = int(np.sqrt(N_punti))                                                      
    '''n_bins = op.sturges(len(dati))'''                                                
    x_range = (np.min(x_exp), np.max(x_exp))                                               
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conteggi, bordi, patches = ax.hist(x_exp, 
                                       bins=n_bins, 
                                       range=x_range, 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.7,
                                       density=True, 
                                       label=f'Dati ($N$={N_punti})') 

    ax.plot (x_pdf, y_pdf , color = 'red', lw = 1.5)

    ax.set_title('Distribuzione dei dati generati')
    ax.set_xlabel('Valore')
    ax.set_ylabel('Densità di probabilità') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)                                                           
    ax.grid(axis='x', alpha=0.3)                                                           
    plt.show()

# 3 - toy exp poisson
    ro = 2
    dt = 0.5
    lam = ro * dt
    N_toy = 2000
    N_events = []

    for n in range (N_toy) :
        c = count (dt, ro)
        N_events.append (c)
    
    x_pdf_1 = np.arange (0, np.max(N_events),1)
    y_pdf_1 = poisson_vett (x_pdf_1,lam)

    N_punti = len(N_events)
    
    min_val = np.min(N_events)
    max_val = np.max(N_events)
    bins_discreti = np.arange(min_val - 0.5, max_val + 1.5, 1)
    n_bins = int(np.sqrt(len(N_events)))                                                  
    '''n_bins = op.sturges(len(dati))'''                                                
    x_range = (np.min(N_events), np.max(N_events))                                               
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conteggi, bordi, patches = ax.hist(N_events, 
                                       bins=bins_discreti, 
                                       range=x_range, 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.7,
                                       density=True, 
                                       label=f'Dati ($N$={N_punti})') 

    ax.plot (x_pdf_1, y_pdf_1 , color = 'red', lw = 1.5)

    ax.set_title('Distribuzione dei dati generati')
    ax.set_xlabel('Valore')
    ax.set_ylabel('Densità di probabilità') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)                                                           
    ax.grid(axis='x', alpha=0.3)                                                           
    plt.show()


# 4 - log-like
    lam_val = np.linspace(0.1,3,100)
    LL_val = []
    for l in lam_val :
        log = log_likelihood_2 (l, poisson_vett, N_events)
        LL_val.append(log)

    fig, ax = plt.subplots ()

    ax.plot(lam_val, LL_val, color = 'red', lw = 2, label = 'log-likelihood')
    ax.legend ()
    plt.show ()

    l_mu = lambda l : log_likelihood_2 (l, poisson_vett, N_events)
    lam_hat = GR_iter_max ( l_mu, 0.8, 1.2, 0.0001) 

    print (lam_hat)
    
    LLR_val = LLR(lam_val, poisson_vett, log_likelihood_2, N_events, lam_hat)

    target = -0.5
    precisione = 0.0001
    lam_min = 0.5
    lam_max = 1.5

    LL_max_val = log_likelihood_2(lam_hat, poisson_vett, N_events)

    llr_lam_func = lambda pdf, sample, theta: log_likelihood_2(theta, pdf, sample) - LL_max_val
    sigma_lam_minus = intersect_LLR(llr_lam_func, poisson_vett, N_events, lam_min, lam_hat, target, precisione)
    sigma_lam_plus = intersect_LLR(llr_lam_func, poisson_vett, N_events, lam_hat, lam_max, target, precisione)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Grafico Media
    ax.plot(lam_val, LLR_val, label='Log-L-ratio', color='red')
    ax.scatter([sigma_lam_minus, sigma_lam_plus], [-0.5, -0.5], color='black', marker='o', zorder=5)
    ax.axvline(lam_hat, color='darkred', linestyle='--', label=f'Max MLE: {lam_hat:.2f}')
    ax.axhline(y=-0.5, color='gray', linestyle='--', label=r'$L_{max} - 0.5$')
    ax.set_title(r'Scansione e Fit di Lambda ($\mu$)')
    ax.set_xlim(lam_hat - 0.5, lam_hat + 0.5)
    ax.set_ylim(-1, 0.5)
    ax.legend()
    plt.show ()
  






if __name__ == "__main__" :
    main ()
           
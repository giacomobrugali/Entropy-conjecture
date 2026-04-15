import numpy as np
from math import exp, log
import Golden_ratio as Gr
import matplotlib.pyplot as plt


def exp_pdf (x, tau) :
    '''
    the exponential probability density function
    '''
    if tau == 0. : return 1.
    return exp (-1 * x / tau) / tau


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def likelihood (theta, pdf, sample) :
    '''
    the likelihood function calculated
    for a sample of independent variables idendically distributed 
    according to their pdf with parameter theta
    '''
    risultato = 1.
    for x in sample:
      risultato = risultato * pdf (x, theta)
    return risultato


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def log_likelihood (theta, pdf, sample) :
    '''
    the log-likelihood function calculated
    for a sample of independent variables idendically distributed 
    according to their pdf with parameter theta
    '''
    risultato = 0.
    for x in sample:
      if (pdf (x, theta) > 0.) : risultato = risultato + log (pdf (x, theta))    
    return risultato
    

def log_likelihood_2 (theta, pdf, sample):
    """
    Calcola la log-likelihood per un campione di variabili indipendenti
    identicamente distribuite secondo la pdf con parametro theta.
    Funziona anche se sample è un array NumPy.
    """
    p = pdf(sample, theta)          # valutiamo la pdf su tutti i dati
    p = np.where(p > 0, p, 1e-300) # evita log(0) sostituendo valori <=0 con un numero piccolo
    return np.sum(np.log(p))


def log_likelihood_2_parametrica(theta, pdf, sample, *args):
    """
    theta: parametro da stimare (es. m)
    pdf: funzione della pdf (es. pdf_gauss)
    sample: i dati
    *args: altri parametri della pdf che restano fissi (es. s)
    """
    # Ora passiamo a pdf: sample (x), theta (m) e tutti gli altri parametri (*args)
    p = pdf(sample, theta, *args) 
    
    # Protezione per log(0)
    p = np.where(p > 0, p, 1e-300) 
    
    return np.sum(np.log(p))


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


def LLR_single(theta, log_likelihood_func, sample, theta_hat):
    """
    Calcola il Log-Likelihood Ratio (LLR) traslato.
    Il valore massimo sarà sempre 0.
    """
    # Calcoliamo la log-likelihood nel punto theta
    l_theta = log_likelihood_func(theta, sample)
    
    # Calcoliamo la log-likelihood nel punto di massimo (già trovato con la sezione aurea)
    l_max = log_likelihood_func(theta_hat, sample)
    
    # Il logaritmo del rapporto è la differenza dei logaritmi
    return l_theta - l_max

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# esempio dio utilizzo della log-likelihood 2 per la stima di mu e sigma di una gaussiana

def main () :

    mu_vero = 5                                                        # definisco il vero valore di mu

    sigma_vero = 10                                                    # definisco il vero mvalore di sigma

    dati = mr.generate_N_gauss_TCL_numpy(1000, mu_vero, sigma_vero)    # genero un set di N dati distribuiti secondo la PDF da studiare con in valori veri
     
    mu_possibili = np. linspace(-5,15,10000)                           # genero un set di possibili valori di mu

    sigma_possibili = np. linspace(1,20,10000)                         # genero un set di possibili valori di sigma

    LM = []

    LS = []

    pdf = lambda x, m : PDF.pdf_gauss (x, m, 10)                       # modifico la gaussiana affinchè dipenda da un solo parametro alla volta, fissando il secondo

    pdf_1 = lambda x, s : PDF.pdf_gauss (x, 5, s)                      # cambio il parametro da fissare

    for m in mu_possibili :

        L_m = lk.log_likelihood_2 (m, pdf, dati)                       # calcolo il valore della log-like come somma di tutto il set di dati per ogni possibile valore di mu
        LM.append (L_m)
    for s in sigma_possibili :
        L_s = lk.log_likelihood_2 (s, pdf_1, dati)                     # calcolo il valore della log-like come somma di tutto il set di dati per ogni possibile valore di sigma
        LS.append (L_s)

    
    g_mu = lambda m : lk.log_likelihood_2 (m, pdf, dati)               # definisco la funzione ad una variabile da fornirme al Golden_ratio per trovare il max di mu

    g_sig = lambda s : lk.log_likelihood_2 (s, pdf_1, dati)            # definisco la funzione ad una variabile da fornirme al Golden_ratio per trovare il max di sigma

    mu_fit = Gr.GR_iter_max ( g_mu, -5, 15, 0.00001)                   # calcolo il max della log-like per mu

    sigma_fit = Gr.GR_iter_max ( g_sig, 1, 20, 0.00001)                # calcolo il max della log-like per sigma

    print (mu_fit)
    print (sigma_fit)
     
                                                                                                                 # ora devo calcolare le incertezze a dx e sx sia per mu che per sigma

    llr_mu = lk.LLR(mu_possibili, pdf, lk.log_likelihood_2, dati, mu_fit)                                        # calcolo la LLR per ogni punto : sposta il massimo della log-like in 0
    llr_sigma = lk.LLR(sigma_possibili, pdf_1, lk.log_likelihood_2, dati, sigma_fit)                             # calcolo la LLR per ogni punto : sposta il massimo della log-like in 0

    llr_mu_func = lambda p, s, x: lk.log_likelihood_2(x, p, s) - lk.log_likelihood_2(mu_fit, p, s)               # definisco la funzione LLR da passare ad INTERSECT_LLR affinchè trovi le intersez.
    llr_sig_func = lambda p, s, x: lk.log_likelihood_2(x, p, s) - lk.log_likelihood_2(sigma_fit, p, s)           # definisco la funzione LLR da passare ad INTERSECT_LLR affinchè trovi le intersez.

    target = -0.5
    precisione = 0.0001
    mu_min = -5
    mu_max = 15
    sigma_min = 1
    sigma_max = 20

    mu_minus = bs.intersect_LLR(llr_mu_func, pdf, dati, mu_min, mu_fit, target, precisione)                      # trovo l'incertezza sx di mu
    mu_plus = bs.intersect_LLR(llr_mu_func, pdf, dati, mu_fit, mu_max, target, precisione)                       # trovo l'incertezza dx di mu
    sigma_minus = bs.intersect_LLR(llr_sig_func, pdf_1, dati, sigma_min, sigma_fit, target, precisione)          # trovo l'incertezza sx di sigma
    sigma_plus = bs.intersect_LLR(llr_sig_func, pdf_1, dati, sigma_fit, sigma_max, target, precisione)           # trovo l'incertezza dx di sigma



    # --- Plotting ---
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    
    # Grafico Media
    LM = [g_mu(m) for m in mu_possibili]
    ax[0,0].plot(mu_possibili, LM, label='Log-L Media', color='red')
    ax[1,0].plot(mu_possibili, llr_mu, label='LLR Media', color='blue')
    ax[1,0].scatter([mu_minus, mu_plus], [-0.5, -0.5], color='black', marker='o', zorder=5)
    ax[0,0].axvline(mu_fit, color='darkred', linestyle='--', label=f'Max MLE: {mu_fit:.2f}')
    ax[1,0].axhline(y=-0.5, color='gray', linestyle='--', label=r'$L_{max} - 0.5$')
    ax[0,0].set_title(r'Scansione e Fit della Media ($\mu$)')
    ax[1,0].set_xlim(mu_fit - 1, mu_fit + 1)
    ax[1,0].set_ylim(-1, 1)
    ax[0,0].legend()
    ax[1,0].legend()

    # Grafico Sigma
    LS = [g_sig(s) for s in sigma_possibili]
    ax[0,1].plot(sigma_possibili, LS, label='Log-L Sigma', color='red')
    ax[1,1].plot(sigma_possibili, llr_sigma, label='LLR Sigma', color='blue')
    ax[1,1].scatter([sigma_minus, sigma_plus], [-0.5, -0.5], color='black', marker='o', zorder=5)
    ax[0,1].axvline(sigma_fit, color='darkblue', linestyle='--', label=f'Max MLE: {sigma_fit:.2f}')
    ax[1,1].axhline(y=-0.5, color='gray', linestyle='--', label=r'$L_{max} - 0.5$')
    ax[0,1].set_title(r'Scansione e Fit della Sigma ($\sigma$)')
    ax[1,1].set_xlim(sigma_fit - 1, sigma_fit + 1)
    ax[1,1].set_ylim(-0.75, 0.25)
    ax[0,1].legend()
    ax[1,1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__" :
    main ()    

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''








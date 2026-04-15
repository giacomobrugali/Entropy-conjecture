import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, chi2
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL, UnbinnedNLL, LeastSquares
from IPython.display import display
from scipy.stats import chi2

# =============================================================================
# 1. DEFINIZIONE DEI MODELLI (PDF e CDF)
# =============================================================================

# Modello per Fit Binned (richiede le Cumulative Distribution Functions - CDF)
def mod_total_binned(bin_edges, N_signal, mu, sigma, N_background, tau):
    """
    Ritorna il numero atteso di eventi in ogni bin integrando la PDF.
    """
    # N * (CDF(edge_high) - CDF(edge_low))
    sig_counts = N_signal * (norm.cdf(bin_edges, mu, sigma))
    bkg_counts = N_background * (expon.cdf(bin_edges, 0, tau))
    return sig_counts + bkg_counts

# Modello per Fit Unbinned (richiede la Probability Density Function - PDF)
def mod_signal_unbinned(x, mu, sigma):
    return norm.pdf(x, mu, sigma)

# Modello per Least Squares (approssimazione con PDF scalata)
def func_approx_ls(x, N_events, mean, sigma, bin_width):
    return N_events * norm.pdf(x, mean, sigma) * bin_width

def pdf_totale_unb(x, f_sig, mu, sigma, tau):
    """
    Modello Unbinned: deve restituire la densità di probabilità (PDF).
    f_sig è la frazione di segnale (0 <= f_sig <= 1). Nel fit unbinned standard, la funzione deve essere una vera PDF (l'integrale su tutto il range deve essere 1). Per questo usiamo una frazione fsig​ invece di due numeri indipendenti NS​ e NB​
    """
    # Calcoliamo le singole PDF
    pdf_sig = norm.pdf(x, mu, sigma)
    pdf_bkg = expon.pdf(x, 0, tau)
    
    # La PDF totale deve essere normalizzata a 1. 
    # Usiamo una frazione f_sig per combinare le due componenti.
    return f_sig * pdf_sig + (1 - f_sig) * pdf_bkg

def mod_total_polinomiale(bin_edges, N_signal, mu, sigma, c0, c1, c2):
    """
    Modello con Segnale Gaussiano + Fondo Polinomiale di 2° grado
    """
    # 1. PARTE SEGNALE (usiamo la CDF della Gaussiana)
    # N * CDF_norm
    sig_cumulative = N_signal * norm.cdf(bin_edges, mu, sigma)
    
    # 2. PARTE FONDO (Integrale del polinomio: c0 + c1*x + c2*x^2)
    # Calcoliamo l'integrale analitico valutato ai bordi dei bin
    bkg_cumulative = (c0 * bin_edges) + \
                     (0.5 * c1 * bin_edges**2) + \
                     (1/3 * c2 * bin_edges**3)
    
    # Restituiamo la somma delle componenti cumulative
    return sig_cumulative + bkg_cumulative

def func_approx_ls(x, N_events, mean, sigma, bin_width):
    """
    Modello per Least Squares: scala la PDF per l'area totale e la larghezza del bin.
    - x: centri dei bin
    - N_events: parametro di normalizzazione (area)
    - bin_width: larghezza del bin (costante se l'istogramma è uniforme)
    """
    return N_events * norm.pdf(x, mean, sigma) * bin_width


# =============================================================================
# 2. FUNZIONE PRINCIPALE DI ANALISI (ESERCIZIO REGRESSIONE)
# =============================================================================

def run_binned_analysis(data):
    """
    Esegue la procedura completa di fit su dati binnati.
    """
    # --- Preparazione dati ---
    counts, bin_edges = np.histogram(data, bins=50, range=(0, 30))
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    N_events_total = len(data)

    # --- Inizializzazione Cost Function (Extended Maximum Likelihood) ---
    cost_func = ExtendedBinnedNLL(counts, bin_edges, mod_total_binned)
    
    # --- Inizializzazione Minuit ---
    # Valori iniziali stimati dai dati
    m = Minuit(cost_func, 
               N_signal=N_events_total/2, mu=np.mean(data), sigma=np.std(data),
               N_background=N_events_total/2, tau=2.0)

    # Limiti fisici (devono essere positivi)
    m.limits['N_signal', 'N_background', 'sigma', 'tau'] = (0, None)

    # --- STEP 1: Pre-fit del Background ---
    # Mascheriamo la regione del segnale (es. tra 5 e 15)
    print("\n--- Esecuzione Pre-fit Background ---")
    cost_func.mask = (bin_centres < 5) | (bin_centres > 15)
    m.fixed["N_signal", "mu", "sigma"] = True
    m.migrad()

    # --- STEP 2: Pre-fit del Segnale ---
    print("--- Esecuzione Pre-fit Segnale ---")
    cost_func.mask = None # Rimuovi maschera
    m.fixed = False       # Sblocca tutto
    m.fixed["N_background", "tau"] = True # Blocca bkg ai valori trovati
    m.values["N_signal"] = N_events_total - m.values["N_background"]
    m.migrad()

    # --- STEP 3: Fit Finale (Tutti i parametri liberi) ---
    print("--- Esecuzione Fit Finale ---")
    m.fixed = False
    m.migrad()
    m.minos() # Calcolo incertezze asimmetriche

    # --- Risultati ---
    print(f"\nFit Valido: {m.valid}")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"{p:>12} = {v:8.3f} +/- {e:8.3f}")

    # Calcolo Qualità del Fit (Chi2)
    # Se i conteggi sono alti, il valore della cost function alla fine è ~ Chi2
    ndof = len(counts) - m.nfit
    chi2_val = m.fval
    p_value = 1 - chi2.cdf(chi2_val, ndof)
    print(f"Chi2/ndof = {chi2_val:.2f}/{ndof} (p-value: {p_value:.3f})")

    return m, bin_edges, counts


def run_unbinned_analysis(data_sample):
    """
    Esegue il fit Maximum Likelihood sui singoli eventi (senza binning).
    """
    # --- Inizializzazione Cost Function (Unbinned Negative Log-Likelihood) ---
    # Nota: passiamo l'intero array 'data_sample' direttamente
    cost_func = UnbinnedNLL(data_sample, pdf_totale_unb)
    
    # --- Inizializzazione Minuit ---
    # Stimiamo i valori iniziali (f_sig di solito parte da 0.5 o stime visive)
    m = Minuit(cost_func, 
               f_sig=0.2,            # 20% segnale, 80% fondo
               mu=np.mean(data_sample), 
               sigma=np.std(data_sample), 
               tau=2.0)

    # Limiti fisici
    m.limits['f_sig'] = (0, 1)      # La frazione deve essere tra 0 e 1
    m.limits['sigma', 'tau'] = (1e-6, None) # Devono essere positivi

    # --- Esecuzione Fit ---
    print("\n--- Esecuzione Unbinned ML Fit ---")
    m.migrad()
    m.minos()

    # --- Risultati ---
    print(f"Fit Valido: {m.valid}")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"{p:>12} = {v:8.4f} +/- {e:8.4f}")

    # --- NOTA IMPORTANTE SULLA QUALITÀ DEL FIT ---
    # Nel caso unbinned, m.fval NON è un Chi2. Non esiste un p-value diretto.
    # Per valutare la qualità, si deve proiettare il fit su un istogramma 
    # e calcolare il Chi2 "a mano" dopo.
    print("\nNota: fval unbinned non è interpretabile come Chi2.")
    
    return m


def run_least_squares_analysis(data):
    """
    Esegue il fit Least Squares sui conteggi di un istogramma.
    """
    # --- Preparazione dati ---
    counts, bin_edges = np.histogram(data, bins=30, range=(0, 20))
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Incertezza sui conteggi (assumendo statistica di Poisson: sigma = sqrt(N))
    # Nota: se counts = 0, l'errore sarebbe 0 e il fit fallirebbe (divisione per zero).
    # Spesso si impone un minimo di 1 o si usano solo bin con conteggi > 0.
    errors = np.sqrt(counts)
    errors[errors == 0] = 1 # Correzione per bin vuoti
    
    # --- Inizializzazione Cost Function (Least Squares) ---
    # Richiede: x (centri), y (conteggi), y_error (incertezze), funzione modello
    cost_func = LeastSquares(bin_centres, counts, errors, func_approx_ls)
    
    # --- Inizializzazione Minuit ---
    # Passiamo 'bin_width' come parametro FISSO (non deve essere fittato)
    m = Minuit(cost_func, 
               N_events=len(data), 
               mean=np.mean(data), 
               sigma=np.std(data),
               bin_width=bin_width)

    # Blocchiamo la larghezza del bin perché è una proprietà dell'istogramma, non del modello
    m.fixed["bin_width"] = True
    m.limits['N_events', 'sigma'] = (0, None)

    # --- Esecuzione Fit ---
    print("\n--- Esecuzione Least Squares Fit ---")
    m.migrad()
    m.minos()

    # --- Risultati ---
    print(f"Fit Valido: {m.valid}")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        if not m.fixed[p]: # Stampa solo i parametri liberi
            print(f"{p:>12} = {v:8.3f} +/- {e:8.3f}")

    # --- BONTÀ DEL FIT (Chi2) ---
    # Nei Minimi Quadrati, il valore minimo della funzione (fval) è ESATTAMENTE il Chi2.
    from scipy.stats import chi2
    chi2_val = m.fval
    ndof = len(counts) - m.nfit
    p_value = 1 - chi2.cdf(chi2_val, ndof)
    
    print(f"\nChi2: {chi2_val:.2f}")
    print(f"ndof: {ndof}")
    print(f"p-value: {p_value:.4f}")
    
    return m, bin_centres, counts, errors

# =============================================================================
# 3. VISUALIZZAZIONE
# =============================================================================


def plot_results(m, bin_edges, counts):
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.figure(figsize=(10, 6))
    
    # Plot dati
    plt.errorbar(bin_centres, counts, yerr=np.sqrt(counts), fmt='ko', label='Dati (Poisson)')
    
    # Plot fit totale
    x_plot = np.linspace(bin_edges[0], bin_edges[-1], 200)
    # Per il plot usiamo la differenza di CDF su piccoli step per simulare la linea continua
    # o più semplicemente la funzione modello valutata sui centri bin (moltiplicata per normalizzazione)
    y_fit = mod_total_binned(bin_edges, *m.values)
    # Nota: mod_total_binned restituisce array per ogni bin. 
    # Per una curva fluida usiamo la PDF pesata:
    x_fine = np.linspace(bin_edges[0], bin_edges[-1], 500)
    bin_w = bin_edges[1] - bin_edges[0]
    y_fine = (m.values['N_signal'] * norm.pdf(x_fine, m.values['mu'], m.values['sigma']) +
              m.values['N_background'] * expon.pdf(x_fine, 0, m.values['tau'])) * bin_w
    
    plt.plot(x_fine, y_fine, 'r-', lw=2, label='Fit Totale')
    plt.xlabel('x')
    plt.ylabel('Conteggi')
    plt.legend()
    plt.show()


# =============================================================================
# ESTRAZIONE DATI DA M
# =============================================================================

# Valore centrale
    mu_val = m.values["mu"]

# Errore parabolico (Hessian)
    mu_err = m.errors["mu"]

    #print(f"Risultato: {mu_val:.2f} +/- {mu_err:.2f}")

    m.minos()
    error_low = m.merrors["mu"].lower
    error_high = m.merrors["mu"].upper

    #print(f"Mu: {m.values['mu']:.2f} {error_low:+.2f} {error_high:+.2f}")

    # Matrice di covarianza completa
    print(m.covariance)

# Matrice di correlazione (più facile da leggere, valori tra -1 e 1)
    print(m.covariance.correlation())


# =============================================================================
# ESEMPIO DI UTILIZZO (Main)
# =============================================================================


if __name__ == "__main__":
    # Generazione dati sintetici per test (o caricamento da file)
    # data = np.loadtxt("data.txt") 
    np.random.seed(42)
    d_sig = np.random.normal(10, 2, 2000)
    d_bkg = np.random.exponential(5, 8000)
    sample = np.concatenate([d_sig, d_bkg])
    sample = sample[(sample > 0) & (sample < 30)] # clip range

    # Esecuzione
    minuit_obj, edges, cnts = run_binned_analysis(sample)
    
    # Visualizzazione
    plot_results(minuit_obj, edges, cnts)


# =============================================================================
# 4. TOOLS DI ANALISI STATISTICA AVANZATA
# =============================================================================

def get_goodness_of_fit(m, n_bins):
    """
    Calcola il p-value per valutare la bontà del fit (Goodness of Fit).
    Valido se i conteggi per bin sono > 5.
    """
    # Il valore minimo della cost function (fval) per ML binata 
    # si comporta come un Chi2 sotto l'ipotesi corretta.
    chi2_val = m.fval 
    ndof = n_bins - m.nfit
    
    p_value = 1 - chi2.cdf(chi2_val, ndof)
    
    print(f"\n--- Analisi Qualità del Fit ---")
    print(f"Chi2 / ndof: {chi2_val:.2f} / {ndof} = {chi2_val/ndof:.2f}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("ATTENZIONE: Il p-value è basso (< 0.05). Il modello potrebbe non descrivere bene i dati.")
    else:
        print("Il modello descrive i dati in modo statisticamente accettabile.")
        
    return p_value, chi2_val

def compare_models_lrt(m_null, m_alt):
    """
    Likelihood Ratio Test (Teorema di Wilks).
    Confronta un modello 'Nullo' (es. solo fondo) con uno 'Alternativo' (fondo + segnale).
    """
    # Delta Chi2 = -2 * ln(L_null / L_alt) = 2 * (ln(L_alt) - ln(L_null))
    # iminuit.fval restituisce già -2 * ln(L), quindi:
    delta_chi2 = m_null.fval - m_alt.fval
    
    # Gradi di libertà = differenza del numero di parametri liberi
    df = m_alt.nfit - m_null.nfit
    
    p_value = 1 - chi2.cdf(delta_chi2, df)
    
    # Calcolo dei Sigma (Significatività statistica)
    from scipy.stats import norm
    sigma = norm.ppf(1 - p_value)

    print(f"\n--- Confronto Modelli (Likelihood Ratio Test) ---")
    print(f"Delta Chi2: {delta_chi2:.3f} (df: {df})")
    print(f"P-value (H0 è corretta): {p_value:.2e}")
    print(f"Significatività: {sigma:.2f} sigma")
    
    return p_value, sigma
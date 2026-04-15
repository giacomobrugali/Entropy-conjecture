# esame 22_01_2024

import numpy as np
import matplotlib.pyplot as plt
import integrazione as it
import myrand as mr
import operazioni as op
import random
from math import floor, ceil
from iminuit import Minuit
from scipy.stats import expon, norm
from iminuit.cost import ExtendedBinnedNLL
import fit_library as FL
import funzioni as fc
from stats import stats



def integrale_HoM(funzione , X_max, X_min, Y_max, Y_min, N_eve):
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

def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)

# 1 - definizione f(X)
def funzione (A,x) :
    if x > 0 and x < (1.5 + np.pi) :
        return A * np.cos(x)**2
    else :
        return 0
    
def funzione_vett (A,X) :
    Y = []
    for x in X :
        if x > 0 and x < (1.5 + np.pi) :
            Y.append (A * np.cos(x)**2)
        else :
            Y.append (0)    
    return Y        

#2 - TaC
def funzione_TAC(N_eve):
    xMin, xMax = -1, 10
    A_norm = 0.42518633791259014
    yMin, yMax = 0, A_norm
    
    lista_x_accettate = []
    f = lambda x: funzione(A_norm, x)

    while len(lista_x_accettate) < N_eve:
        # 1. Genera un punto casuale nel rettangolo
        x_tentativo = np.random.uniform(xMin, xMax)
        y_tentativo = np.random.uniform(yMin, yMax)
        
        # 2. Verifica se il punto è sotto la curva (Acceptance)
        if y_tentativo <= f(x_tentativo):
            lista_x_accettate.append(x_tentativo)
            
    return lista_x_accettate

    
def main () :
# 1
    x_val = np.linspace (-1,10,1000)
    y_val = funzione_vett (0.42518633791259014, x_val)
    func_da_integrare = lambda x: funzione(0.42518633791259014, x)
    area, sigma_area = it.integrale_HoM(func_da_integrare, 10, -1, 3, 0, 1000000)
# 2
    x_acc = funzione_TAC (10000)
# 3
    M = ceil (max (x_acc))
    m = floor (min (x_acc))
    N_bins = op.sturges (len (x_acc))
    x_range = (m, M)
    bin_content, bin_edges = np.histogram (x_acc, bins = N_bins, range = x_range)
    N_events = sum(bin_content)
    fig, ax = plt.subplots ()
    ax.plot (x_val, y_val, color = 'black', lw = 2)
    ax.hist (x_acc, bins = bin_edges, range=(m, M), color = 'red', edgecolor = 'black', density=True)
    plt.show ()
# 4
    MEDIA = op.media (x_acc)
    DEV_STD = op.deviazione_standard(x_acc)
    SKEWNESS = op.skewness (x_acc)
    CURTOSIS =  op.kurtosis (x_acc)

    print ( 'media =', MEDIA)
    print ( 'deviazione standard  =', DEV_STD)
    print ( 'skewness  =', SKEWNESS)
    print ( 'kurtosis  =', CURTOSIS )
# 5
    lista_somme = []
    N = 1000
    for i in range (N) :
        x_acc = funzione_TAC (100)
        s = np.sum(x_acc)
        lista_somme.append (s)
    M_1 = ceil (max (lista_somme))
    m_1 = floor (min (lista_somme))
    N_bins_1 = op.sturges (len (lista_somme))
    x_range_1 = (m_1, M_1)
    bin_content_1, bin_edges_1 = np.histogram (lista_somme, bins = N_bins_1, range = x_range_1)
    N_events_1 = sum(bin_content_1)
    fig, ax = plt.subplots ()
    ax.hist (lista_somme, bins = bin_edges_1, range=(m_1, M_1), color = 'red', edgecolor = 'black', density=True)
    plt.show ()    
# verifica TCL con parametri
    Media_ls = op.media(lista_somme)
    devst_ls = op.deviazione_standard(lista_somme)
    Media_ls_teorica = 100 * MEDIA
    devst_ls_teorica = 100**0.5 * DEV_STD
    scarto_media = abs(Media_ls - Media_ls_teorica) / Media_ls_teorica
    scarto_sigma = abs(devst_ls - devst_ls_teorica) / devst_ls_teorica

    print(f"Scarto relativo Media: {scarto_media:.2%}")
    print(f"Scarto relativo Sigma: {scarto_sigma:.2%}")
# verifica TCL con fit
    from iminuit import Minuit
    from iminuit.cost import BinnedNLL
    from scipy.stats import norm
   

    my_stats_gaus = stats (lista_somme)

    # the cost function for the fit
#  my_cost_func = BinnedNLL (bin_content, bin_edges, gaus_model)
    my_cost_func = BinnedNLL (bin_content, bin_edges, fc.mod_gaus)

    my_minuit = Minuit (my_cost_func, 
                      mu = my_stats_gaus.mean (), 
                      sigma = my_stats_gaus.sigma ())

    my_minuit.migrad ()
    my_minuit.minos ()
    print (my_minuit.valid)
    from scipy.stats import chi2
    print ('associated p-value: ', 1. - chi2.cdf (my_minuit.fval, df = my_minuit.ndof))
    if 1. - chi2.cdf (my_minuit.fval, df = my_minuit.ndof) > 0.10:
        print ('the event sample is compatible with a Gaussian distribution')

    




    



if __name__ == "__main__" :
    main ()















# esame 05_02_2024
 





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

















# esame 19_02_2024





import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
import myrand as mr

z = []
Dl = []
errore = []
with open('SuperNovae.txt', 'r') as file:
        for riga in file:
        # Divide la riga in base agli spazi (o tabulazioni)
            colonne = riga.split()
        
        # Converte i valori in numeri (float) e li aggiunge alle liste
            z.append(float(colonne[0]))
            Dl.append(float(colonne[1]))
            errore.append(float(colonne[2]))
c = 3 * 10**5

def mod_lineare(x, m, q):
    return m * x + q

def mod_acc (x, H0, q) :
    return c/H0 * (x + (0.5 * (1 - q) * x**2))

def mod_lineare_vett(X, m, q):
    X1 = np.asarray (X)
    return m * X1 + q

def mod_acc_vett (X, H0, q) :
    X2 = np.asarray (X)
    return c/H0 * (X2 + (0.5 * (1 - q) * X2**2))

def toy_exp(z, Dl, errore):
    # Il ciclo while continua a girare finché non trova un fit valido
    mentre_non_valido = True
    while mentre_non_valido:
        # Estraiamo SEMPRE 30 indici
        indici = np.random.choice(len(z), size=30, replace=False)
        
        z_sub = [z[i] for i in indici]
        Dl_sub = [Dl[i] for i in indici]
        err_sub = [errore[i] for i in indici]
        
        least_squares = LeastSquares(z_sub, Dl_sub, err_sub, mod_acc)
        my_minuit = Minuit(least_squares, H0=70, q=0) 
        
        my_minuit.migrad()
        
        # Verifichiamo se il fit è riuscito su QUESTI 30 dati
        if my_minuit.fmin.is_valid:
            h0 = my_minuit.values['H0']
            q = my_minuit.values['q']
            
            # Se i valori sono fisici, interrompiamo il while e restituiamo
            if 0 < h0 < 150 and -5 < q < 5:
                return h0, q  # Qui la funzione esce e "consegna" il risultato
        
        # Se arriviamo qui, il fit era brutto: il while ricomincia da capo
        # e pescherà altri 30 indici diversi.






def main () :
    z = []
    Dl = []
    errore = []
# 1 estrazione dei dati

    with open('SuperNovae.txt', 'r') as file:
        for riga in file:
        # Divide la riga in base agli spazi (o tabulazioni)
            colonne = riga.split()
        
        # Converte i valori in numeri (float) e li aggiunge alle liste
            z.append(float(colonne[0]))
            Dl.append(float(colonne[1]))
            errore.append(float(colonne[2]))

    #print(f"Dati letti correttamente: {len(redshift)} righe.")

# 2 - grafico dei dati
    #fig, ax = plt.subplots ()
    #ax.errorbar(z, Dl, yerr=errore, fmt='o', label='Dati simulati', color='black')
    #plt.show ()

# 3 - fit lineare
    least_squares = LeastSquares(z, Dl, errore, mod_lineare)                              
    
    my_minuit = Minuit(least_squares, m=0, q=0)                                                  
    
    my_minuit.migrad()                                                                           
    my_minuit.hesse()                                                                          
    m_val = my_minuit.values['m']
    q_val = my_minuit.values['q']
    m_err = my_minuit.errors['m']
    q_err = my_minuit.errors['q']
    
    H_0 = c / m_val
    sigma_H0 = (c / (m_val**2)) * m_err

    #print ('la costante di hubble vale :', H_0, '±', sigma_H0 )

# 4 fit quadratico
    least_squares_1 = LeastSquares(z, Dl, errore, mod_acc)                              
    
    my_minuit = Minuit(least_squares_1, H0=1, q=0)                                                  
    
    my_minuit.migrad()                                                                           
    my_minuit.hesse()                                                                          

    H0_val = my_minuit.values['H0']
    q1_val = my_minuit.values['q']
    H0_err = my_minuit.errors['H0']
    q1_err = my_minuit.errors['q']
    
    #print ('la costante di hubble vale :', H0_val, '±', H0_err )

# 5 grafici sovrapposti
    x_val = np.linspace (min(z), max(z), 10000)
    y_lin = mod_lineare_vett(x_val, m_val, q_val)
    y_acc = mod_acc_vett (x_val, H0_val, q1_val)
    fig, ax = plt.subplots (figsize=(10, 8))
    ax.errorbar(z, Dl, yerr=errore, fmt='o', label='Dati simulati', color='gray')
    ax.plot (x_val, y_lin, color = 'blue', label = 'fit lineare', lw = 2, zorder=5)
    ax.plot (x_val, y_acc, color = 'red', label = 'fit parabolico', lw = 2, zorder=5)
    ax.legend ()
    plt.show ()

# 5 estrazione dei dati
    H_0_1 = H0_val
    Omega = (1 + q1_val) * (2/3)
    sig_H0 = H0_err
    sig_omega = (2/3)*q1_err

    print ('la costante di hubble vale :', H_0_1, '±', sig_H0 )
    print ('la densità dell universo vale : :', Omega, '±', sig_omega )

# 6 sub_sampling
    N_indici = 30
    indici = mr.generate_range_int (0, 411, N_indici)

    z_sub = [z[i] for i in indici]
    Dl_sub = [Dl[i] for i in indici]
    errore_sub = [errore[i] for i in indici]

    '''versione con array
# Creazione dei sotto-campioni (sub-samples)
    z_sub = z[indici]
    Dl_sub = Dl[indici]
    errore_sub = errore[indici]'''

    least_squares_2 = LeastSquares(z_sub, Dl_sub, errore_sub, mod_acc)                              
    
    my_minuit = Minuit(least_squares_2, H0=1, q=0)                                                  
    
    my_minuit.migrad()                                                                           
    my_minuit.hesse()                                                                          

    H0_val_1 = my_minuit.values['H0']
    q1_val_1 = my_minuit.values['q']
    H0_err_1 = my_minuit.errors['H0']
    q1_err_1 = my_minuit.errors['q']

    print ('la costante di hubble vale :', H0_val_1, '±', H0_err_1 )

# 7 multi sub_sampling
    N_exp = 50
    H0_list = []
    q_list = []

    for n in range(50):
        h_val, q_val = toy_exp(z, Dl, errore) 
        H0_list.append(h_val)
        q_list.append(q_val)

    n_bins_H0 = 30
    n_bins_q = 30                                                                                     
    x_range_H0 = (np.min(H0_list), np.max(H0_list))      
    x_range_q = (np.min(q_list), np.max(q_list))                                      
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2 ,figsize=(8, 5))
    ax[0].hist(H0_list, bins=n_bins_H0, range=x_range_H0, color='skyblue', edgecolor='black', alpha=0.7, label= 'Distribuzione H0 ',density = True) 
    ax[1].hist(q_list, bins=n_bins_q, range=x_range_q, color='red', edgecolor='black', alpha=0.7, label= 'Distribuzione q ', density = True) 

    # 3. Formattazione
    ax[0].set_title('Distribuzione di H 0')
    ax[1].set_title('Distribuzione diq')
    #ax.set_xlabel('Valore')
    #ax.set_ylabel('Densità di probabilità') 
    ax[0].legend()
    ax[1].legend()
    ax[0].grid(axis='y', alpha=0.3)                                                          
    ax[0].grid(axis='x', alpha=0.3)     
    ax[1].grid(axis='y', alpha=0.3)                                                          
    ax[1].grid(axis='x', alpha=0.3)                                                    
    plt.show()    

# 8 joint-pdf
    # Creazione del grafico per la distribuzione congiunta
    plt.figure(figsize=(8, 6))

# Scatter plot: ogni punto è un esperimento toy (30 supernovae)
    plt.scatter(q_list, H0_list, color='darkviolet', alpha=0.6, edgecolors='black', label='Esperimenti Toy')

# Aggiunta di etichette e stile
    plt.xlabel('Parametro di decelerazione $q$', fontsize=12)
    plt.ylabel('Costante di Hubble $H_0$ [km/s/Mpc]', fontsize=12)
    plt.title('Distribuzione Congiunta nel piano $(q, H_0)$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.show()












if __name__ == "__main__" :
    main ()






# esame 24_06_2024








import numpy as np
import matplotlib.pyplot as plt
import myrand as mr
import operazioni as op
import funzioni as f
import grafici as grf
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
import PDFs as PDF

def generatore (tau):
    mentre_non_valido = True
    while mentre_non_valido:
        x = mr.rand_exp (tau)
        if 0 <= x <= (3*tau) :
                return x  
def func (x,A,B,m,s,t) :
     return A *(PDF.pdf_gauss(x, m, s)) + B/t * (np.exp((-x)/t))

def func_vett (X,A,B,m,s,t) :
     X1 = np.asarray (X)
     return A *(PDF.pdf_gauss(X, m, s)) + B/t * (np.exp(-(X1) / t))
     

def main () :

# 1 - creazione dati 
    N_exp = 2000
    Lam = 1/200
    tau = 1/Lam
    x_e = []
    for n in range (N_exp) :
         t = generatore (tau)
         x_e.append (t)
    N_gau = 200
    mu =190
    sigma = 20
    x_g = mr.generate_N_gauss_TCL_numpy(N_gau, mu, sigma)

# 2 istogramma unione
    x_E = np.asarray (x_e)
    x_tot = np.concatenate([x_g, x_E])
    counts, edges = grf.istogramma(x_tot)

# 3 fit campione
    x_centers = (edges[:-1] + edges[1:]) / 2.                                             

    y_values = counts                                                                     
    y_errors = np.sqrt(counts)

    bin_width = edges[1] - edges[0]

    mask = y_values > 0                                                                  
    x_fit = x_centers[mask]                                                             
    y_fit = y_values[mask]                                                               
    err_fit = y_errors[mask]

    F = lambda x,A,B,m,s,t : bin_width * func(x, A, B, m, s, t)

    least_squares = LeastSquares(x_fit, y_fit, err_fit, F)                 
    my_minuit = Minuit(least_squares, A = 200, B = 2000, m = 190, s = 20, t = 200 )
    my_minuit.migrad()

    A_val = my_minuit.values['A']
    B_val = my_minuit.values['B']
    m_val = my_minuit.values['m']
    s_val = my_minuit.values['s']
    t_val = my_minuit.values['t']

    A_err = my_minuit.errors['A']
    B_err = my_minuit.errors['B']
    m_err = my_minuit.errors['m']
    s_err = my_minuit.errors['s']
    t_err = my_minuit.errors['t']

    print ('A = ', A_val)
    print ('B = ', B_val)
    print ('mu = ', m_val)
    print ('sigma = ', s_val)
    print ('tau = ', t_val)
   

    N_punti = len(x_tot) 
    x_sovrapp = np.linspace (np.min(x_tot), np.max(x_tot), len(x_tot))                             
    y_sovrapp = bin_width * func_vett (x_sovrapp,A_val,B_val,m_val,s_val,t_val)                                       
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x_tot, 
                                       bins=int(np.sqrt(N_punti)), 
                                       range=(np.min(x_tot), np.max(x_tot)), 
                                       color='skyblue', 
                                       edgecolor='black', 
                                       alpha=0.7, 
                                       label=f'Dati ($N$={N_punti})'
                                       ) 

    ax.plot (x_sovrapp, y_sovrapp, color = 'red')

    ax.set_title('Distribuzione dei dati generati')
    ax.set_xlabel('Valore')
    ax.set_ylabel('Densità di probabilità') 
    ax.legend()
    ax.grid(axis='y', alpha=0.3)  
    plt.show()

# log_like
     
    



if __name__ == "__main__" :
    main () 
 




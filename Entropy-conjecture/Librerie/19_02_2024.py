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
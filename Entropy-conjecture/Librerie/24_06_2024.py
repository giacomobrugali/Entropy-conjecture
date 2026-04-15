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
import numpy as np
from scipy.stats import norm
from scipy.stats import chi2

def gauss_binned(x, N_signal, mu, sigma):
    return N_signal * norm.cdf(x, mu, sigma)

def gauss (x, mu, sigma) :
    return norm.pdf(x, mu, sigma)
    
def phi (x, m, q) :              #funzione lineare
   
   return (x * m) + q
   
def Q_2_calc (x_i, y_i, sigma_y, m, q):          #calolo teorio del Q_2
    q2 = 0
    for x, y, s in zip(x_i, y_i, sigma_y):
        q2 += ((y - phi(x, m, q)) / s) ** 2
    return q2
      
def parabolic (x, a, b, c) :              #funzione lineare
   
   return (a*(x**2)) + (b * x) +c   

# the fitting function
def mod_gaus (bin_edges, mu, sigma):
    return norm.cdf (bin_edges, mu, sigma)

def modello_chi2(x, dof):
    return chi2.pdf(x, df=dof)

def modello_chi2_vett(X, dof):
    X1 = np.asarray (X)
    return chi2.pdf(X1, df=dof)
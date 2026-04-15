import numpy as np
import Factorial as fct

def linear (x, m, q) :              #funzione lineare
   
   return (x * m) + q
   
def Q_2_calc (x_i, y_i, sigma_y, m, q):          #calolo teorio del Q_2
    q2 = 0
    for x, y, s in zip(x_i, y_i, sigma_y):
        q2 += ((y - phi(x, m, q)) / s) ** 2
    return q2
      
def parabolic (x, a, b, c) :              
   
   return (a*(x**2)) + (b * x) +c  
   
   
def phi (x, l) :
   
   return np.exp(x * l) 
   
   
def psi (x, a, b, c) :
   
   return a * (np.exp(b*x)) + c
   
def calc_probability(Q2_list, Q2):
    """
    Calcola la probabilità cumulativa P(Q2_i <= Q2)
    a partire da una lista di valori Q2_list.

    Parametri:
    -----------
    Q2_list : lista o array di valori Q^2 simulati
    Q2      : valore di Q^2 per cui calcolare la probabilità cumulativa

    Ritorna:
    --------
    prob    : frazione di valori in Q2_list minore o uguale a Q2
    """
    count = sum(q <= Q2 for q in Q2_list)
    prob = count / len(Q2_list)
    return prob


def poisson (arr, lam) :

   poiss = []
   for i in arr :
      
      y = (((lam**i)*(np.exp(-lam)))/(fct.factorial(int(i))))
      poiss.append (y)
   
   return poiss


def rayleigh (x, N) :
   return (2 * x * np.exp((-(x**2))/N)) / N



























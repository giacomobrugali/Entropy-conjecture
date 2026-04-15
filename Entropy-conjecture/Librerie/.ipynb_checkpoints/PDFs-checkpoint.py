import numpy as np
from . import operazioni as op
#from . import Factorial as fct
import math

def pdf_exp (x, t0) :
   #calcola la pdf esp per un punto
   if (x <= 0) : return 0
   return (1/t0) * np.exp ( -(x/t0))  
   
def poisson (x, lam) :
         
   return (((lam**x)*(np.exp(-lam)))/(fct.factorial(int(x))))  
   
def pdf_gauss(x, m, s):
    return (1/(s*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-m)/s)**2)   
   

def poisson_2(x, lam):
    """
    PDF di Poisson valutata su singoli valori o array di interi.
    """
    x = np.asarray(x)  # può essere singolo numero o array

    # caso singolo numero
    if x.ndim == 0:
        return np.exp(x * np.log(lam) - lam - math.lgamma(int(x)+1))
    
    # caso array
    else:
        return np.exp(x * np.log(lam) - lam - np.array([math.lgamma(int(i)+1) for i in x]))

   
def pdf_exp_np(x, t0):
    x = np.array(x)
    t0 = np.array(t0)
    return (1/t0) * np.exp(-x / t0)   
       
   
def pdf_exp_arr (arr, t0) :
   #calcola la pdf esp per un array
   lis = []
   for i in arr : 
      if (i < 0) : lis.append(0)
      else: 
         n = (1/t0) * np.exp ( -(i/t0))
         lis.append (n)
   return lis
   
   
def pdf_fondo (x) :
   
   if x <= np.pi and x >= 0 :
      return 0.5 * np.sin(x) 
      
   else :
      return 0
      
def pdf_fondo_arr (arr) :   
   L = []
   for i in arr :
      if i <= np.pi and i >= 0 :      
         L.append (0.5 * np.sin(i))       
   return L        
   
   
def double_gauss (x, mu, sigma_sx, sigma_dx) :
   
   norm = np.sqrt(2/np.pi) / (sigma_sx + sigma_dx)  # fattore di normalizzazione
   
   if x < mu:
            return (norm * np.exp(-0.5*((x-mu)/sigma_sx)**2))
   else:
            return (norm * np.exp(-0.5*((x-mu)/sigma_dx)**2))   
   
   
   
   
   

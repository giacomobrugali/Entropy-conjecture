import numpy as np
import random
import likelihood as lh
import matplotlib.pyplot as plt

def GR_iter_min ( g, x_min, x_max, prec) :
   
   x_1 = 0.
   x_2 = 0.
   r = 0.618	
   L = abs(x_min - x_max)
   
   while (L > prec) :
      
      x_1 = x_min + (r * (x_max - x_min))
      x_2 = x_min + ((1. - r) * (x_max - x_min))
   
      if ( g(x_2) > g(x_1) ) :         
         x_min = x_2
      
      else : 
         x_max = x_1 
               
      L = abs(x_min - x_max)
      
   return (x_min + x_max) / 2      
         
   
   
'''def GR_iter_max ( g, x_min, x_max, prec) :
   
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
   return (x_min + x_max) / 2         '''
   
   
def GR_rec_min ( g, x_min, x_max, prec) :
   
   x_1 = 0.
   x_2 = 0.
   r = 0.618	
   L = abs(x_min - x_max)
   
   
      
   x_1 = x_min + (r * (x_max - x_min))
   x_2 = x_min + ((1. - r) * (x_max - x_min))
   
   if ( L < prec ) :  return  (x_min + x_max) / 2 
   elif ( g(x_2) > g(x_1) ) : return GR_rec_min ( g, x_2, x_max, prec)   
   else : return GR_rec_min ( g, x_min, x_1, prec)         


def GR_rec_max ( g, x_min, x_max, prec) :
   
   x_1 = 0.
   x_2 = 0.
   r = 0.618	
   L = abs(x_min - x_max)
   
   
      
   x_1 = x_min + (r * (x_max - x_min))
   x_2 = x_min + ((1. - r) * (x_max - x_min))
   
   if ( L < prec ) :  return  (x_min + x_max) / 2 
   elif ( g(x_2) < g(x_1) ) : return GR_rec_max ( g, x_2, x_max, prec)   
   else : return GR_rec_max ( g, x_min, x_1, prec)                  
     
def GR_rec_max_LL (pdf , sample, x_min, x_max, prec) :
   
   g = lambda theta: lh.log_likelihood(theta, pdf, sample)
   tau_hat = GR_iter_max ( g, x_min, x_max, prec)  
   return tau_hat 
   
def GR_iter_max(g, x_min, x_max, prec):
    r = 0.618
    inv_r = 1.0 - r
    
    # Inizializziamo i due punti interni
    # x1 è a destra, x2 è a sinistra
    x2 = x_min + inv_r * (x_max - x_min)
    x1 = x_min + r * (x_max - x_min)
    
    # Calcoliamo i valori iniziali (solo una volta!)
    f1 = g(x1)
    f2 = g(x2)
    
    while abs(x_max - x_min) > prec:
        if f2 < f1:
            # Il massimo è a destra, restringo da sinistra
            x_min = x2
            x2 = x1      # Riciclo il punto vecchio
            f2 = f1      # Riciclo il valore vecchio
            x1 = x_min + r * (x_max - x_min)
            f1 = g(x1)   # UNICA nuova chiamata alla funzione
        else:
            # Il massimo è a sinistra, restringo da destra
            x_max = x1
            x1 = x2      # Riciclo il punto vecchio
            f1 = f2      # Riciclo il valore vecchio
            x2 = x_min + inv_r * (x_max - x_min)
            f2 = g(x2)   # UNICA nuova chiamata alla funzione
            
    return (x_min + x_max) / 2   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   

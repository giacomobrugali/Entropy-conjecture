import numpy as np
import random as rn
import matplotlib.pyplot as plt
import operazioni as op

def linear_congruental_generator ( A, C ,M , seed) :
   L = []
   x = seed
   for i in range (M) :
      x = ((A * x) + C) % M
      L.append (x)
   return L   
   

def random_exp ( l, n) :
   
   L = []
         
   for i in range (n) :
      u = rn.random ()
      x = -(l) * np.log(1 - u)
      L.append (x)
   return L
   
   
def rand_range (x_m, x_M) :
   
   return x_m + (rn.random() * (x_M - x_m))
   
   
def pdf_gauss(x, m, s):
    return (1/(s*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-m)/s)**2)
  
  
def poisson (tau, T_MAX, N) :
   
   L = []
   
   for i in range (N) :
      t_total = 0
      count = 0
      
      while ( t_total < T_MAX) :
         
         dt = np.random.exponential (scale = tau)
         t_total += dt
         
         if t_total < T_MAX :
            count += 1
      L.append (count)  
   return np.array (L)   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   

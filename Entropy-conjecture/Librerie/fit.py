      """fit lineare"""
      
      
import numpy as np
import phi as phi
import myrand as mr
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares

def main () :

#creazione punti secondo la funzione phi + epsilon   
   x_i = np.arange (0, 10, 1)
   y_i = np.zeros (10)
   epsilon_sigma = 2
   e_i = mr.generate_TCL_ms (0., epsilon_sigma, 10)
   m = int(input('m : '))
   q = int(input('q : '))
   sigma_y = epsilon_sigma * np.ones(len(y_i))
   
   for i in x_i :      
      y_i[i] = phi.linear(i, m, q) + e_i[i]
      
#minimi quadrati operazioni sulla funzione phi      
   ls = LeastSquares (x_i, y_i, sigma_y, phi.phi)
   my_minuit = Minuit (ls, m = 0, q = 0)
   my_minuit.migrad ()
   my_minuit.hesse ()   
   vali = my_minuit.valid 
   Q_2 = my_minuit.fval
   DoF = my_minuit.ndof
   m_fit = my_minuit.values[0]
   q_fit = my_minuit.values[1]  
   m_sig = my_minuit.errors[0]
   q_sig = my_minuit.errors[1]  
   print ( 'successo : ', vali)   
   print ( 'Q_2 = ', Q_2,'  ', 'DoF = ', DoF)
   print ( 'm = ', m,'  ','sigma m =  ', m_sig)
   print ( 'q = ', q,'  ','sigma q =  ', q_sig)  
      

   
if __name__ == "__main__" :
   main ()   
   
   
   
   
   
   
   
           """fit parabolico"""
   
import numpy as np
import phi as phi
import myrand as mr
import matplotlib.pyplot as plt
import rand_gen as rg
from iminuit import Minuit
from iminuit.cost import LeastSquares


def main () :
   
   x_i = np.arange (0, 10, 1)
   y_i = np.zeros (10)
   epsilon_sigma = 11
   e_i = mr.generate_TCL_ms (0., epsilon_sigma, 10)
   a = rg.rand_range(1, 10)
   b = rg.rand_range(1, 10)
   c = rg.rand_range(1, 10)
   sigma_y = epsilon_sigma * np.ones(len(y_i))
   
   print ( 'a vero = ', a,'  ','b vero =  ', b, '  ','c vero =', c)
   
   for i in x_i :      
      y_i[i] = phi.parabolic(i, a, b, c) + e_i[i]
      
      
   ls = LeastSquares (x_i, y_i, sigma_y, phi.parabolic)
   my_minuit = Minuit (ls, a = 0, b = 0, c = 0)
   my_minuit.migrad ()
   my_minuit.hesse ()   
   vali = my_minuit.valid 
   Q_2 = my_minuit.fval
   DoF = my_minuit.ndof
   a_fit = my_minuit.values[0]
   b_fit = my_minuit.values[1]
   c_fit = my_minuit.values[2]  
   a_sig = my_minuit.errors[0]
   b_sig = my_minuit.errors[1] 
   c_sig = my_minuit.errors[2] 
   #print ( 'successo : ', vali)   
   #print ( 'Q_2 = ', Q_2,'  ', 'DoF = ', DoF)
   print ( 'a = ', a_fit,'  ','sigma a =  ', a_sig)
   print ( 'b = ', b_fit,'  ','sigma b =  ', b_sig) 
   print ( 'c = ', c_fit,'  ','sigma c =  ', c_sig)
   print(my_minuit.covariance) 
   for par, val, err in zip (my_minuit.parameters, my_minuit.values, my_minuit.errors) :
       print(f'{par} = {val:.3f} +/- {err:.3f}')                                            # formatted output
      
   fig, ax = plt.subplots (1,1)
   
   ax.errorbar (x_i, y_i,xerr = 0., yerr = sigma_y, color = 'red', marker = 'o', linestyle = 'none', label = f'a = {a}, b = {b}, c = {c}')
   plt.legend()
   plt.show ()
   
if __name__ == "__main__" :
   main ()      
   
   
   

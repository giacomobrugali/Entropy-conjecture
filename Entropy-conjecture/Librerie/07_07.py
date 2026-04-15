#Esame 07_07_2025

import numpy as np
import matplotlib.pyplot as plt
import myrand as mr
from iminuit import Minuit
from iminuit.cost import LeastSquares
import integrazione as my_int

def number_generator () :
    X_i = np.linspace(1,7,5)
    Y_i = []
    for x in X_i :
        PHI = np.exp(-0.2 * x)
        e = mr.rand_TCL_ms (0, 0.04, N_sum = 10)
        y = PHI + e
        Y_i.append (y) 
    return X_i, Y_i        

def modello_exp(x, l):
    return np.exp(l * x)

def Exp (l):
    X_e = np.linspace (0,10,1000)
    Y_e = []
    for x in X_e :
        y = np.exp(l * x)
        Y_e.append (y)
    return X_e, Y_e    

def grafico (x, y) :
    X_e, Y_e = Exp(-0.2)
    fig, ax = plt.subplots()
    ax.scatter (x, y, color = 'red', marker = 'o')
    ax.plot (X_e, Y_e, color = 'blue')
    plt.show ()

def MinimiQuad (X , Y):      
   sigma_Y = np.ones(len(X))
   ls = LeastSquares (X, Y, sigma_Y, modello_exp)
   my_minuit = Minuit (ls, l = 0)
   my_minuit.migrad ()
   my_minuit.hesse ()   
   vali = my_minuit.valid 
   Q_2 = my_minuit.fval
   DoF = my_minuit.ndof
   l_fit = my_minuit.values[0] 
   l_sig = my_minuit.errors[0]
   print ( 'successo : ', vali)   
   print ( 'Q_2 = ', Q_2,'  ', 'DoF = ', DoF)
   print ( 'l = ', l_fit,'  ','sigma l =  ', l_sig)
   return l_fit , l_sig

# stima errore aggiuntivo




def main () :
    X_i, Y_i = number_generator ()
    print (X_i)
    print (Y_i)
    grafico (X_i,Y_i)
    X_t, Y_t = Exp(-0.2)
    MinimiQuad (X_i, Y_i)
    l_fit,l_sig = MinimiQuad (X_i, Y_i)
    X_max = 6
    X_min = 1
    Y_max = 1
    Y_min = 0
    funz_da_integ = lambda x : modello_exp (x, l_fit)
    I, I_sigma= my_int.integrale_HoM(funz_da_integ, X_max, X_min, Y_max, Y_min, 100000)
    print ( 'I = ', I,'  ','sigma lI =  ', I_sigma)
    
    funz_da_integ_1 = lambda x : modello_exp (x, l_fit + l_sig)
    funz_da_integ_2 = lambda x : modello_exp (x, l_fit - l_sig)
    I_1, I_sigma_1 = my_int.integrale_HoM(funz_da_integ_1, X_max, X_min, Y_max, Y_min, 100000)
    I_2, I_sigma_2 = my_int.integrale_HoM(funz_da_integ_2, X_max, X_min, Y_max, Y_min, 100000)

    sig_agg = 0.5 * (I_1 - I_2)
    print ( 'integrale originale : ', I)
    print ( 'integrale up : ', I_1)
    print ( 'integrale down : ', I_2)
    print ( 'l incertezza aggiuntiva è : ', sig_agg)
    


if __name__ == "__main__" :
    main ()
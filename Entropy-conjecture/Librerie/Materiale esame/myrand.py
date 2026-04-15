#!/usr/bin/python

import random
from math import sqrt
import numpy as np
import PDFs as pdf

def generate_uniform (N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali distribuiti fra 0 ed 1
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (random.random ())
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_range (xMin, xMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali distribuiti fra xMin ed xMax
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    if type (N) != int : N = int (N)
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_range (xMin, xMax))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TAC (f, xMin, xMax, yMax) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo try and catch
    '''
    x = rand_range (xMin, xMax)
    y = rand_range (0, yMax)
    while (y > f (x)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_gauss_TAC (mean, sigma, xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo try and catch
    '''
    x = rand_range (xMin, xMax)
    yMax = 1/(sigma * np.sqrt(2 * np.pi))
    y = rand_range (0, yMax)
    while (y > pdf.pdf_gauss(x, mean,sigma)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

def rand_double_gauss_TAC (mu, sigma_sx, sigma_dx, xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo try and catch
    '''
    x = rand_range (xMin, xMax)
    yMax = 1/(sigma_dx * np.sqrt(2 * np.pi))
    y = rand_range (0, yMax)
    while (y > pdf.double_gauss(x, mu, sigma_sx, sigma_dx)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x
    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TAC (f, xMin, xMax, yMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo try and catch, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TAC (f, xMin, xMax, yMax))
    return randlist

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_gauss_TAC (mean, sigma, xMin, xMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali distribuiti secondo una gaussiana
    con il metodo try and catch, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_gauss_TAC (mean, sigma, xMin, xMax))
    return randlist

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

def generate_double_gauss_TAC (mu, sigma_sx, sigma_dx, xMin, xMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo try and catch, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_double_gauss_TAC (mu, sigma_sx, sigma_dx, xMin, xMax))
    return randlist

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

def rand_TCL (xMin, xMax, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    su un intervallo fissato
    '''
    y = 0.
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TCL (xMin, xMax, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, in un certo intervallo,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TCL_ms (mean, sigma, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    note media e sigma della gaussiana
    '''
    y = 0.
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;




# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_N_gauss_TCL_numpy(N_points, mean, sigma, N_sum=12):
    '''
    Generazione vettorizzata di N_points basata sul TCL.
    '''
    # Calcolo dei parametri dell'intervallo uniforme
    delta = np.sqrt(3 * N_sum) * sigma
    xMin = mean -  delta
    xMax = mean +  delta
    
    # 1. Generiamo una matrice (N_points x N_sum) di numeri uniformi
    # Ogni riga rappresenta i campioni da sommare per ottenere un singolo valore gaussiano
    matrix = np.random.uniform(xMin, xMax, (N_points, N_sum))
    
    # 2. Calcoliamo la media lungo l'asse 1 (le righe)
    # Questo produce un array di N_points elementi
    gauss_array = np.mean(matrix, axis=1)
    
    return gauss_array




# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TCL_ms_positivi (mean, sigma, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale positivo
    con il metodo del teorema centrale del limite
    note media e sigma della gaussiana
    '''
    y = 0.
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    if y > 0 : return y 
    else : return rand_TCL_ms_positivi (mean, sigma, N_sum = 10)



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_gauss_TCL (mean, sigma, N_sum = 10) :
    ''' generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    note media e sigma della gaussiana'''
    y = 0.
    delta = np.sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + np.random.uniform(xMin, xMax)
    return y / N_sum


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

def generate_TCL_ms (mean, sigma, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, note media e sigma della gaussiana,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def inv_exp (y, lamb = 1) :
    '''
    Inverse of the primitive of the exponential PDF.
    pdf(x) = lambda * exp(-lambda x) x >= 0, 0 otherwise.
    F(x) = int_{0}^{x} pdf(x)dx = 1 - exp(-lambda * x) for x >= 0, 0 otherwise.
    F^{-1}(y) = - (ln(1-y)) / lambda
    '''
    return -1 * np.log (1-y) / lamb


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_exp (tau) :
    '''
    generazione di un numero pseudo-casuale esponenziale
    con il metodo della funzione inversa
    a partire dal tau dell'esponenziale
    '''
    lamb = 1. / tau
    return inv_exp (random.random (), lamb)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_exp (tau, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali esponenziali
    con il metodo della funzione inversa, noto tau dell'esponenziale,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_exp (tau))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_poisson (mean) :
    '''
    generazione di un numero pseudo-casuale Poissoniano
    a partire da una pdf esponenziale
    '''
    total_time = rand_exp (1.)
    events = 0
    while (total_time < mean) :
        events = events + 1
        total_time = total_time + rand_exp (1.)
    return events


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_poisson (mean, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali Poissoniani
    a partire da una pdf esponenziale
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_poisson (mean))
    return randlist




import numpy as np
import matplotlib.pyplot as plt
import PDFs as PDF

def number_generator () :
    X_i = np.linspace(1,7,5)
    Y_i = []
    for x in X_i :
        PHI = np.exp(-0.2 * x)
        e = PDF.pdf_gauss(x, 0, 0.04)
        y = PHI + e
        Y_i.append (y) 
    return X_i, Y_i       
 
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

def main () :
    X_i, Y_i = number_generator ()
    print (X_i)
    print (Y_i)
    grafico (X_i,Y_i)

if __name__ == "__main__" :
    main ()
 
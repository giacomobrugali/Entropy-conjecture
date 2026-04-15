import numpy as np
import rand_gen as rg
import matplotlib.pyplot as plt
import likelihood as lh
import PDFs as pdf
import Golden_ratio as Gr
import operazioni as op
import Bisezione as bi

def main () :

   tau_vero = rg.rand_range (5, 15)
   n = 1000
   tau = np.linspace ( 1, 20, 1000)
   L = rg.random_exp(tau_vero, n)
   m = op.media(L)
   eps = 1e-300
   x_val = np.arange( len(L))
   values = []
   values1 = []
   
   for i in tau :
      loglik = lh.log_likelihood (i, pdf.pdf_exp, L) 
      values.append (loglik)
                        
   tau_hat = Gr.GR_rec_max_LL (pdf.pdf_exp , L, 5, 15, 0.0001)
   
   g = lambda x : (lh.log_likelihood (x, pdf.pdf_exp, L)) - (lh.log_likelihood (tau_hat, pdf.pdf_exp, L)) + 0.5
   
   for i in tau :
      gi = g(i)
      values1.append (gi)
            
   sigma_negativo = bi.bisezione_ric ( g, tau_hat - 2, tau_hat, 0.00001)
   sigma_positivo = bi.bisezione_ric ( g, tau_hat , tau_hat + 2, 0.00001)         
            
   print('tau_hat = ', tau_hat)
   print('sigma_negativo = ', tau_hat - sigma_negativo)
   print('sigma_positivo = ', sigma_positivo - tau_hat)
   
   fig, ax = plt.subplots(1,1)
   ax.plot (tau , values1, color = 'blue')
   plt.axhline (0, color = 'black')
   plt.axvline (tau_hat, color = 'black', linestyle = '--')
   ax.scatter ( tau_hat, g(tau_hat), color = 'red')
   ax.scatter ( sigma_negativo, 0, color = 'green')
   ax.scatter ( sigma_positivo, 0, color = 'green')
   plt.xlim(tau_hat- 1 ,tau_hat + 1)
   plt.ylim(-1 , 1)
   plt.show ()
   
   
if __name__ == "__main__" :
   main()   
   
   

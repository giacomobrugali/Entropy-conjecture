                                  """07 - 07 - 2025"""


"""Nel contensto dell’analisi dei dati il calcolo di un integrale definito di una funzione parametrica può essere
soggetto a due tipi di incertezza: una dovuta al termine algoritmico legato al calcolo dell’integrale stesso,
l’altra legata alla precisione con la quale si conosce la funzione integranda o i suoi parametri.

1. Si generino cinque punti (xi, yi) con le coordinate xi equamente distanziate fra 1 e 7, e le yi seguano
l’andamento yi = φ(xi, λ) + εi, dove la funzione φ(x, λ) è una curva esponenziale decrescente con
λ = −0.2 e i numeri εi seguono una distribuzione di densità di probabilità Gaussiana centrata in
zero con varianza 0.04.

2. Si disegnino i punti generati sovrapposti all’andamento della funzione.

3. Si esegua il fit dei punti ottenuti con la libreria iMinuit, determinando il valore ottimale di λ e la
sua incertezza, controllando che il fit abbia avuto successo.

4. Si calcoli l’integrale della curva esponenziale nell’intervallo (1, 6), utilizzando il parametro ottenuto
dal fit, utilizzando il metodo del hit-or-miss, determinando anche l’incertezza statistica dell’integrale.

5. Si determini una stima dell’incertezza aggiuntiva sulla stima dell’integrale dovuta all’errore sul co-
efficiente ˆλ."""









import numpy as np
import myrand as mr
import phi as phi
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import funzioni_fit as ff
import integrazione as integ
import operazioni as op


def main () :
   
# 1 - creazione dei punti
    l = -0.2
    mu = 0
    var = 0.04
    sigma = (var)**0.5   
    x_i = np.linspace (1,7,5)
    e_i = mr.generate_TCL_ms (mu, var, 5)
    y_i = []
   
    for x, e in zip (x_i, e_i) :
      
       y = phi.phi (x,l) + e
       y_i.append(y)

# 3 - fit con iMinuit LeastSquares
    sigma_y = np.ones(len(y_i))
    ls = LeastSquares (x_i, y_i, sigma_y, phi.phi)
    my_minuit = Minuit (ls, l = 0)
    my_minuit.migrad ()
    my_minuit.hesse ()   
    vali = my_minuit.valid 
    Q_2 = my_minuit.fval
    DoF = my_minuit.ndof
    l_fit = my_minuit.values[0]
    l_sig = my_minuit.errors[0]
    print ( 'successo : ', vali)   
    print ( 'l = ', l_fit,'  ','sigma l =  ', l_sig)
  
# 4 - integrazione tra (1 , 6) con HoM, e visualizzazione grafico
    g = lambda x : phi.phi(x, l_fit)
    I , I_sigma = integ.integral_HoM (g, 6, 1, 1, 0, 1000)
    print('I = ', I)
    print('sigma I = ', I_sigma)
   
   
    X_val = np.linspace(0,8,100)
    Y_val = phi.phi(X_val, l)
    fig, ax = plt.subplots(1,1)
    ax.scatter (x_i, y_i, color = 'red', alpha = 0.6, marker = 'o')
    ax.plot (X_val, phi.phi(X_val, l), color = 'blue', lw = 2, label = 'phi(x,l) = e^(-lx)')
    mask = (X_val >= 1) & (X_val <= 6)
    plt.fill_between(X_val[mask], Y_val[mask], 0, color='cyan', alpha=0.7)
    plt.legend ()
    plt.show ()
# 5 - incertezza aggiuntiva 
    mu = l_fit
    sigma = l_sig   
    l_val = mr.generate_TCL_ms (mu, var, 5)
    I_val = []
    I_sigma_val = []
    
    for l in l_val :
        
        g1 = lambda x : phi.phi(x, l)
        I , I_sigma = integ.integral_HoM (g1, 6, 1, 1, 0, 1000)
        I_val.append(I)
        I_sigma_val.append(I_sigma)
        
    I_5 = op.media (I_val)
    sigma_I_5 = op.deviazione_standard(I_val)
    print ('I_5 = ', I_5, '  ' , 'sigma I_5 = ', sigma_I_5)
           
if __name__ == "__main__":
   main()   
   
   
   
   
   
   
                   """24 - 02 - 2025"""   
   
   
"""Per decidere se ritenere valido un fit con il metodo dei minimi quadrati si utilizza il test del χ2, che fissa una
soglia di accettazione Q2
0 sul valore della somma degli scarti quadratici, Q2, che caratterizza il singolo fit.
Questa tecnica ha il costo di accettare soltanto una frazione dei fit effettuati, che viene definita in italiano
specificità (true positive rate in inglese).
Quando il test viene applicato al caso in cui ci sono due possibili ipotesi sotto indagine, anche la probabilità
che il Q2 di un fit effettuato con il modello sbagliato sia minore della soglia Q2
0 non è nulla ed indica il
tasso di falsi positivi (false positive rate in inglese).
La curva che mostra l’andamento della specificità rispetto al tasso di falsi positivi è chiamata Receiver
Operating Characteristic (spesso abbreviato in ROC) e rappresenta graficamente l’efficacia di un test di
ipotesi.

1. Si crei una libreria che implementi la generazione di numeri pseudo-casuali Gaussiani utilizzando
la tecnica del teorema centrale del limite e la si utilizzi per generare dieci coppie di punti (xi, yi) tali
per cui:
yi = φ(xi, θ) + εi , (1)
dove i numeri εi sono indipendenti, identicamente distribuiti secondo una distribuzione di densità
di probabilità Gaussiana con media μ = 0 e deviazione standard σ = 1.5, mentre la funzione φ(x, θ)
ha la seguente forma:
φ(x, θ) = 0.5 x2 + 1 , (2)
con le xi distribuite a distanza regolare fra 0 e 10.

2. Si faccia il disegno dei punti così generati e si esegua il fit dei punti utilizzando la più generica
parabola possibile, verificando il successo e la bontà del fit.

3. Utilizzando le tecnica dei toy experiment si costruisca la distribuzione del Q2 atteso dal fit con 1000
diversi tentativi e se ne disegni l’istogramma, scegliendone con un algoritmo opportuno gli estremi
ed il binning.

4. In modo analogo, si costruisca la distribuzione del Q2 atteso nel caso in cui la funzione di fit utilizzata
sia:
ψ(x, θ) = a ebx + c (3)
e la si disegni sovrapposta all’istogramma del punto precedente.

5. Si disegni, al variare della soglia di rigetto Q2
0, la probabilità di accettare il risultato di un fit nel caso
in cui sia fatto con il modello corretto (parabolico) rispetto alla probabilità di accettarlo nel caso del
modello sbagliato (esponenziale)."""




import numpy as np
import myrand as mr
from iminuit import Minuit
from iminuit.cost import LeastSquares 
import matplotlib.pyplot as plt
import phi as phi
from math import floor, ceil
import operazioni as op


def fi (N) :

# 1 - creazione dei punti   
   x_i = np.linspace(0,10, N)
   e_i = mr.generate_TCL_ms (0, 1.5, N)
   y_i = []
   
   for x,e in zip (x_i, e_i) :
      
      y = phi.parabolic (x, 0.5, 0, 1) + e
      y_i.append(y)
   return y_i  
   
   
def main () :
      
   x_i = np.linspace(0,10, 10)   
   y_i = fi(10)      
# 2 - creazione grafico e fit
   sigma_y = np.ones(len(y_i))
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
   print ( 'successo : ', vali)   
   print ( 'a = ', a_fit,'  ','sigma a =  ', a_sig)     
   print ( 'b = ', b_fit,'  ','sigma b =  ', b_sig)
   print ( 'c = ', c_fit,'  ','sigma c =  ', c_sig)
   print ( 'chi rid = ', Q_2 / DoF) 
   
   
   x_val = np.linspace(-1,12,100)
   y_val = phi.parabolic(x_val,0.5,0,1)
   y_val1 = phi.parabolic(x_val,a_fit,b_fit,c_fit)
   fig, ax = plt.subplots (1,1)
   ax.plot (x_val,y_val, color = 'blue', lw = 2, label = 'y = 0.5*X^2 + 1' )
   ax.plot (x_val,y_val1, color = 'green', lw = 2, label = 'phi fit' )
   ax.scatter(x_i, y_i, color = 'red', marker = 'o')
   plt.legend ()
   plt.show ()
   
# 3 - toy experiment e istogramma
   N = 1000
   CHI_2 = []
   for n in range (N) :
      
      x_i = np.linspace(0,10, 10)
      P = fi (10)
      sigma_y = np.ones(len(P))
      ls = LeastSquares (x_i, P, sigma_y, phi.parabolic)
      my_minuit = Minuit (ls, a = 0, b = 0, c = 0)
      my_minuit.migrad ()
      Q_2 = my_minuit.fval
      CHI_2.append (Q_2)
      
   #x_min = 0
   #x_max = np.ceil (max (CHI_2))
   #N_bins = op.sturges (len (CHI_2))
   N_bins = op.sturges(len(CHI_2))

   #bin_edges = np.linspace (x_min, x_max, N_bins)
   #h_para, edges_para = np.histogram (CHI_2, bins = bin_edges)

   fig, ax = plt.subplots ()
   ax.hist (CHI_2, bins = N_bins, color = 'red', edgecolor = 'black', label = 'parabolic')
   plt.legend ()
   plt.show ()
      
# 4 - istogramma modello sbagliato
   N = 1000
   chi_2 = []
   for n in range (N) :
      
      x_i = np.linspace(0,10, 10)
      P = fi (10)
      sigma_y = np.ones(len(P))
      ls = LeastSquares (x_i, P, sigma_y, phi.psi)
      my_minuit = Minuit (ls, a = 0, b = 0, c = 0)
      my_minuit.migrad ()
      Q_2 = my_minuit.fval
      chi_2.append (Q_2)
      
   #x_min = 0
   #x_max = np.ceil (max (CHI_2))
   #N_bins = op.sturges (len (CHI_2))
   N_bins1 = op.sturges(len(chi_2))

   #bin_edges = np.linspace (x_min, x_max, N_bins)
   #h_para, edges_para = np.histogram (CHI_2, bins = bin_edges)

   fig, ax = plt.subplots ()
   ax.hist (chi_2, bins = N_bins, color = 'blue', edgecolor = 'black', label = 'exponential')
   plt.legend ()
   plt.show ()     

   fig, ax = plt.subplots ()
   ax.hist (CHI_2, bins = N_bins, color = 'red', edgecolor = 'black', label = 'parabolic', alpha = 0.6)
   ax.hist (chi_2, bins = N_bins1, color = 'blue', edgecolor = 'black', label = 'exponential', alpha = 0.6)
   plt.legend ()
   plt.show ()    

# 5 - true positives vs false positives

   Q2_merger = CHI_2 + chi_2
   Q2_merger.sort ()

   prob_para = [phi.calc_probability (CHI_2, Q2) for Q2 in Q2_merger]
   prob_expo = [phi.calc_probability (chi_2, Q2) for Q2 in Q2_merger]

   fig, ax = plt.subplots ()
   ax.plot (prob_expo, prob_para,
             color = 'red',
             label = 'ROC curve',
            )
   ax.set_xlabel ('false positives')
   ax.set_ylabel ('true positives')
   plt.show ()











   
   
if __name__ == "__main__":
   main()      
   
   
   
   
   
   
   
   
   
      
   
   
   
   
                   """10 - 02 - 2025"""  
   
   
   
"""La distribuzione di densità di probabilità (pdf) esponenziale è profondamente legata a quella Poissoniana
e viene utilizzata, ad esempio, per descrivere gli inter-tempi di decadimento di processi con un tasso di
decadimento costante (spesso detto rate ρ, che è il numero di eventi medio atteso nell’unità di tempo)
osservati in una data finestra di tempo ∆t.

1. Si definisca in una libreria la funzione densità di probabilità di Poisson con parametro λ e se ne
tracci il grafico per valori interi del parametro da 0 a 10.

2. Si scriva, in una libreria, una funzione per generare 1000 eventi distribuiti esponenzialmente data
una costante esponenziale fissata λ = 10 e si disegni l’istogramma dei valori ottenuti scegliendo
gli estremi ed il numero di bin dell’istogramma con un algoritmo adeguato; si sovrapponga la cor-
rispondente forma funzionale della pdf all’istogramma.

3. Sapendo che λ = ρ × ∆t, si fissi ρ = 2 Hz e si effettuino 2000 toy experiment che simulino il
numero di eventi osservati in un intervallo di tempo ∆t = 0.5 s. Si disegni il risultato ottenuto in un
istogramma e lo si sovrapponga alla distribuzione di Poisson corrispondente, scritta analiticamente
(implementandola nella libreria sviluppata al punto precedente).

4. Si scriva, sempre nella libreria sviluppata, la funzione che implementa il logaritmo della verosimi-
glianza (chiamata log_likelighood) associata alla distribuzione degli eventi, utilizzando la pdf di
Poisson come modello atteso e se ne faccia il disegno al variare del parametro λ della Poissoniana.

5. Si determini il valore di λ per cui log_likelighood è massima utilizzando il metodo della sezione
aurea e si determini l’incertezza associata a questo stimatore con il metodo grafico, stampando i
risultati a schermo"""




import myrand as mr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import phi as phi
import operazioni as op
import PDFs as pdf
import likelihood as L
import Golden_ratio as Gr
import Bisezione as bi

def main () :
   
#1 - grafico poisson 
   N = np.arange( 0,11,1)
   x_val = np.arange (0,50)
   fig, ax = plt.subplots (1,1)
   colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


  
   for l, c in zip(N, colors):
      y = phi.poisson(x_val, l)
      ax.plot(x_val, y, color=c, label=f"λ={l}")

   plt.legend ()
   plt.show ()
   
#2 - istogramma esponenziale
   y_val_exp = mr.generate_exp (10, 1000)  
   
   fig, ax = plt.subplots (1,1)
   N_bins = op.sturges(len(y_val_exp))
   ax.hist ( y_val_exp, bins = N_bins, color = 'red', edgecolor = 'black', density = True)
   ax.plot (x_val, pdf.pdf_exp_arr (x_val, 10), color = 'blue', lw = 2)
   
   plt.show ()
   
#3 - toy experiments   
   ρ = 2 #Hz 
   t = 0.5 #s
   λ = ρ * t
   N = 2000
   boh = mr.generate_poisson (λ, 2000, seed = 0.)
   x_val1 = np.arange(0, np.max(boh)+1)
            
   fig, ax = plt.subplots (1,1)
   bins = np.arange(0, max(boh)+2) - 0.5
   #N_bins = op.sturges(len(boh))
   ax.hist ( boh, bins = bins, color = 'red', edgecolor = 'black', density = True)
   ax.plot (x_val1, phi.poisson (x_val1, λ), color = 'blue', lw = 2)
   plt.show()

#4 - log likelihood
   l = np.linspace( 0.1, 5, 100)
   values = []
   for i in l :
                 
      loglik = L.log_likelihood (i, pdf.poisson, boh)      
      values.append(loglik)
      
   fig, ax = plt.subplots (1,1)
   
   ax.plot(l, values, color ='red', lw = 2, label = 'log-likelihood')
   
   ax.legend ()
   
   plt.tight_layout()
   plt.show ()

#5 - massimo e incertezza
   λ = 1
   boh = mr.generate_poisson (λ, 2000, seed = 0.)
   l = np.linspace( 0.1, 5, 600)
   lamda_hat = Gr.GR_rec_max_LL (pdf.poisson_2 , boh, 0, 2, 0.00001)   
   values1 = []
      
   g = lambda x : ((L.log_likelihood (x, pdf.poisson, boh)) - (L.log_likelihood (lamda_hat, pdf.poisson, boh))) + 0.5
 
   
   for i in l :
      gi = g(i)
      values1.append (gi)
   
   sigma_negativo = bi.bisezione_iter ( g, lamda_hat - 0.1, lamda_hat, 0.001)
   sigma_positivo = bi.bisezione_iter ( g, lamda_hat , lamda_hat + 2, 0.001)         
            
            
   print('lamda_hat = ', lamda_hat)
   print('sigma_negativo = ', lamda_hat - sigma_negativo)
   print('sigma_positivo = ', sigma_positivo - lamda_hat)
   
   fig, ax = plt.subplots (1,1)
   
   ax.plot (l, values1, color = 'black', lw = 2)
   ax.scatter (lamda_hat, g(lamda_hat), color = 'red', marker = 'o')
   plt.axhline (0, color = 'black')
   ax.scatter (sigma_negativo, 0, color = 'green', marker = 'o')
   ax.scatter (sigma_positivo, 0, color = 'green', marker = 'o')
   plt.xlim(lamda_hat- 0.05 ,lamda_hat + 0.05)
   plt.ylim(-0.5 , 0.5)
   plt.show ()
   
   

      
   
if __name__ == "__main__" :
   main()      
   
       
   
   
   
   
   
   
                    """27 - 01 - 2025"""    
   
   

"""Molto spesso ci si trova nella situazione di dover determinare la quantità di segnale presente in uno spettro
misurato, sapendo di aver osservato un sistema fisico che produce eventi di interesse (segnale) contaminati
da impulsi spuri (il rumore di fondo).

1. Si scriva una libreria di Python che contenga l’implementazione di una distribuzione di densità
di probabilità, chiamata pdf_fondo, definita su un intervallo [0, π]. La funzione avrà la forma di
A sin(x), dove il parametro A è determinato dalle richiesta che sull’intervallo di definizione la fun-
zione sia normalizzata. Si scriva la funzione in modo che, oltre alla variabile x, prenda in ingresso
tutti i parametri necessari a definirla.

2. Si faccia il disegno della funzione sull’intervallo di definizione e si controlli che la funzione è nor-
malizzata, utilizzando il metodo hit-or-miss di integrazione.

3. Si generi un campione di 2000 punti pseudo-casuali distribuiti secondo pdf_fondo utilizzando il
metodo della funzione inversa e se ne disegni l’istogramma, scegliendone con un algoritmo oppor-
tuno il binnaggio (non si dimentichi che la funzione cumulativa è una particolare primitiva della pdf
corrispondente, perché deve avere limite uguale a zero a meno infinito ed uguale ad 1 a più infinito).

4. Si generi, utilizzando il metodo try-and-catch, un secondo campione di 1000 eventi secondo una
distribuzione Gaussiana con media π/4 e larghezza π/20, lo si unisca in un unico container a quello
generato precedentemente e se ne disegni l’istogramma.

5. A partire dal risultato del punto precedente, utilizzando la tecnica dei toy experiment, si determini
quale sia il bias medio dello stimatore ˆμ che determina la posizione del massimo del segnale con un
fit Gaussiano fra 0 e π/2."""




import PDFs as pdf
import matplotlib.pyplot as plt
import integrazione as integ
import numpy as np
import operazioni as op
import random as rn
import myrand as mr
import funzioni_fit as ff


def random_fondo (n) :
   
   L = []
         
   for i in range (n) :
      u = rn.random ()
      x = np.arccos(1 - 2*u)
      L.append (x)
   return L


def main () :

# 2 - grafico e integrale HoM 
   x_val = np.linspace (0, np.pi, 300)
   y_val = pdf.pdf_fondo_arr (x_val)
   I, I_sigma = integ.integral_HoM(pdf.pdf_fondo, np.pi, 0, 1, 0, 1000)
   print ( 'I = ', I, '  ','sigma I = ', I_sigma)
   
   fig,ax = plt.subplots (1,1)
   
   ax.plot (x_val, y_val, color = 'blue', lw = 2)
   
   plt.show ()
   
# 3 - generazione random con l'inversa    
   L = random_fondo (2000)
   fig,ax = plt.subplots (1,1)
   N_bins = op.sturges(len(L))
   
   ax.hist (L, bins = N_bins, color = 'red', edgecolor = 'black')
   plt.show ()
   
# 4 - try-and-catch gaussiana    
   mean = np.pi / 4
   sigma = np.pi / 20
   xMin = -2
   xMax = 5
   N = 1000

   G = mr.generate_gauss_TAC (mean, sigma, xMin, xMax, N, seed = 0.) 
   #lstG = G.tolist()                         trasforma array in liste (in questo caso sono già liste)
   #lstL = L.tolist()

   M = G + L
   fig,ax = plt.subplots (1,1)
   ax.hist (M, bins = N_bins, color = 'red', edgecolor = 'black')
   plt.show ()
   
# 5 - stima bias
   N_toys = 1000
   M = []
      
   for n in range (N_toys) :
      
      g = mr.generate_gauss_TAC (mean, sigma, 0, np.pi/2, 500, seed = 0.)
      mu_experiment = ff.UNBINNED_gauss(g)
      M.append (mu_experiment[1])
   
   bias = op.media(M) - mean
   print ('bias = ',bias)







   
   
   
if __name__ == "__main__" :
   main ()   
   
   
   
   
   
   
   
   
   
   
                      """13 - 01 - 2025"""   
   
   
   
"""Per poter riprodurre dati sperimentali con asimmetria non nulla, si può utilizzare un modello che genera-
lizza la distribuzione Gaussiana, con le code a sinistra e destra del massimo caratterizzate da parametri σ
diversi.

1. Si scriva una libreria di Python che contenga l’implementazione di una distribuzione di densità di
probabilità, chiamata double_Gaus, definita sull’asse reale. La funzione dovrà possedere un massimo
e due code Gaussiane, a destra e sinistra del massimo rispettivamente, con sigma differenti (σsx e
σdx), ricordando che la funzione deve essere continua su tutto l’asse reale.

2. Si scriva un programma in Python che ne faccia un disegno per controllare il risultato.

3. Si controlli che la funzione è normalizzata, utilizzando il metodo di integrazione hit-or-miss

4. Si generi un campione di 1000 punti pseudo-casuali distribuiti secondo la pdf double_Gaus utiliz-
zando il metodo try-and-catch, se ne disegni l’istogramma scegliendone con un algoritmo appropriato
minimo, massimo e numero di bin e si stampino a schermo la media e la mediana del campione ot-
tenuto.

5. Si assuma che σsx < σdx e si trovi una formula che ricavi il rapporto σdx/σsx a partire dalla differenza
fra media e mediana del campione, utilizzando il metodo dei toy experiment per determinarla em-
piricamente. Si provi ad utilizzare un fit per ottenere il risultato: assumendo che l’incertezza sulla
mediana sia uguale a quella sulla media, quale variabile viene utilizzata come indipendente? Come
si può quantificare l’affidabilità della formula?"""





import numpy as np
import PDFs as pdf
import matplotlib.pyplot as plt
import integrazione as integ
import myrand as mr
import operazioni as op
from iminuit import Minuit
from iminuit.cost import LeastSquares
import phi as phi

def double_gauss (x, mu, sigma_sx, sigma_dx) :
   
   norm = np.sqrt(2/np.pi) / (sigma_sx + sigma_dx)  # fattore di normalizzazione
   
   if x < mu:
            return (norm * np.exp(-0.5*((x-mu)/sigma_sx)**2))
   else:
            return (norm * np.exp(-0.5*((x-mu)/sigma_dx)**2))
      
# definizione double_gauss     
def double_gauss_arr(arr, mu, sigma_sx, sigma_dx):
    L = []
    norm = np.sqrt(2/np.pi) / (sigma_sx + sigma_dx)  # fattore di normalizzazione
    for i in arr:
        if i < mu:
            L.append(norm * np.exp(-0.5*((i-mu)/sigma_sx)**2))
        else:
            L.append(norm * np.exp(-0.5*((i-mu)/sigma_dx)**2))
    return L

      

def main () :

# 2 - grafico 
   x_val = np.linspace (20, 190,500)
   y_val = double_gauss_arr (x_val, 50, 2, 30)
   
   fig, ax = plt.subplots (1,1)
   ax.plot (x_val, y_val, color = 'red', lw = 2)
   plt.show ()
   
   
# 3 - integrazione HoM
   g = lambda x : double_gauss (x,0,1,8)
   I, I_sigma = integ.integral_HoM(g, 35, -5, 1, 0, 10000)
   print ('I = ', I)
   
# 4 - toy experiments e istogramma
   N_toys = 1000
   mu = 50
   sigma_sx = 2
   sigma_dx = 30
   xMin = 20
   xMax = 190
   DG = mr.generate_double_gauss_TAC (mu, sigma_sx, sigma_dx, xMin, xMax, N_toys, seed = 0.)
   m = mu - 4*sigma_sx
   M = mu + 4*sigma_dx
   h_min = np.floor (min (DG))
   h_max = np.ceil (max (DG))
   n_bins = op.sturges (len (DG))

   bin_content, bin_edges = np.histogram (DG, bins = n_bins, range = (h_min, h_max))

   fig, ax = plt.subplots (nrows = 1, ncols = 1)
   ax.hist (DG,
         bins = bin_edges,
         color = 'deepskyblue',
        )
   
   ax.axvline (x=op.mediana (DG), color='red', linestyle='--', label='median')
   ax.axvline (x=op.media (DG), color='blue', linestyle='--', label='mean')
   ax.legend ()
   
   plt.show ()

# 5 - fit formula
   sigma_sx = 4
   mu = 5
   sigma_dx = np.linspace(5,40,200)
   xMin = -30
   xMax = 200
   N_eventi = 100
   N_toys = 200
   x_delta = []
   y_quoz = []
   for sd in sigma_dx :
   
      delta = []
      
      for n in range (N_toys) :
      
         dg = mr.generate_double_gauss_TAC (mu, sigma_sx, sd, xMin, xMax, N_eventi)
         mean = op.media (dg)
         med = op.mediana (dg)
         delt = abs(mean - med)         
         delta.append (delt)
        
      
      dlt = op.media(delta)
      quoz = sd / sigma_sx 
      y_quoz.append (quoz) 
      x_delta.append (dlt)      
      
   fig, ax = plt.subplots (1,1)
   ax.plot (x_delta, y_quoz, color ='black', lw = 2)
   #plt.show ()   
     
#5 - minimi quadrati lineare     
   sigma_y =  np.ones(len(y_quoz))
   ls = LeastSquares (x_delta, y_quoz, sigma_y, phi.linear)
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
   """print ( 'Q_2 = ', Q_2,'  ', 'DoF = ', DoF)"""
   print ( 'm = ', m_fit,'  ','sigma m =  ', m_sig)
   print ( 'q = ', q_fit,'  ','sigma q =  ', q_sig)   



   
if __name__ == "__main__" :   
   
   
   
 
 
 
 
 
 
                        """10 - 10 - 2025"""
 
 


"""Secondo l’algoritmo di Box-Müller, dati due numeri pseudo-casuali x1 ed x2 generati uniformemente
nell’intervallo (0, 1), si dimostra che i due numeri g1 e g2 calcolati con le equazioni seguenti:
g1 = √−2 log(x1) cos (2πx2) (1)
g2 = √−2 log(x1) sin (2πx2) (2)
possano essere considerati due numeri pseudo-casuali distribuiti secondo una distribuzione di densità di
probabilità normale.

1. Si scriva una funzione chiamata generate_gaus_bm che generi coppie di numeri pseudo-casuali
distribuiti secondo una densità di probabilità Gaussiana utilizzando l’algoritmo di Box-Müller, im-
plementata in una libreria dedicata.

2. Si generino N = 1000 numeri pseudo-casuali utilizzando la funzione appena sviluppata e li si disegni
in un istogramma, scegliendone con un algoritmo opportuno gli estremi ed il binnaggio.

3. Si determinino media e varianza della distribuzione ottenuta e relativi errori.

4. Si mostri graficamente che, al variare del numero N di eventi generati, la sigma della distribuzione
non cambia, mentre l’errore sulla media si riduce.

5. Si trasformi l’algoritmo in modo che generi numeri pseudo-casuali con densità di probabilità Gaus-
siana con media μ = 5 e varianza σ2 = 4. Si generi un nuovo campione di N = 1000 eventi con il
nuovo algoritmo e se ne disegni la distribuzione, sempre scegliendo in modo opportuno gli estremi
ed il binnaggio dell’istogramma corrispondente"""





import numpy as np
import myrand as mr
import random as rn
import operazioni as op
import matplotlib.pyplot as plt
import funzioni_fit as ff

def rand_gauss_mb () :
   
   x1 = rn.random ()
   x2 = rn.random ()
   if x1 != x2 :
      g1 = np.sqrt((-2)*(np.log(x1))) * np.cos(2*np.pi*x2)
      g2 = np.sqrt((-2)*(np.log(x1))) * np.sin(2*np.pi*x2)
      
      return g1, g2
      
      
def generate_gauss_mb (N) :

   randlist = []
   for i in range (N//2):
       x1 = rn.random ()
       x2 = rn.random ()
       g1 = np.sqrt((-2)*(np.log(x1))) * np.cos(2*np.pi*x2)
       g2 = np.sqrt((-2)*(np.log(x1))) * np.sin(2*np.pi*x2)
       randlist.append(g1)
       randlist.append(g2)
        
   return randlist [:N]    #nel caso di disparità tronco la lista a N
   
def generate_gauss_mb_2 (N, mu, sigma) :

   randlist = []
   for i in range (N//2):
       x1 = rn.random ()
       x2 = rn.random ()
       g1 = np.sqrt((-2)*(np.log(x1))) * np.cos(2*np.pi*x2)
       g2 = np.sqrt((-2)*(np.log(x1))) * np.sin(2*np.pi*x2)
       randlist.append(mu + (sigma*g1))
       randlist.append(mu + (sigma*g2))
        
   return randlist [:N]    #nel caso di disparità tronco la lista a N   
   

def main () :
   
#2 - istogramma
   N = 1000
   G = generate_gauss_mb (N) 
   
   h_min = np.floor (min (G))
   h_max = np.ceil (max (G))
   n_bins = op.sturges (len (G))

   bin_content, bin_edges = np.histogram (G, bins = n_bins, range = (h_min, h_max))

   fig, ax = plt.subplots (nrows = 1, ncols = 1)
   ax.hist (G,
         bins = bin_edges,
         color = 'deepskyblue', density = True
        )
   
   ax.axvline (x=op.mediana (G), color='red', linestyle='--', label='sigma')
   ax.axvline (x=op.deviazione_standard (G), color='blue', linestyle='--', label='mean')
   ax.legend ()
   
   plt.show ()
   
#3 - fit con gaussiana
   BG = ff.UNBINNED_gauss(G)   
   print ('mu = ', BG[1],'  ','sigma mu = ', BG[3])
   print ('sigma = ', BG[2],'  ','sigma sigma = ', BG[4])
   
#4 - grafici sigma e sigma_mu
   N = np.linspace (50, 1000, 300)
   err_mu = []
   sigma = []
   for n in N :
       g = generate_gauss_mb (int(n)) 
       BG = ff.UNBINNED_gauss(g)
       err_mu.append(BG[3])
       sigma.append(BG[2])
   
   fig, ax = plt.subplots (1,1)
   ax.plot(N, sigma, color = 'red', label = 'sigma', lw = 2)
   ax.plot(N, err_mu, color = 'blue', label = 'errore media', lw = 2)
   plt.legend()
   plt.show () 

#5 - grafico mb_2
   N = 1000
   mu = 5
   var = 4
   sigma = np.sqrt(var)
   G1 = generate_gauss_mb_2 (N, mu, sigma) 
   
   h_min = np.floor (min (G1))
   h_max = np.ceil (max (G1))
   n_bins = op.sturges (len (G1))

   bin_content, bin_edges = np.histogram (G1, bins = n_bins, range = (h_min, h_max))

   fig, ax = plt.subplots (nrows = 1, ncols = 1)
   ax.hist (G1,
         bins = bin_edges,
         color = 'deepskyblue', edgecolor = 'black', density = True)
   
   ax.axvline (x=op.media (G1), color='red', linestyle='--', label=f"media = {op.media(G1):.3f}")
   ax.axvline (x=op.media(G1) - op.deviazione_standard (G1), color='blue', linestyle='--', label=f"deviazione standard = {op.deviazione_standard(G1):.3f}")
   ax.axvline (x=op.media(G1) + op.deviazione_standard (G1), color='blue', linestyle='--')
   ax.legend ()
   
   plt.show ()

       
         
   
if __name__ == "__main__" :
   main ()   
   
    
 
 
 
 
 
 
 
 
 
 
 
                        """16 - 09 - 2025"""
 
 

"""L’ottimizzazione dell’integrazione numerica con il metodo Monte Carlo si ottiene, fra le altre cose, con una
scelta oculata delle coordinate x dei punti generati casualmente. Infatti, più essi ricoprono in maniera
ottimale l’insieme di definizione della funzione da integrare, migliore è la precisione ottenuta nella sua
stima, a parità di punti generati. La sequenza sn generata secondo il seguente algoritmo:
sn+1 = (sn + α) mod 1 (1)
produce un insieme di punti, distribuiti fra 0 ed 1, che hanno la proprietà di ben riempire questo insieme
di definizione, in particolare se α = (√5 − 1)/2.

1. Si scriva una libreria che contenga una classe di python, chiamata additive_recurrence, che generi
la sequenza di numeri sn dell’equazione (1), che abbia come variabili membro il parametro α, il
numero di partenza della sequenza e l’ultimo numero generato, che assegni un valore ad α durante
l’inizializzazione della classe ed implementi i metodi seguenti:
• get_number per ottenere un numero della sequenza
• set_seed per inizializzare la sequenza

2. Si faccia un test del funzionamento della classe generando una sequenza di 1000 numeri e scriven-
done i primi 10 a schermo.

3. Si aggiunga alla libreria una funzione chiamata MC_mod che calcoli l’integrale definito di f (x) = 2x2
nell’intervallo (0, 1), utilizzando il metodo crude Montecarlo dove la generazione dei punti lungo
l’asse x non sia fatta in modo pseudo-casuale, ma utilizzando la classe additive_recurrence.

4. Utilizzando il metodo dei toy experiment, si determini l’incertezza del calcolo dell’integrale in fun-
zione del numero totale N_points di punti generati per la stima di un singolo integrale, disegnan-
done l’andamento dell’errore in funzione di N_points al variare fra 10 e 25000.

5. Si rifaccia il medesimo test con l’algoritmo crude Montecarlo studiato a lezione e si confrontino i due
risultati: quale è più efficiente?"""




import class_exam as ce
import numpy as np
import matplotlib.pyplot as plt
import time as time
import integrazione as integ

def main () :

# 2 - test sequenza   
   N = 1000
   alpha = (np.sqrt(5) -1) / 2
   first = 0
   last = 1
   l = ce.additive_recurrence( alpha, first, last=None)
   L = [] 
   for n in range (N) :
 
      L.append (l.get_number())
   print (len(L))
   print (L[:10])
   
# 3 - MC_mod
   g = lambda x : 2 * (x**2)
   I, I_sigma = ce.MC_mod (g, 3, 0, 1000)
   print ('I = ', I)
   print ('I sigma = ', I_sigma)

# 4 - toy experiments
   start = time.time ()
   rang = np.linspace (10,25000,5000)
   g = lambda x : 2 * (x**2)
   sigma = []
   for r in rang :
      I, I_sigma = ce.MC_mod (g, 3, 0, r)
      sigma.append(I_sigma)
   end = time.time ()   
   fig, ax = plt.subplots (1,1)
   ax.plot (rang,sigma, color = 'deepskyblue')
   print('tempo = ',end - start, 'secondi')
   plt.show  ()
      
#5 - crude montecarlo
   start1 = time.time ()
   rang1 = np.linspace (10,25000,5000)
   g = lambda x : 2 * (x**2)
   sigma1 = []
   for r in rang :
      I, I_sigma = integ.integral_crude_MC(g, 1, 0, r)
      sigma1.append(I_sigma)
   end1 = time.time ()   
   fig, ax = plt.subplots (1,1)
   ax.plot (rang1,sigma1, color = 'deepskyblue')
   print('tempo 1 = ',end1 - start1, 'secondi')
   plt.show  ()









   
if __name__ == "__main__" :
   main ()       
 
 
 
 
 
 
  
  
  
                    """08 - 07 - 2025"""
  
  
  


"""In una torrida giornata di luglio, in un villaggio sperduto in Armorica, il druido Panoramix sbagliò ricetta
ed invece della solita pozione magica produsse, nel suo calderone, la grappa più alcolica mai distillata in
Gallia.

1. Si scriva una funzione che simuli il cammino degli abitanti del villaggio dopo aver bevuto la grappa,
assumendo che si spostino in piano, che ogni passo abbia direzione casuale uniforme angolarmente
ed una lunghezza distribuita secondo una distribuzione Gaussiana con media 1 e larghezza 0.2,
troncata a valori positivi.

2. Immaginando che il calderone si trovi alle coordinate (0, 0) sul piano, si scriva una funzione che
calcoli la posizione (x, y) raggiunta da Asterix dopo N = 10 passi e si disegni il suo percorso.

3. Si consideri ora l’intera popolazione: si determini la posizione (x, y) di ogni abitante dopo N =
10 passi a partire dal calderone e si disegni le distribuzione della distanza raggiunta dal punto di
partenza, assumendo la popolazione totale composta da 10000 persone.

4. Si determinino media, varianza, asimmetria e curtosi della distribuzione ottenuta.

5. Se la lunghezza dei passi è costante uguale ad 1, la distribuzione delle distanze r dopo N passi segue
una distribuzione di Rayleigh:
f (r) = 2r
N e−r2/N . (1)
Si utilizzi un fit per determinare, a partire dalla distribuzione di distanze costruita in queste ipotesi,
il numero di passi effettuati, sapendo che la distribuzione di Rayleigh è presente in scipy come
scipy.stats.rayleigh e che per ottenere la forma funzionale di interesse per il problema questa
distribuzione ha come parametri loc = 0 e scale = √N/2 (dove N è il numero di passi)."""




import numpy as np
import matplotlib.pyplot as plt
import myrand as mr
import operazioni as op
from iminuit import Minuit
from iminuit.cost import LeastSquares
import phi as phi
from scipy.stats import rayleigh
from iminuit.cost import ExtendedBinnedNLL
from IPython.display import display



# 1 - random walk
def random_walk (N) :
   
   x_val = [0.0]
   y_val = [0.0]
   x_succ = 0
   y_succ = 0
   
   for n in range (N) :
       ang = mr.rand_range (0, 2*np.pi)
       l = mr.rand_TCL_ms_positivi (1, 0.2)
       x_succ = x_succ + np.cos(ang)*l
       y_succ = y_succ + np.sin(ang)*l
       x_val.append(x_succ)
       y_val.append(y_succ)
   
   return x_val, y_val
   
   
def random_walk_1 (N) :
   
   x_val = [0.0]
   y_val = [0.0]
   x_succ = 0
   y_succ = 0
   
   for n in range (N) :
       ang = mr.rand_range (0, 2*np.pi)
       l = mr.rand_TCL_ms_positivi (1, 0.2)
       x_succ = x_succ + np.cos(ang)*1
       y_succ = y_succ + np.sin(ang)*1
       x_val.append(x_succ)
       y_val.append(y_succ)
   
   return x_val, y_val   
   

def posizione (arr1, arr2,N) :
   
   
   x_finale = arr1[N]
   y_finale = arr2[N]
   
   return x_finale, y_finale
   
def distanza (x, y) :
   
   l = np.sqrt((x**2)+(y**2))
   return l

def modello (bin_edges, N) :
   return rayleigh.cdf (bin_edges, 0, np.sqrt (N/2))



def main () :
   
# 2 - random walk  
   """N = 10
   x_val, y_val = random_walk (N)
   x_f , y_f = posizione (x_val, y_val, N)
   print ('x finale = ', x_f)
   print ('y finale = ', y_f)
   
   fig, ax = plt.subplots (1,1)
   ax.plot (x_val, y_val, color = 'green',linestyle = '--', label=f"passi random walk = {N}" )
   ax.scatter (0, 0, color = 'red', marker = 'o' )
   ax.scatter (x_f, y_f, color = 'blue', marker = 'o' )
   ax.axvline (x=0, color='black', linestyle='-')
   ax.axhline (y=0, color='black', linestyle='-')
   plt.legend ()
   plt.show ()
   
# 3/4 - distribuzione popolazione e istogramma
   N_pop = 10000  
   N_passi = 10
   L = []
   for n in range (N_pop) :
      x_val, y_val = random_walk (N_passi)
      x_f , y_f = posizione (x_val, y_val, N_passi)
      l = distanza (x_f, y_f)
      L.append (l)
   m = op.media (L)
   var = op.varianza (L)
   skw = op.skewness (L)
   kurt = op.kurtosis (L)   
   h_min = np.floor (min (L))
   h_max = np.ceil (max (L))
   n_bins = op.sturges (len (L))

   bin_content, bin_edges = np.histogram (L, bins = n_bins, range = (h_min, h_max))

   fig, ax = plt.subplots (nrows = 1, ncols = 1)
   ax.hist (L,
         bins = bin_edges,
         color = 'deepskyblue', edgecolor = 'black', density = True,
         label=f"media = {m:.3f}, varianza = {var:.3f}, skewness = {skw:.3f}, kurtosis = {kurt:.3f}" )  
   plt.legend ()
   plt.show ()"""
   
# 3 - caratterizzazione della distribuzione popolazione 
   N_pop = 30000  
   N_passi = 10
   L = []
   for n in range (N_pop) :
      x_val, y_val = random_walk_1 (N_passi)
      x_f , y_f = posizione (x_val, y_val, N_passi)
      l = distanza (x_f, y_f)
      L.append (l)

   h_min = np.floor (min (L))
   h_max = np.ceil (max (L))
   n_bins = op.sturges (len (L))

   bin_content, bin_edges = np.histogram (L, bins = n_bins, range = (h_min, h_max))

   fig, ax = plt.subplots (nrows = 1, ncols = 1)
   ax.hist (L,
         bins = bin_edges,
         color = 'deepskyblue', edgecolor = 'black', density = True,
          )  
   
   plt.show ()


# 5 - fit
   x_i = np.linspace (min(L), max(L), len(L))
   y_i = L
   h_max = max(L)
   sigma_y = np.ones(len(y_i))
   N_events = sum(L)
   cost_func = ExtendedBinnedNLL (bin_content, bin_edges, modello)
   my_minuit = Minuit (cost_func, 
                    N = h_max)
   my_minuit.migrad ()
   my_minuit.minos ()

   display (my_minuit)
  





   
   
if __name__ == "__main__" :
   main ()   
   

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
 
 
 
 
 
   
   
   
   
   

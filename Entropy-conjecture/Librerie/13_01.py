# esame 13_01_2025
import numpy as np
import operazioni as op
import matplotlib.pyplot as plt
from math import floor, ceil
from iminuit import Minuit
from scipy.stats import expon, norm
from iminuit.cost import ExtendedBinnedNLL
from iminuit.cost import UnbinnedNLL
from IPython.display import display
import funzioni as func
import myrand as mr
import phi as phi
from iminuit import Minuit
from iminuit.cost import LeastSquares


# 1-doppia gaussiana
def double_gauss (x, m, s1, s2):

    if x < m :
        return (2/((s1+s2)*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-m)/s1)**2)   
    if x > m :
        return (2/((s1 + s2)*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-m)/s2)**2)
    if x == m :
        return (2/((s1 + s2)*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-m)/s1)**2)

def double_gauss_fast(x, m, s1, s2):
    # 1. Creiamo un array di zeri della stessa lunghezza di x per contenere i risultati
    y = np.zeros_like(x)
    
    # 2. Calcoliamo la costante di normalizzazione (corretta col fattore 2)
    norm = 2 / ((s1 + s2) * np.sqrt(2 * np.pi))
    
    # 3. Creiamo le "maschere": ci dicono DOVE applicare le formule
    sinistra = x < m    # Questo è un array di True/False
    destra = x >= m     # Anche questo
    
    # 4. Applichiamo la formula per s1 SOLO dove 'sinistra' è True
    # Nota: usiamo x[sinistra] per prendere solo le x che ci interessano
    y[sinistra] = norm * np.exp(-0.5 * ((x[sinistra] - m) / s1)**2)
    
    # 5. Applichiamo la formula per s2 SOLO dove 'destra' è True
    y[destra] = norm * np.exp(-0.5 * ((x[destra] - m) / s2)**2)
    
    return y

#2-grafico della doppia gaussiana
def graf_double_gauss (m, s1, s2):
    X_val = np.linspace (m - (5*s1), m + (5*s2), 100000)
    Y_val = double_gauss_fast(X_val, m, s1, s2)
    fig, ax = plt.subplots()
    ax.plot(X_val, Y_val, color = 'red')
    plt.show ()

#3-integrazione grafico
 
def integral_HoM( m, s1, s2, N_eve):
    x_val = np.random.uniform((m-7*s1), (m+7*s2), N_eve)
    y_val = np.random.uniform(0, 0.4, N_eve)

    y_teorici = double_gauss_fast (x_val, m, s1, s2)

    punti_sotto = y_val < y_teorici 
    
    count = np.sum(punti_sotto)
    
    A = ((m+7*s2) - (m-7*s1)) * (0.4)
    n = count / N_eve
    I = A * n
    I_sigma = A * np.sqrt(n * (1 - n) / N_eve)
    
    return I, I_sigma, x_val, y_val, punti_sotto        

def grafico_integrazione(x_val, y_val, punti_sotto, funzione, x_min, x_max):
    plt.figure(figsize=(10, 6))
    
    # A. Separiamo i punti usando la maschera booleana
    # punti_sotto è un array di True/False. 
    # Usando ~punti_sotto (la tilde inverte) otteniamo i punti sopra.
    
    x_in = x_val[punti_sotto]      # Punti SOTTO (Hit)
    y_in = y_val[punti_sotto]
    
    x_out = x_val[~punti_sotto]    # Punti SOPRA (Miss)
    y_out = y_val[~punti_sotto]
    
    # B. Disegniamo i punti (Scatter plot)
    # s=1 rende i punti piccoli, alpha=0.5 li rende trasparenti per vedere la densità
    plt.scatter(x_out, y_out, color='red', s=1, alpha=0.5, label='Punti sopra')
    plt.scatter(x_in, y_in, color='green', s=1, alpha=0.5, label='Punti sotto')
    
    # C. Disegniamo la linea della funzione
    # Creiamo un array denso di x per avere una linea liscia
    x_linea = np.linspace(x_min, x_max, 1000)
    y_linea = double_gauss_fast (x_linea, 0, 0.5, 2)
    
    plt.plot(x_linea, y_linea, color='black', linewidth=2, label='Funzione f(x)')
    
    # D. Abbellimenti
    plt.legend(loc='upper right')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Integrazione Monte Carlo Hit-or-Miss')
    plt.grid(True, alpha=0.3)
    plt.show()

def rand_double_gauss_TAC(m, s1, s2, N_eve):
    X_list = []
    
    # Calcoliamo il massimo della funzione per il box
    # Nota: Assicurati di usare la formula corretta col *2 al numeratore
    yMax = 2 / ((s1 + s2) * np.sqrt(2 * np.pi))
    
    # Definiamo il range sulle x (5 sigma per sicurezza)
    x_min = m - 5 * s1
    x_max = m + 5 * s2

    # Continuiamo finché non abbiamo riempito la lista con N_eve punti
    while len(X_list) < N_eve:
        
        # 1. Genera un candidato casuale
        x = mr.rand_range(x_min, x_max)
        y = mr.rand_range(0, yMax)
        
        # 2. Valuta la PDF nel punto x
        # Nota: qui passo x come numero singolo, double_gauss deve gestirlo
        pdf_val = double_gauss_fast(np.array([x]), m, s1, s2)[0] 
        
        # 3. Condizione di accettazione (Hit)
        if y < pdf_val:
            X_list.append(x)
            
    return X_list

def main () :
        '''s1 = 2
        s2 = 5
        m = 0
        graf_double_gauss (m, s1, s2)
        I, I_sigma, x_val, y_val, punti_sotto = integral_HoM( 0, 0.5, 2, 1000000)
        print ('I = ', I)
        grafico_integrazione(x_val, y_val, punti_sotto, lambda x : double_gauss (x,m, s1, s2), (0-7*s1), (0+7*s2))
        X_list =  rand_double_gauss_TAC(m, s1, s2, 1000)
        fig, ax = plt.subplots ()
        N_bins = op.sturges (len (X_list))
        x_min = min (X_list)
        x_max = max (X_list)
        media = op.media (X_list)
        mediana = op.mediana (X_list)
        plt.hist(X_list, bins=N_bins, range=(x_min, x_max), 
         color='skyblue', edgecolor='black', alpha=0.7, density=True, label='Campione MC')

        # Aggiungiamo le linee verticali per Media e Mediana
        plt.axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media ({media:.2f})')
        plt.axvline(mediana, color='green', linestyle='-.', linewidth=2, label=f'Mediana ({mediana:.2f})')

        # Abbellimenti
        plt.title(f'Istogramma Campione (N={1000}) con binning ottimizzato')
        plt.xlabel('Valore x')
        plt.ylabel('Densità di Probabilità')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show ()'''

        # 5-toy experiments 
        
        # --- 1. SETUP ---





''' fit lineare di un set di dati, con errore sulle y, successivo alla generazione di N toy experiment secondo la distribuzione Double_gauss'''

s1 = 1
s2 = np.linspace(2, 5, 30)
mu = 0
N_exp = 50
points_per_exp = 100 

D = []
D_dev = []

# --- 2. TOY EXPERIMENTS ---
print("Simulazione in corso...")
for i in s2:
    D_middle = []
    for n in range(N_exp):
        
        X_list = rand_double_gauss_TAC(mu, s1, i, points_per_exp)
        
        
        media = op.media(X_list)
        Mediana = op.mediana(X_list)
        s_ds = media - Mediana
        D_middle.append(s_ds)
    
    # Calcolo media e deviazione standard dei 50 toy
    S_middle = op.media(D_middle)
    S_dev = op.deviazione_standard(D_middle)
    
    D.append(S_middle)
    D_dev.append(S_dev)

# Trasformiamo in numpy array per il fit (Minuit lavora meglio così)
x_val = np.array(s2)
y_val = np.array(D)
y_err = np.array(D_dev)

# --- 3. FIT ---
ls = LeastSquares(x_val, y_val, y_err, phi.linear)
my_minuit = Minuit(ls, m=0, q=0)
my_minuit.migrad()
my_minuit.hesse()

# Estrazione Risultati
m_fit = my_minuit.values[0]
q_fit = my_minuit.values[1]
m_sig = my_minuit.errors[0]
q_sig = my_minuit.errors[1]

# Stampe
print(f'Successo: {my_minuit.valid}')
print(f'Q^2 = {my_minuit.fval:.2f}  DoF = {my_minuit.ndof}')
print(f'm = {m_fit:.4f} +/- {m_sig:.4f}')
print(f'q = {q_fit:.4f} +/- {q_sig:.4f}')

# --- 4. GRAFICO COMPLETO (Punti + Fit) ---
fig, ax = plt.subplots(figsize=(8,6))

# A. Disegniamo i dati con le barre d'errore (NON usare plot, usa errorbar)
ax.errorbar(x_val, y_val, yerr=y_err, fmt='o', color='red', label='Simulazione MC')

# B. Disegniamo la linea del fit
# Creiamo tanti punti x per fare una linea liscia
x_plot = np.linspace(min(x_val), max(x_val), 100)
# Calcoliamo la y usando i parametri trovati dal fit
y_plot = phi.linear(x_plot, m_fit, q_fit) 

ax.plot(x_plot, y_plot, color='blue', linestyle='--', label=f'Fit: y = {m_fit:.2f}x + {q_fit:.2f}')

ax.set_xlabel('Sigma 2 (Asimmetria)')
ax.set_ylabel('Differenza (Media - Mediana)')
ax.set_title('Calibrazione Asimmetria')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# --- 5. RISPOSTA ALL'ESERCIZIO (FORMULA INVERSA) ---
print("\n--- FORMULA FINALE ---")
print("Per trovare sigma2 (o il rapporto) data una differenza misurata 'diff':")
print(f"sigma2 = (diff - ({q_fit:.4f})) / {m_fit:.4f}")




if __name__ == "__main__" :
     main ()
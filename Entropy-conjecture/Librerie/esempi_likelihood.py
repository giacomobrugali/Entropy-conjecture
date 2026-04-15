#esempi likelihood
import numpy as np
import likelihood as lk
import myrand as mr
import PDFs as PDF
import matplotlib.pyplot as plt
import Golden_ratio as Gr
import Bisezione as bs


def main () :

    mu_vero = 5
    sigma_vero = 10
    dati = mr.generate_N_gauss_TCL_numpy(1000, mu_vero, sigma_vero)
    mu_possibili = np. linspace(-5,15,10000)
    sigma_possibili = np. linspace(1,20,10000)
    LM = []
    LS = []
    pdf = lambda x, m : PDF.pdf_gauss (x, m, 10)
    pdf_1 = lambda x, s : PDF.pdf_gauss (x, 5, s)
    for m in mu_possibili :
        L_m = lk.log_likelihood_2 (m, pdf, dati)
        LM.append (L_m)
    for s in sigma_possibili :
        L_s = lk.log_likelihood_2 (s, pdf_1, dati)
        LS.append (L_s)
    
    g_mu = lambda m : lk.log_likelihood_2 (m, pdf, dati)
    g_sig = lambda s : lk.log_likelihood_2 (s, pdf_1, dati)

    mu_fit = Gr.GR_iter_max ( g_mu, -5, 15, 0.00001)
    sigma_fit = Gr.GR_iter_max ( g_sig, 1, 20, 0.00001)

    print (mu_fit)
    print (sigma_fit)


    
    llr_mu = lk.LLR(mu_possibili, pdf, lk.log_likelihood_2, dati, mu_fit)
    llr_sigma = lk.LLR(sigma_possibili, pdf_1, lk.log_likelihood_2, dati, sigma_fit)

    llr_mu_func = lambda p, s, x: lk.log_likelihood_2(x, p, s) - lk.log_likelihood_2(mu_fit, p, s)
    llr_sig_func = lambda p, s, x: lk.log_likelihood_2(x, p, s) - lk.log_likelihood_2(sigma_fit, p, s)

    target = -0.5
    precisione = 0.0001
    mu_min = -5
    mu_max = 15
    sigma_min = 1
    sigma_max = 20

    mu_minus = bs.intersect_LLR(llr_mu_func, pdf, dati, mu_min, mu_fit, target, precisione)
    mu_plus = bs.intersect_LLR(llr_mu_func, pdf, dati, mu_fit, mu_max, target, precisione)
    sigma_minus = bs.intersect_LLR(llr_sig_func, pdf_1, dati, sigma_min, sigma_fit, target, precisione)
    sigma_plus = bs.intersect_LLR(llr_sig_func, pdf_1, dati, sigma_fit, sigma_max, target, precisione)



    # --- Plotting ---
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    
    # Grafico Media
    LM = [g_mu(m) for m in mu_possibili]
    ax[0,0].plot(mu_possibili, LM, label='Log-L Media', color='red')
    ax[1,0].plot(mu_possibili, llr_mu, label='LLR Media', color='blue')
    ax[1,0].scatter([mu_minus, mu_plus], [-0.5, -0.5], color='black', marker='o', zorder=5)
    ax[0,0].axvline(mu_fit, color='darkred', linestyle='--', label=f'Max MLE: {mu_fit:.2f}')
    ax[1,0].axhline(y=-0.5, color='gray', linestyle='--', label=r'$L_{max} - 0.5$')
    ax[0,0].set_title(r'Scansione e Fit della Media ($\mu$)')
    ax[1,0].set_xlim(mu_fit - 1, mu_fit + 1)
    ax[1,0].set_ylim(-1, 1)
    ax[0,0].legend()
    ax[1,0].legend()

    # Grafico Sigma
    LS = [g_sig(s) for s in sigma_possibili]
    ax[0,1].plot(sigma_possibili, LS, label='Log-L Sigma', color='red')
    ax[1,1].plot(sigma_possibili, llr_sigma, label='LLR Sigma', color='blue')
    ax[1,1].scatter([sigma_minus, sigma_plus], [-0.5, -0.5], color='black', marker='o', zorder=5)
    ax[0,1].axvline(sigma_fit, color='darkblue', linestyle='--', label=f'Max MLE: {sigma_fit:.2f}')
    ax[1,1].axhline(y=-0.5, color='gray', linestyle='--', label=r'$L_{max} - 0.5$')
    ax[0,1].set_title(r'Scansione e Fit della Sigma ($\sigma$)')
    ax[1,1].set_xlim(sigma_fit - 1, sigma_fit + 1)
    ax[1,1].set_ylim(-0.75, 0.25)
    ax[0,1].legend()
    ax[1,1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__" :
    main ()    
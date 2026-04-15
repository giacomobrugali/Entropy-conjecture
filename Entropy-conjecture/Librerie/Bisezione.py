
def bisezione_iter(g, x_min, x_max, prec):
    while (x_max - x_min) > prec:
        aver = 0.5 * (x_max + x_min)
        if g(aver) * g(x_min) > 0:
            x_min = aver
        else:
            x_max = aver
    return 0.5 * (x_max + x_min)


def bisezione_ric(g, x_min, x_max, prec):
    aver = 0.5 * (x_min + x_max)
    
    if abs(x_max - x_min) < prec:
        return aver
    
    if g(aver) * g(x_min) > 0:
        return bisezione_ric(g, aver, x_max, prec)
    else:
        return bisezione_ric(g, x_min, aver, prec)

         
def intersect_LLR(
    g,              # funzione di cui trovare lo zero
    pdf,            # probability density function of the events
    sample,         # sample of the events
    xMin,           # minimo dell'intervallo          
    xMax,           # massimo dell'intervallo 
    ylevel,         # value of the horizontal intersection    
    prec = 0.0001   # precisione della funzione        
):
    """
    Funzione che calcola zeri con il metodo della bisezione
    """

    def gprime(x):
        return g(pdf, sample, x) - ylevel  # x sostituisce theta

    xAve = xMin
    while (xMax - xMin) > prec:
        xAve = 0.5 * (xMax + xMin)
        if gprime(xAve) * gprime(xMin) > 0.:
            xMin = xAve
        else:
            xMax = xAve
    return xAve

    
    
    
    
    
    
    
    
    
             

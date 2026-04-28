

def factorial(n):
    if n == 0:
        return 1
    # Crea un array da 1 a n e ne calcola il prodotto
    return np.prod(np.arange(1, int(n) + 1))

def factorial_1(arr):
    # 1. Definiamo la logica per il singolo numero DENTRO la funzione
    # Questa "sotto-funzione" è invisibile all'esterno
    def _calcola_singolo(n):
        if n == 0:
            return 1
        return np.prod(np.arange(1, int(n) + 1))
    
    # 2. Vettorizziamo la sotto-funzione al volo
    vettorizzata = np.vectorize(_calcola_singolo)
    
    # 3. Restituiamo il risultato applicato all'input
    return vettorizzata(arr)
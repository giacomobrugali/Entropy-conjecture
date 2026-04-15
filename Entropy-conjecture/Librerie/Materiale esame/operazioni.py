import numpy as np

def media (arr) : 
   
   n = len(arr)
   somma = np.sum(arr)
   
   return somma / n
   
def mediana (arr) : 
   
   arr = np.sort(arr)
   
   n = len(arr)
   
   if n % 2 == 1 :
      return arr [n//2]
      
   else :
      return ( arr [n//2 - 1] + arr [n//2]) / 2
   
   
   
def varianza (arr) : 

   n = len(arr)
   somma = np.sum(arr)
   media = somma / n
   scarti_quadratici = 0
   
   for elemento in arr :
      
      scarti_quadratici += ( elemento - media )**2
      
   return ((1/(n-1)) * scarti_quadratici)
   
def deviazione_standard(arr):
    n = len(arr)
    if n <= 1:
        return 0.0
    scarti_quadratici = sum((x - sum(arr)/n)**2 for x in arr)
    return (scarti_quadratici / (n-1))**0.5
   
def skewness (arr) :  
   
   dv = deviazione_standard (arr)
   m = media (arr)
   l = len (arr)
   scarti_cubici = 0
   for elemento in arr : 
      
      scarti_cubici += (( elemento - m) / dv )**3
   return (1/l) * scarti_cubici   
   
def kurtosis (arr) :
   dv = deviazione_standard (arr)
   m = media (arr)
   l = len (arr)
   scarti_4 = 0
   for elemento in arr : 
      
      scarti_4 += (( elemento - m) / dv )**4
   return (1/l) * scarti_4      
   
def dev_stndrd_mean (arr) : 

   n = len(arr)
   somma = np.sum(arr)
   media = somma / n
   scarti_quadratici = 0
   
   for elemento in arr :
      
      scarti_quadratici += ( elemento - media )**2
      
   return (((1/(n-1)) * scarti_quadratici)**0.5)/(n**0.5)

def sturges (N_events) :
   return int(np.ceil ( 1 + np.log2(N_events)))
   
def factorial (x) :
   
   if x == 0 :
      raise ValueError
   
   elif x == 0 or x == 1 :
      return 1
      
   else :
      return x * factorial (x - 1)
      
      
def bisezione_iterativa(g, x_min, x_max, prec):
    while (x_max - x_min) > prec:
        aver = 0.5 * (x_max + x_min)
        if g(aver) * g(x_min) > 0:
            x_min = aver
        else:
            x_max = aver
    return 0.5 * (x_max + x_min)


def bisezione_ricorsiva(g, x_min, x_max, prec):
    aver = 0.5 * (x_min + x_max)
    
    if abs(x_max - x_min) < prec:
        return aver
    
    if g(aver) * g(x_min) > 0:
        return bisezione_ricorsiva(g, aver, x_max, prec)
    else:
        return bisezione_ricorsiva(g, x_min, aver, prec)


def GR_iter_min ( g, x_min, x_max, prec) :
   
   x_1 = 0.
   x_2 = 0.
   r = 0.618	
   L = abs(x_min - x_max)
   
   while (L > prec) :
      
      x_1 = x_min + (r * (x_max - x_min))
      x_2 = x_min + ((1. - r) * (x_max - x_min))
   
      if ( g(x_2) > g(x_1) ) :         
         x_min = x_2
      
      else : 
         x_max = x_1 
               
      L = abs(x_min - x_max)
      
   return (x_min + x_max) / 2      
         
   
   
def GR_iter_max ( g, x_min, x_max, prec) :
   
   x_1 = 0.
   x_2 = 0.
   r = 0.618	
   L = abs(x_min - x_max)
   
   while (L > prec) :
      
      x_1 = x_min + (r * (x_max - x_min))
      x_2 = x_min + ((1. - r) * (x_max - x_min))
   
      if ( g(x_2) < g(x_1) ) :         
         x_min = x_2
      
      else : 
         x_max = x_1 
               
      L = abs(x_min - x_max)
      
   return (x_min + x_max) / 2         
   
   
def GR_rec_min ( g, x_min, x_max, prec) :
   
   x_1 = 0.
   x_2 = 0.
   r = 0.618	
   L = abs(x_min - x_max)
   
   
      
   x_1 = x_min + (r * (x_max - x_min))
   x_2 = x_min + ((1. - r) * (x_max - x_min))
   
   if ( L < prec ) :  return  (x_min + x_max) / 2 
   elif ( g(x_2) > g(x_1) ) : return GR_rec_min ( g, x_2, x_max, prec)   
   else : return GR_rec_min ( g, x_min, x_1, prec)         


def GR_rec_max ( g, x_min, x_max, prec) :
   
   x_1 = 0.
   x_2 = 0.
   r = 0.618	
   L = abs(x_min - x_max)
   
   
      
   x_1 = x_min + (r * (x_max - x_min))
   x_2 = x_min + ((1. - r) * (x_max - x_min))
   
   if ( L < prec ) :  return  (x_min + x_max) / 2 
   elif ( g(x_2) < g(x_1) ) : return GR_rec_max ( g, x_2, x_max, prec)   
   else : return GR_rec_max ( g, x_min, x_1, prec)                  
     
     
def associa(arr1, arr2):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    def f(k):
        index = np.where(arr1 == k)[0]
        if len(index) == 0:
            raise ValueError(f"{k} non trovato in arr1")
        return arr2[index[0]]
    return f
    
def freedman_diaconis_bins(data):
    """Return number of bins according to Freedman–Diaconis rule."""
    """bisogna usare degli array, se si lavora con delle liste vanno prima trasformate in array"""
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    n = len(data)
    if iqr == 0:
        # fallback to sqrt rule if IQR=0
        return int(np.sqrt(n))
    bin_width = 2 * iqr / (n ** (1/3))
    bins = int((data.max() - data.min()) / bin_width)
    return max(1, bins)    
                
   
def factorial (x) :
   
   if x < 0 :
      raise ValueError
   
   elif x == 0 or x == 1 :
      return 1
      
   else :
      return x * factorial (x - 1)



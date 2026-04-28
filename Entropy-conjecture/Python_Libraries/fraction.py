from math import gcd
class Fraction :
   
   def __init__ (self, numeratore, denominatore) :
      
      self.n = numeratore
      self.d = denominatore
            
      if denominatore == 0 :
         raise ValueError (' denominatore non può essere 0')
      
      if type(numeratore) != int :
         raise TypeError (' il numeratore devve essere intero')
      
      if type(denominatore) != int :
         raise TypeError (' il denominatore devve essere intero')
         
      divisore = gcd ( numeratore, denominatore)
      
      self.numeratore = numeratore // divisore  
      self.denominatore = denominatore // divisore   
      
   def __str__(self):
   
        return f"{self.numeratore}/{self.denominatore}"  
         
         
   def __add__ (self, other) :
      
      new_n = self.numeratore * other.denominatore + other.numeratore * self.denominatore
      new_d = self.denominatore * other.denominatore
      return Fraction (new_n , new_d)
      
      
   def __sub__ (self, other) :
      
      new_n = self.numeratore * other.denominatore - self.denominatore * other.numeratore
      new_d = self.denominatore * other.denominatore
      return Fraction (new_n , new_d)
      
   def __mul__ (self, other) :
      
      new_n = self.numeratore  * other.numeratore
      new_d = self.denominatore * other.denominatore
      return Fraction (new_n , new_d)
      
   def __truediv__ (self, other) :
      
      new_n = self.numeratore  * other.denominatore
      new_d = self.denominatore * other.numeratore
      return Fraction (new_n , new_d)
   
   
def test_fraction (self) :
      
      print ( self.numeratore, self.denominatore)
      print ( other.numeratore, other.denominatore)
      
      divisore = gcd ( self.n, self.d)
      if ( self.numeratore * divisore ) == self.n and ( self.denominatore * divisore ) == self.d :
         print (' la riduzione è avvenuta con successo')
      
      else : 
         print (' la riduzione è fallita')
   
      
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
      

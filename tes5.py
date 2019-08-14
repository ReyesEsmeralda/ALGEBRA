from numpy import *
from GaussPivoteo import *
a = array([[2, 3, 7], [-2, 5, 6], [8, 9, 4]], float)
b = array([[3], [5], [8]], float)
x = gauss(a,b)
print(x)
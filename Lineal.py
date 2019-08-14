import numpy as np

def gaussjordan1(a,b):
    n=len(b)
    c=np.concatenate([a,b],axis=1)
    for e in range(n):
        t=c[e,e]
        for j in range(e,n+1):
            c[e,j]=c[e,j]/t
        for i in range(n):
            if i!=e:
                t=c[i,e]
                for j in range(e,n+1):
                    c[i,j]=c[i,j]-t*c[e,j]
    x=c[:,n]
    return x

def gaussjordan2(a,b):
    n=len(b)
    c=np.concatenate([a,b],axis=1)    #Con un print va a imprimir todo
    for e in range(n):
        c[e,e:]=c[e,e:]/c[e,e]
        for i in range(n):
            if i!=e:
                c[i,e:]=c[i,e:]-c[i,e]*c[e,e:]  #Con un print va a imprimir las sumas
    x=c[:,n]
    return x

def gaussjordanIn(a,b):
    n=len(b)
    I=np.identity(n)
    c=np.concatenate([a,b],axis=1)
    c=np.concatenate([c,I],axis=1)
    #print (c)
    for e in range(n):
        t=c[e,e]
        for j in range(e,2*n+1):
            c[e,j]=c[e,j]/t
        for i in range(n):
            if i!=e:
                t=c[i,e]
                for j in range(e,2*n+1):
                    c[i,j]=c[i,j]-t*c[e,j]
    Inv=c[:,n+1:]
    return Inv



def gaussjordan1(a,b):
    n=len(b)
    c=np.concatenate([a,b],axis=1)
    for e in range(n):
        t=c[e,]
        for j in range(e,n+1):
            c[e,j]=c[e,j]/t
        for i in range(e,n):
            if i!=e:
                t=c[i,e]
                for j in range(e,n+1):
                    c[i,j]=c[i,j]-t*c[e,j]
    x=c[:,n]
    return x


def gauss1(a,b):
    n = len(b)
    c = np.concatenate([a, b], axis=1)
    for e in range(n):
        t = c[e, e]
        for j in range(e, n + 1):
            c[e, j] = c[e, j] / t
        for i in range(e,n):
            if i != e:
                t = c[i, e]
                for j in range(e, n + 1):
                    c[i, j] = c[i, j] - t * c[e, j]
        x=np.zeros([n,1])
        x[n-1]=c[n-1,n]
        for i in range(n-2,-1,-1):
            s=0
            for j in range(i + 1,n):
             s=s + c[i,j]*x[j]
            x[i]=c[i,n]-s
    return x
Fin de la conversación de chat
Escribe un mensaje...


def gauss1(a,b):
    n = len(b)
    c = np.concatenate([a, b], axis=1)
    for e in range(n):
        t = c[e, e]
        for j in range(e, n + 1):
            c[e, j] = c[e, j] / t
        for i in range(e,n):
            if i != e:
                t = c[i, e]
                for j in range(e, n + 1):
                    c[i, j] = c[i, j] - t * c[e, j]
        x=np.zeros([n,1])
        x[n-1]=c[n-1,n]
        for i in range(n-2,-1,-1):
            s=0
            for j in range(i + 1,n):
             s=s + c[i,j]*x[j]
            x[i]=c[i,n]-s
    return x
Fin de la conversación de chat
Escribe un mensaje...


def jacobi(a,b,x):
    n=len(x)
    t=x.copy()
    for i in range(n):
        s=0
        for j in range(n):
            if i!=j:
                s=s + a [i,j]* t[j]
        x[i]=(b[i] -s) / a[i,i]
    return x
def jacobim(a,b,x,e,m):
    n=len(x)
    t=x.copy()
    for k in range(m):
        x=jacobi(a,b,x)
        d=np.linalg.norm(np.array(x)-np.array(t))
        if d<e:
            return[x,k]
        else:
            t=x.copy()
    return[[],m]


    def gausseidel(a,b,x):
        n=len(x)    
        for i in range(n):  
            s=0
            for j in range(n):
                if i!=j:
                    s=s+a[i,j]*x[j] 
                    x[i]=(b[i]-s)/a[i,i]    
                    return x


#Ejercicio 1
a = array([[3, 0, -1], [-7/5, 2, -1], [-1, 1, (-7/5 + 1)]], float)
print(a)
b = array([[5], [2], [1]], float)
print(b)
c = concatenate([a, b], axis=1)
print(c)
c[0, 0:] = c[0, 0:] / c[0, 0]
print(c)
c[1, 0:] = c[1, 0:] - c[1, 0] * c[0, 0:]
c[2, 0:] = c[2, 0:] - c[2, 0] * c[0, 0:]
print(c)
c[1, 1:] = c[1, 1:] / c[1, 1]
print(c)
c[0, 1:] = c[0, 1:] - c[0, 1] * c[1, 1:]
c[2, 1:] = c[2, 1:] - c[2, 1] * c[1, 1:]
print(c)
c[2, 2:] = c[2, 2:] / c[2, 2]
print(c)
c[0, 2:] = c[0, 2:] - c[0, 2] * c[2, 2:]
c[1, 2:] = c[1, 2:] - c[1, 2] * c[2, 2:]
print(c)
x = c[:, 3]
print(x)

x = gaussjordan1(a, b)
print(x)
r = dot(a, x)
print(r)

#Ejercicio 2
from Lineal import *
a = array([[1/2, 1/3, 1/4], [2/3, 2/4, 2/5], [3/4, 3/5, 3/6]], float)
b = array([[2], [4], [6]], float)
c = concatenate([a, b], axis=1)
print(c)
print(gaussjordan1(a, b))

from GaussBasico import *
a = array([[1/2, 1/3, 1/4, 1/5], [2/3, 2/4, 2/5, 2/6], [3/4, 3/5, 3/6, 3/7], [4/5, 4/6, 4/7, 4/8]], float)
b = array([[2], [4], [6], [8]], float)
c = concatenate([a, b], axis=1)
print(c)
x = gauss1(a, b)
print(x)

#Ejercicio 3
from Lineal import *
a = array([[1, 1/2, 1/3], [1/4, 1/5, 1/6], [1/7, 1/8, 1/9]], float)
b = array([[4], [5], [6]], float)
c = concatenate([a, b], axis=1)
print(c)
x = gaussjordan2(a, b)
print(x)
print(inversa(a, b))

#Ejercicio 4
from numpy import *
a = array([[0.52, 0.20, 0.25], [0.30, 0.50, 0.2], [0.18, 0.30, 0.55]], float)
b = array([[60], [50], [40]], float)
c = concatenate([a, b], axis=1)
print(c)

from Jacobi import *
x = array([[1], [1], [1]], float)
[x, k] = jacobim(a, b, x, 0.0001, 200)
print(x)
print(k)

from GaussSeidel import *
x = array([[1], [1], [1]], float)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)

from GaussSeidel import *
x = array([[1], [1], [1]], float)
[x, k] = gaussseidelm(a, b, x, 0.0001, 20)
print(x)
print(k)

#Ejercicio 5
from numpy import *
a = array([[4, 2, 5], [2, 5, 1], [2, 4, 3]], float)
b = array([[18], [27.30], [16.20]], float)
c = concatenate([a, b], axis=1)
print(c)

from Jacobi import *
x = array([[1], [1], [1]], float)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)

from GaussSeidel import *
x = array([[1], [1], [1]], float)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)

from GaussSeidel import *
x = array([[1], [1], [1]], float)
[x, k] = gaussseidelm(a, b, x, 0.0001, 20)
print(x)
print(k)

#Ejercicio 1
a = array([[3, 0, -1], [-7/5, 2, -1], [-1, 1, (-7/5 + 1)]], float)
print(a)
b = array([[5], [2], [1]], float)
print(b)
c = concatenate([a, b], axis=1)
print(c)
c[0, 0:] = c[0, 0:] / c[0, 0]
print(c)
c[1, 0:] = c[1, 0:] - c[1, 0] * c[0, 0:]
c[2, 0:] = c[2, 0:] - c[2, 0] * c[0, 0:]
print(c)
c[1, 1:] = c[1, 1:] / c[1, 1]
print(c)
c[0, 1:] = c[0, 1:] - c[0, 1] * c[1, 1:]
c[2, 1:] = c[2, 1:] - c[2, 1] * c[1, 1:]
print(c)
c[2, 2:] = c[2, 2:] / c[2, 2]
print(c)
c[0, 2:] = c[0, 2:] - c[0, 2] * c[2, 2:]
c[1, 2:] = c[1, 2:] - c[1, 2] * c[2, 2:]
print(c)
x = c[:, 3]
print(x)

x = gaussjordan1(a, b)
print(x)
r = dot(a, x)
print(r)

#Ejercicio 2
from Lineal import *
a = array([[1/2, 1/3, 1/4], [2/3, 2/4, 2/5], [3/4, 3/5, 3/6]], float)
b = array([[2], [4], [6]], float)
c = concatenate([a, b], axis=1)
print(c)
print(gaussjordan1(a, b))

from GaussBasico import *
a = array([[1/2, 1/3, 1/4, 1/5], [2/3, 2/4, 2/5, 2/6], [3/4, 3/5, 3/6, 3/7], [4/5, 4/6, 4/7, 4/8]], float)
b = array([[2], [4], [6], [8]], float)
c = concatenate([a, b], axis=1)
print(c)
x = gauss1(a, b)
print(x)

#Ejercicio 3
from Lineal import *
a = array([[1, 1/2, 1/3], [1/4, 1/5, 1/6], [1/7, 1/8, 1/9]], float)
b = array([[4], [5], [6]], float)
c = concatenate([a, b], axis=1)
print(c)
x = gaussjordan2(a, b)
print(x)
print(inversa(a, b))

#Ejercicio 4
from numpy import *
a = array([[0.52, 0.20, 0.25], [0.30, 0.50, 0.2], [0.18, 0.30, 0.55]], float)
b = array([[60], [50], [40]], float)
c = concatenate([a, b], axis=1)
print(c)

from Jacobi import *
x = array([[1], [1], [1]], float)
[x, k] = jacobim(a, b, x, 0.0001, 200)
print(x)
print(k)

from GaussSeidel import *
x = array([[1], [1], [1]], float)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)

from GaussSeidel import *
x = array([[1], [1], [1]], float)
[x, k] = gaussseidelm(a, b, x, 0.0001, 20)
print(x)
print(k)

#Ejercicio 5
from numpy import *
a = array([[4, 2, 5], [2, 5, 1], [2, 4, 3]], float)
b = array([[18], [27.30], [16.20]], float)
c = concatenate([a, b], axis=1)
print(c)

from Jacobi import *
x = array([[1], [1], [1]], float)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)
x = jacobi(a, b, x); print(x)

from GaussSeidel import *
x = array([[1], [1], [1]], float)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)
x = gaussseidel(a, b, x); print(x)

from GaussSeidel import *
x = array([[1], [1], [1]], float)
[x, k] = gaussseidelm(a, b, x, 0.0001, 20)
print(x)
print(k)
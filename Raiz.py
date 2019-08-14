def MetodoBiseccion(
f= x-math.sin(x),
x0 = -2,
x1 = 2,
t = 0.4):
    if f(x0)*f(x1)<0:
        xr = x0
        while abs (f(xr))>t:
            xr=(x0+x1)/2
            if f(x0)*f(x1)<0:
                x1=xr
            else:
                x0=xr
        return xr
    else:
        return 'No hay cambio de signo.'
    print(Respuesta es: (xr))
input("Pulse INTRO para finalizar...")

def newtonIterationFunction(x):
    return  math.pow(x,4) - 10(math.pow(x,3)) + 3(math.pow(x,2)) + x + 23
x = 0

for i in range(100):
    print "Iteraciones: ",str(i),"Valor aproximado: ", str(x)
    xold = x
    x = newtonIterationFunction(x)
    if x == xold:
        print "Solución encontrada!"
        break

input("Pulse INTRO para finalizar...")

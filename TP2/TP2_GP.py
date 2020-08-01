import numpy as np
import matplotlib.pyplot as plt


def J(A, b, v):
    X = A @ v
    J = 0.5 * (X @ v) - (b @ v)
    return J


def gradJ(A, b, v):
    grad = A @ v - b
    return grad

def g(x):
    return np.maximum(np.zeros_like(x), 1. - 100 * (x - 0.7)**2)

#------ Methode de gradient - fonction quadratique -------#
def gradient(A, b):

    #--- pas optimal
    n = np.shape(A)[1]
    [ValPropre, VecPropre] = #---- A COMPLETER -----#
    pas = #---- A COMPLETER -----#

    #----------------
   

	#---- A COMPLETER -----#


    return v


def gradientcontrainte(A, b, gvec): # gvec definit la contrainte

    #--- pas optimal
    n = np.shape(A)[1]
    [ValPropre, VecPropre] = #---- A COMPLETER -----#
    pas = #---- A COMPLETER -----#

    #----------------

	#---- A COMPLETER -----#

    return v

#-- mesh
n = 100
x = np.linspace(0., 1., n + 2)

#-- matrice
dx = 1. / (n + 1)
A = #---- A COMPLETER -----#
b = #---- A COMPLETER -----#

#-- contrainte
gvec = g( x[1:-1])

#-- resolution
u = gradientcontrainte( A, b, gvec)

print("verif contrainte = ", np.prod((A @ u - b) * (u != gvec)))

#--- plot

#---- A COMPLETER -----#

    #---



test3()
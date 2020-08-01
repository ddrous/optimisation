#  -*- coding:Latin-1 -*-

from math import *
import numpy as np
from scipy import *
from random import *
from pylab import * # pour utiliser plot
import codecs

# TP : EXO 2

#----------------------------------------------------------------------------------------#
# Optim - minimisation


def f(x,y):
    return (x+1)**2 + (y-2)**2

def gradfeps(x,eps):
    return np.array([2*(x[0]+1)-2/eps*(x[1]-x[0]+1),2*(x[1]-2)+2/eps*(x[1]-x[0]+1)])

def gradientpasfixe(xinit, gradJ,eps):

    pas = 1e-4
    #----------------
    nbit = 0
    x = xinit.copy()
    xold = x.copy()

    grad = gradJ(x,eps)
    err = 1.
    xlist = [list(x)]


    while (err > 1e-12 and nbit < 1e5):
        xold[:] = x # commande [:] permettant de copier une liste dans une autre, independamment 
        x -= #----------- A COMPLETER -------------#
        xlist.append(list(x))
        grad[:] = #----------- A COMPLETER -------------#
        err = #----------- A COMPLETER -------------#
        nbit += 1


    return x, np.array( xlist), nbit

def test():
    eps= #----------- A COMPLETER -------------#

    xinit = np.array([-2.,2.]) # initialization dans la mŽthode de gradient
    x, xlist, nbit = #----------- A COMPLETER -------------#
    print(x)
    print(nbit)

    plt.figure()
    n = 80
    x = np.linspace(-5.,5,n)
    y = np.linspace(-5.,5,n)
    xx, yy = np.meshgrid(x,y)
    fval = f(xx,yy)
    plt.contourf(x,y,fval,50) # lignes de niveau de la fonction f en couleur
    #- fonction f penalisee
    fepsval = f(xx,yy) + 1./eps * (yy-xx+1)**2
    plt.contour(x,y,fepsval,20,colors='k') # lignes de niveau de la fct pŽnalisŽe en noir
    #- iteres
    plt.plot(xlist[:, 0], xlist[:, 1], 'r-*', label="gradient penalise")
    plt.legend(loc='best')
    plt.show()

test()

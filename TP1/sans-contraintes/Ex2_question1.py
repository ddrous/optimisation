#  -*- coding:Latin-1 -*-

from math import *
import numpy as np
from scipy import *
from random import *
from pylab import * # pour utiliser plot
import codecs

# TP : EXO 2: QUESTION 1

#----------------------------------------------------------------------------------------#
# Optim - minimisation


def f(x,y):
    return (x+1)**2 + (y-2)**2

def feps(x, eps):
    return (x[0]+1)**2 + (x[1]-2)**2 + (1/eps)*(x[1]-x[0]+1)**2

def gradfeps(x, eps):
    return np.array([2*(x[0]+1)-2/eps*(x[1]-x[0]+1),2*(x[1]-2)+2/eps*(x[1]-x[0]+1)])

def hessfeps(x, eps):
    return np.array([[2+2/eps, -2/eps], [-2/eps, 2+2/eps]])

def gradientpasfixe(xinit, gradJ, eps):

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
        x -= pas * grad
        xlist.append(list(x))
        grad[:] = gradJ(x,eps)
        err = np.linalg.norm(x - xold)
        nbit += 1

    return x, np.array( xlist), nbit


# Fonction pour determiner le pas optimal par la methode de Newton
def newtonmin(x, d, feps, gradfeps, hessfeps, eps):
    # Fonction a minimiser
    def q(rho, eps):
        return feps(x + rho*d, eps)
    def q_prime(rho, eps):
        return gradfeps(x + rho*d, eps)@d.T
    def q_second(rho, eps):
        return (hessfeps(x + rho*d, eps)@d) @ d.T

    rho = 0 # position initiale 
    rho_suiv = rho - q_prime(rho, eps)/q_second(rho, eps) # position suivante
    compteur = 0
    while (abs(rho_suiv-rho)>1e-10 and compteur<100):
        rho = rho_suiv
        rho_suiv = rho - q_prime(rho, eps)/q_second(rho, eps)
        compteur=compteur+1

    return rho

def gradientpasoptimal(xinit, J, gradJ, HessJ, eps):

    nbit = 0
    x = xinit.copy()
    xold = x.copy()

    grad = gradJ(x,eps)
    err = 1.
    xlist = [list(x)]
    paslist = []

    while (err > 1e-12 and nbit < 1e5):
        xold[:] = x
        pas = newtonmin(x, -grad, J, gradJ, HessJ, eps)     # pas optimal
        paslist.append(pas)
        x -= pas * grad
        xlist.append(list(x))
        grad[:] = gradJ(x,eps)
        err = np.linalg.norm(x - xold)
        nbit += 1

    return x, np.array(xlist), np.array(paslist), nbit


# Fonction pour faire les graphes
def plot_grad(xlist, eps, type_methode):
    plt.figure()
    n = 80
    x = np.linspace(-5.,5,n)
    y = np.linspace(-5.,5,n)
    xx, yy = np.meshgrid(x,y)

    #- lignes de niveau de la fonction f en couleur
    fval = f(xx,yy)
    img = plt.contourf(x,y,fval,50)
    plt.colorbar(img)
    
    #- lignes de niveau de la fct p�nalis�e en noir
    fepsval = f(xx,yy) + 1./eps * (yy-xx+1)**2
    plt.contour(x,y,fepsval,20,colors='k') 
   
    #- iteres
    plt.plot(xlist[:, 0], xlist[:, 1], 'r-*', label=u"gradient penalis\u00E9")
    
    #- contraintes
    plt.plot(x, x-1, '--',  color="orange", label="contrainte $y=x-1$")
    
    if type_methode == "fixe":
        plt.title("Gradient a pas fixe")
    elif type_methode == "optimal":
        plt.title("Gradient a pas optimal")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.legend(loc='best')
    plt.show()


def plt_erreur(xlist, type_methode):
    erreur = []
    for i in range(len(xlist)):
        erreur.append(np.linalg.norm(xlist[i]-np.array([1,0])))
    
    plt.figure()
    plt.plot(erreur, 'g.-', label="$\Vert (1,0)-x \Vert$")

    if type_methode == "fixe":
        plt.title("Erreur de convergence - pas fixe")
    elif type_methode == "optimal":
        plt.title("Erreur de convergence - pas optimal")
    plt.legend(loc='best')
    plt.show()

def test_gradfixe(eps):
    xinit = np.array([-2.,2.]) # initialization dans la methode de gradient

    x, xlist, nbit = gradientpasfixe(xinit, gradfeps, eps)
    
    np.set_printoptions(precision = 4)
    print("\nGRADIENT A PAS FIXE")
    print("minimum atteint en:", x)
    print("nombre d'iterations: ", nbit)

    plot_grad(xlist, eps, "fixe")
    plt_erreur(xlist, "fixe")


def test_gradoptimal(eps):
    xinit = np.array([-2.,2.]) # initialisation dans la methode de gradient
    
    x, xlist, paslist, nbit = gradientpasoptimal(xinit, feps, gradfeps, hessfeps, eps)
    
    np.set_printoptions(precision = 4)  # Pour afficher 4 chiffres apres la virgule
    print("\nGRADIENT A PAS OPTIMAL")
    print("minimum atteint en:", x)
    print("les differents pas optimaux:", paslist)
    print("nombre d'iterations:", nbit)

    plot_grad(xlist, eps, "optimal")
    plt_erreur(xlist, "optimal")

if __name__ == '__main__':
    eps = 0.001                # epsilon

    test_gradfixe(eps)                      # test methode du gradient a pas fixe
    test_gradoptimal(eps)                   # test methode du gradient a pas optimale

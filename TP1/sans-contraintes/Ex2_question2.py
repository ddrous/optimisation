#  -*- coding:Latin-1 -*-

from math import *
import numpy as np
from scipy import *
from random import *
from pylab import * # pour utiliser plot
import codecs

# TP : EXO 2 : QUESTION 2

#----------------------------------------------------------------------------------------#
# Optim - minimisation

def f(x,y):
    return (x-a)**2 + (y-b)**2

def feps(x, eps):
    return (x[0]-a)**2 + (x[1]-b)**2 + (1/eps)*(2*x[0]**2 + x[1]**2 - 1)**2

def gradfeps(x, eps):
    return np.array([2*(x[0]-a) + (8*x[0]/eps)*(2*x[0]**2+x[1]**2-1), 2*(x[1]-b) + (4*x[1]/eps)*(2*x[0]**2+x[1]**2-1)])

def hessfeps(x, eps):
    return np.array([[2 + (8/eps)*(6*x[0]**2+x[1]**2-1), 16*x[0]*x[1]/eps], [16*x[0]*x[1]/eps, 2 + (4/eps)*(2*x[0]**2+3*x[1]**2-1)]])


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
    fepsval = f(xx,yy) + 1./eps * (2*xx**2 + yy**2 - 1)**2
    plt.contour(x,y,fepsval,20,colors='k') 
   
    #- iteres
    plt.plot(xlist[:, 0], xlist[:, 1], 'r-*', label=u"gradient penalis\u00E9")
    
    #- point (a, b)
    plt.plot(a, b, '+',color="beige", label="point (a, b)")

    #- contraintes - equation paremtrique de l'ellipse
    t = np.linspace(-5.,5,n)
    plt.plot(np.cos(t)/np.sqrt(2), np.sin(t), '--',  color="orange", label="ellipse $2x^2+y^2=1$")
    

    plt.title("Gradient a pas optimal")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.legend(loc='best')
    plt.show()


def test_gradoptimal(eps):
    xinit = np.array([-2.,2.]) # initialization dans la m�thode de gradient
    
    x, xlist, paslist, nbit = gradientpasoptimal(xinit, feps, gradfeps, hessfeps, eps)
    
    print("\nGRADIENT A PAS OPTIMAL")
    print("nombre d'iterations:", nbit)
    np.set_printoptions(precision = 4)  # Pour afficher 4 chiffres apres la virgule
    print("point (a,b) du plan:", np.array([a, b]))
    print("minimum atteint en:", x)
    # print("les differents pas optimaux:", paslist)
    print("distance a l'ellipse: %.4f"%np.linalg.norm(x-np.array([a, b])))

    plot_grad(xlist, eps, "optimal")


# Coordonees du point dont on veut determiner la distance a l'elipse
a = -2
b = 2

if __name__ == '__main__':
    eps = 0.001                     # epsilon

    test_gradoptimal(eps)

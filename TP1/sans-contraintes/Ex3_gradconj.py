#  -*- coding:Latin-1 -*-

from math import *
from numpy import *
import numpy as np
from scipy import *
from random import *
from pylab import * # pour utiliser plot 
import time # Pour mesurer les temps d'execution

# TP 1 : EXO 3

# Fonction J en dimmendion 2 pour tracer sur le graphe
def J_2D(x1, x2):
	return 2*x1**2 + 2*x2**2 - 2*x1*x2 - x1 - x2

# gradient de J en dimmendion 2
def gradJ_2D(x1, x2):
	return 4*x1-2*x2-1, 4*x2-2*x1-1

# Fonction J en dimension n quelconque
def J(x, n):
	A = matrixA(n)
	b = np.ones((n), dtype=np.float64)
	return (x.T@A@x)/2 - b.T@x

# Gradient de J en dimension n quelconque
def gradJ(x, n):
	A = matrixA(n)
	b = np.ones((n), dtype=np.float64)
	return A@x - b

# Matrice Hesseisnne de J
def HessJ(x, n):
	return matrixA(n)

#----------------------------------------------------------------------------------------#

def matrixA(n):
# Definition de la matrice A
	A=np.diag(4.*np.ones(n)) + np.diag(-2*np.ones(n-1), k=1) + np.diag(-2*np.ones(n-1), k=-1)
	return A

#----------------------------------------------------------------------------------------#
# Programme de minimisation de J sur R^n par la méthode du gradient conjugué 
def gradConj(xinit, n):
# n est la dimension de la matrice et x0 l initialization
	A = matrixA(n)
	b = np.ones((n), dtype=np.float64)

	# Initialisation
	r = A@xinit - b
	d = -r
	x = xinit.copy()
	xlist = [list(x)]

	# Conditions d'arret
	err = 1.		# norme du residu
	nbit = 0

	while (err > 1e-12 and nbit < 1e5):
		# xold = x.copy()
		
		rho = (-r.T@d) / (d.T@A@d)		# Atention aux erreur de precedence en Python3 car * << @ << /
		x = x + rho*d

		rold = r.copy()
		r = A@x - b
		beta = (r.T@r) / (rold.T@rold)
		d = -r + beta*d

		xlist.append(list(x))
		err = np.linalg.norm(r)
		nbit += 1
	
	return x, np.array(xlist), nbit

# Programme de minimisation de J sur R^n par la méthode du gradient a pas fixe 
def gradPasFixe(xinit, gradJ, n):
	nbit = 0
	err = 1.

	pas = 1e-1
	x = xinit.copy()
	xold = x.copy()
	grad = gradJ(x, n)
	xlist = [list(x)]

	while (err > 1e-12 and nbit < 1e5):
		xold[:] = x
		x -= pas * grad
		xlist.append(list(x))
		grad[:] = gradJ(x, n)
		err = np.linalg.norm(x - xold)
		nbit += 1

	return x, np.array(xlist), nbit	

# Fonction pour determiner le pas optimal par la methode de Newton
def newtonmin(x, d, gradf, hessf, n):
	# derivee de la fonction "q" a minimiser
	def q_prime(rho):
		return gradf(x + rho*d, n)@d.T
	def q_second(rho):
		return (hessf(x + rho*d, n)@d) @ d.T

	rho = 0 # position initiale 
	rho_suiv = rho - q_prime(rho)/q_second(rho) # position suivante
	compteur = 0

	while (abs(rho_suiv-rho)>1e-2 and compteur<100):
		rho = rho_suiv
		rho_suiv = rho - q_prime(rho)/q_second(rho)
		compteur = compteur+1

	return rho

# Fonction pour calculer le pas optimal pour une fonction quadratique avec A sym. def. pos.
def calculPasOptimal(x, gradJ):
	n = len(x)
	A = matrixA(n)
	b = np.ones_like(x)
	
	d = gradJ(x, n)	# d=A@x-b
	return (d.T@d)/(d.T@A@d)

# Programme de minimisation de J sur R^n par la méthode du gradient a aps optimale 
def gradPasOptimal(xinit, gradJ, HessJ, n):
	nbit = 0
	x = xinit.copy()
	xold = x.copy()

	grad = gradJ(x,n)
	err = 1.
	xlist = [list(x)]

	while (err > 1e-12 and nbit < 1e5):
		xold[:] = x
		# pas = newtonmin(x, -grad, gradJ, HessJ, n)     # pas optimal
		pas = calculPasOptimal(x, gradJ)     # pas optimal
		x -= pas * grad
		xlist.append(list(x))
		grad[:] = gradJ(x, n)
		err = np.linalg.norm(x - xold)
		nbit += 1

	return x, np.array(xlist), nbit


# Pour dessiner les lignes de niveau et le gradient
def plot_contours(ax):
	x1 = np.arange(-4.0, 4.0, 0.1)
	x2 = np.arange(-4.0, 4.0, 0.1)

	xx1,xx2 = np.meshgrid(x1,x2);

	z = J_2D(xx1, xx2)
	h = ax.contourf(x1,x2,z)

	U, V = gradJ_2D(xx1, xx2)
	Q = ax.quiver(x1, x2, U, V, units='width')

# Pour afficher les resultats 
def print_results(x, nbit, methode):
	if methode == "conj":
		print("GRADIENT CONJUGUE")
	elif methode == "fixe":
		print("GRADIENT A PAS FIXE")
	elif methode == "opti":
		print("GRADIENT A PAS OPTIMAL")

	print("minimum atteint en:", x)
	print("nombre d'iterations:", nbit)

# Pour dessiner les iteres par la methode du gradient conjuge
def plot_gradConj(ax, x0, n, xexact=None):
	x, xlist, nbit = gradConj(x0, n)
	print_results(x, nbit, "conj")
	ax.plot(xlist[:, 0], xlist[:, 1], 'r-*', label=u"gradient conjug\u00E9")

# Pour dessiner les iteres par la methode du gradient conjuge
def plot_gradPasFixe(ax, x0, n):
	x, xlist, nbit = gradPasFixe(x0, gradJ, n)
	print_results(x, nbit, "fixe")	
	ax.plot(xlist[:, 0], xlist[:, 1], '-*', color="magenta", alpha=1, label=u"gradient a pas fixe")

# Pour dessiner les iteres par la methode du gradient conjuge
def plot_gradPasOptimal(ax, x0, n):
	x, xlist, nbit = gradPasOptimal(x0, gradJ, HessJ, n)
	print_results(x, nbit, "opti")
	ax.plot(xlist[:, 0], xlist[:, 1], '-*', color="orange", alpha=1, label=u"gradient a pas optimal")

# Fonction pour comparer les vitesses de convergence
def plot_error(ax, x0, n, xexact):
	x, xlist, nbit = gradConj(x0, n)
	print_results(x, nbit, "conj")
	xerr1 = [np.linalg.norm(x-xexact) for x in xlist]

	x, xlist, nbit = gradPasFixe(x0, gradJ, n)
	print_results(x, nbit, "fixe")	
	xerr2 = [np.linalg.norm(x-xexact) for x in xlist]

	x, xlist, nbit = gradPasOptimal(x0, gradJ, HessJ, n)	
	print_results(x, nbit, "optimal")
	xerr3 = [np.linalg.norm(x-xexact) for x in xlist]

	len1 = np.shape(xerr1)[0]
	len2 = np.shape(xerr2)[0]
	len3 = np.shape(xerr3)[0]
	plotlen = len2		# Pour limiter la taille des erreurs qu'on graphe

	ax.plot(xerr1[:plotlen], 'r-*', label=u"gradient conjug\u00E9")
	ax.plot(xerr2[:plotlen], '-*', color="magenta", alpha=0.3, label=u"gradient a pas fixe")	
	ax.plot(xerr3[:plotlen], '-*', color="orange", alpha=0.5, label=u"gradient a pas optimal")


#----------------------------------------------------------------------------------------#
# Function pour afficher les resultats
def plot(n):
	# plt.style.use("seaborn")
	ax = plt.subplot(1, 1, 1)

	# # Tracons les contours et le gradient
	plot_contours(ax)

	xinit = np.zeros((n))
	if n == 2:					# initialisation pour la dimension 2
		xinit[0] = -0.9
		xinit[-1] = -3.5

	print("\nINFO:")
	print("dimension du probleme:", n)
	print("position initiale:", xinit, "\n")

	# # Trace les iteres par la methode du gradient conjugue 
	start = time.time()
	plot_gradConj(ax, xinit, n)
	print("temps d'execution: %.3f"%(time.time()-start), "\n")

	# # Tracons les iteres par la methode du gradient a pas constant 
	start = time.time()
	plot_gradPasFixe(ax, xinit, n)
	print("temps d'execution: %.3f"%(time.time()-start), "\n")

	# Tracons des iteres par la methode du gradient a pas optimal 
	start = time.time()
	plot_gradPasOptimal(ax, xinit, n)
	print("temps d'execution: %.3f"%(time.time()-start), "\n")

	# # Tracons les erreurs de convergence
	# A = matrixA(n)
	# b = np.ones((n), dtype=np.float64)
	# xexact = np.linalg.solve(A, b)
	# plot_error(ax, xinit, n, xexact)
	
	plt.xlim(-4, 4)
	plt.ylim(-4, 4)
	# plt.title("Vitesses de convergence")
	plt.title("Comparaison des methodes de gradient")
	plt.legend()
	if n == 2:
		plt.show()


if __name__ == '__main__':
	n = 2
	plot(n)
#  -*- coding:Latin-1 -*-

from math import *
import numpy as np
from scipy import *
from random import *
from pylab import * # pour utiliser plot

# TP : EXO 1

#----------------------------------------------------------------------------------------#
# Optim - minimisation

def f(x):
	return x/8 + np.sqrt((6-x)**2+4)/3

def f_prime(x):
	return 1/8 + (1/3)*(x-6) / np.sqrt((6-x)**2+4)

def f_second(x):
	return (4/3) / (((6-x)**2+4)*np.sqrt((6-x)**2+4))

xx = [] #liste
# x = 3.202			# x_0_min pour une initalisation qui converge
x = 7.662			# x_0_max pour une initalisation qui converge
x_suiv = x - f_prime(x)/f_second(x) # position suivante
xx.append(x) # ajouter a une liste
compteur = 0
while (abs(x_suiv-x)>1e-10 and compteur<100):
	x = x_suiv
	x_suiv = x - f_prime(x)/f_second(x)
	xx.append(x) # ajouter a une liste
	compteur=compteur+1

if (compteur==1000):
	print("Nombre maximal d'iterations atteint")
print("Nombre d'iterations effectuees: ", compteur)

xopt =xx[-1]

print(u"le minimiseur trouv\u00e9 est x =", xopt)

xx = np.array(xx) # transforme une liste en tableau

#----------------------------------------------------------------------------------------#
# Traces

#----------------------------------------------------------------------------------------#
# Traces

x=np.linspace(0,8,200) # discretisation de l'intervalle des x

plt.figure()
M = f(x)
plt.plot(x,M,label="Graphe de la fonction temps",linewidth=2)
legend()
for i in range(len(xx)):
	plt.plot(xx[i], f(xx[i]), 'r+', markersize=10)
	plt.ylabel('f(x)')
	plt.xlabel("x")


plt.show()

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
	return x/8+np.sqrt((6-x)**2+4)/3

xx = [] #liste
x0 = 0 # borne gauche de l'intervalle initial
x1 = 6 # borne droite de l'intervalle initial
xx.append((x0+x1)/2) # ajouter a une liste
compteur = 0
tau = 3		# facteur de reduction
while ((x1-x0)>1e-10 and compteur<100):
	xt0 = x0 + 0.5*(1-1/tau)*(x1-x0)
	xt1 = x0 + 0.5*(1+1/tau)*(x1-x0)
	M0 = f(x0)
	M1 = f(x1)
	if M0 < M1:
		x1 = xt1
	elif M0 > M1:
		x0 = xt0
	else:
		x0 = xt0
		x1 = xt1
	xx.append((x0+x1)/2) # ajouter a une liste
	compteur=compteur+1

if (compteur==1000):
	print("Nombre maximal d'iterations atteint")
print("Nombre d'iterations effectuees: ", compteur)

xopt =xx[-1]

print(u"le minimiseur trouv\u00e9 est x =", xopt)

xx = np.array(xx) # transforme une liste en tableau

#----------------------------------------------------------------------------------------#
# Traces

x=np.linspace(0,6,200) # discretisation de l'intervalle des x

plt.figure()
M = f(x)
plt.plot(x,M,label="Graphe de la fonction temps",linewidth=2)
legend()
for i in range(len(xx)):
	plt.plot(xx[i], f(xx[i]), 'r+', markersize=10)
	plt.ylabel('f(x)')
	plt.xlabel("x")

plt.show()

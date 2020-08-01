#  -*- coding:Latin-1 -*-

from math import *
import numpy as np
from scipy import *
from random import *
from pylab import * # pour utiliser plot

# TP : EXO 1

#----------------------------------------------------------------------------------------#
# Optim - minimisation

def fun(x):
	return x/8+np.sqrt((6-x)**2+4)/3

xx = [] #liste
x0 = 0 # borne gauche de l'intervalle initial
x1 = 6 # borne gauche de l'intervalle initial
xx.append((x0+x1)/2) # ajouter a une liste
compteur = 0
while ((x1-x0)>1e-10 and compteur<100):
	xt0= #----------- A COMPLETER -------------#
	xt1= #----------- A COMPLETER -------------#
	M0= #----------- A COMPLETER -------------#
	M1= #----------- A COMPLETER -------------#
	if M0 < M1:
		#----------- A COMPLETER -------------#
	else:
		#----------- A COMPLETER -------------#
	xx.append((x0+x1)/2) # ajouter a une liste
	compteur=compteur+1

if (compteur==1000):
	print("Nombre maximal d'iterations atteint")
print("Nombre d'iterations : ",compteur)

xopt= #----------- A COMPLETER -------------#

print(u"le minimiseur trouv\u00e9 est x=",xopt)

xx = np.array(xx) # transforme une liste en tableau

#----------------------------------------------------------------------------------------#
# Traces

x=np.linspace(0,6,200) # discretisation de l'intervalle des x

plt.figure()
M = fun(x)
plt.plot(x,M,label="Graphe de la fonction T",linewidth=2)
legend()
for i in range(len(xx)):
	plt.plot(xx[i], fun(xx[i]), 'k+', markersize=10)
	plt.ylabel('T(x)')
	plt.xlabel("x")

plt.show()

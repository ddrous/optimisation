#  -*- coding:Latin-1 -*-

from math import *
from numpy import *
#import numpy as np
from scipy import *
from random import *
from pylab import * # pour utiliser plot
 
# TP 1 : EXO 3

#----------------------------------------------------------------------------------------#

def matrixA(n):
# DŽfinition de la matrice A
	A=np.diag(4.*np.ones(n))+ #----------- A COMPLETER -------------#
	return A

#----------------------------------------------------------------------------------------#
# Programme minimisation de J sur R^n par la mŽthode du gradient conjuguŽ 
def gradConj(n,x0):
# n est la dimension de la matrice et x0 l initialization
	A=matrixA(n)
	b=np.ones((n,1))
	#----------- A COMPLETER -------------#
	return y


#----------------------------------------------------------------------------------------#
# Plot Function

x1 = np.arange(-3.0, 3.0, 0.1)
x2 = np.arange(-3.0, 3.0, 0.1)

xx1,xx2 = np.meshgrid(x1,x2);

z = 2*xx1**2 + 2*xx2**2 -2*xx1*xx2-xx1-xx2;

h = plt.contourf(x1,x2,z)
U = 4*xx1-2*xx2-1
V = 4*xx2-2*xx1-1
Q = plt.quiver(x1, x2, U, V, units='width')
# trace des iteres par la mŽthode du gradient a pas constant 
x0= #----------- A COMPLETER -------------#
y=gradPasCst(2,x0)
# representation des iteres sur la figure
#----------- A COMPLETER -------------#
plt.title("Methode du gradient conjugue")
plt.show()

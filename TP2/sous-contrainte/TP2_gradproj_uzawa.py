import numpy as np
import numpy.linalg as nplin
import matplotlib.pyplot as plt
import time as time 


#------ Matrice A -------#
def matrixA(dx, n):
	A = np.diag(2.*np.ones(n)) + np.diag(-1*np.ones(n-1), k=1) + np.diag(-1*np.ones(n-1), k=-1)
	return A / (dx**2)

#------ Vecteur b -------#
def vectorb(n):
	b = np.ones((n), dtype=np.float64)
	return b

#------ Fonction a minimiser J -------#
def J(A, b, v):
    X = A @ v
    J = 0.5 * (X @ v) - (b @ v)
    return J

#------ Gradient de J -------#
def gradJ(A, b, v):
    grad = A @ v - b
    return grad

#------ Contrainte pour le probleme d'optimisation -------#
def ctrt(g, v):
    return g - v

#------ Matrice Jacobienne de la contrainte -------#
def gradctrt(v):
    return np.diag(-np.ones_like(v), k=0)

#------ Lagrangien L -------#
def L(A, b, g, v, mu):
    return J(A, b, v) + mu.T@ctrt(g, v)

#------ Gradient de L en fonction de v -------#
def gradL(A, b, v, mu):
    return gradJ(A, b, v) + gradctrt(v).T@mu

#------ Fonction pout lisser un signal -------#
def smooth(signal):
    ret = signal
    for i in range(10):
        ret = np.convolve(ret, np.array([1/3, 1/3, 1/3]), mode='same')
    return ret

# #------ L'obstacle de base -------#
def g(x):
    return np.maximum(np.zeros_like(x), 1. - 100 * (x - 0.7)**2)

# # ------ Un obstacle en forme de saut d'obstacle -------#
# def g(x):
#     ret = np.zeros_like(x)
#     n_prime = len(x)
#     for i in range(n_prime):
#         if 0.2 <= x[i] <= 0.21:
#             ret[i] = 0.25
#         if 0.4 <= x[i] <= 0.41:
#             ret[i] = 0.3
#         if 0.6 <= x[i] <= 0.61:
#             ret[i] = 0.45
#         if 0.8 <= x[i] <= 0.81:
#             ret[i] = 0.15
#     return ret

#------ Un obstacle en forme de plateau -------#
# def g(x):
#     ret1 = np.where(0.4<=x, np.ones_like(x), np.zeros_like(x))
#     ret2 =  np.where(x<=0.8, ret1, np.zeros_like(x))
#     return smooth(ret2)

# ------ Un obstacle en forme de cuve -------#
# def g(x):
#     ret = 1+x**3-x
#     ret1 = np.where(0.05<=x, ret, np.zeros_like(x))
#     ret2 = np.where(x<=0.95, ret1, np.zeros_like(x))
#     return smooth(ret2)

#------ Methode du gradient a pas optimal sans contrainte -------#
def gradient(A, b):
    val_propres = np.real(np.linalg.eig(A)[0])
    pas = 2 / (min(val_propres) + max(val_propres))     # pas optimal
    nbit = 0
    err = 1.
    v = np.zeros_like(b)    # v initial
    vold = v.copy()
    grad = gradJ(A, b, v)
    vlist = [list(v)]

    while (err > 1e-8 and nbit < 1e6):
        vold[:] = v
        v -= pas * grad
        vlist.append(list(v))
        grad[:] = gradJ(A, b, v)
        err = np.linalg.norm(v - vold)
        nbit += 1

    return v, np.array(vlist), nbit	

#------ Test pour la methode du gradient -------#
def test_gradient(n):
    dx = 1. / (n + 1)
    A = matrixA(dx, n)
    b = vectorb(n)

    # Solution par la methode du gradient a pas fixe
    v, vlist, nbit = gradient(A, b)
    print("arret au bout de", nbit, "iterations")
    print("solution:       ", v)

    # Solution exacte
    vexact = np.linalg.solve(A, b)
    print("solution exacte:", vexact)

    #-- plot de la solution
    x = np.linspace(0., 1., n + 2)
    vfull = np.zeros_like(x)
    vfull[1:-1] = v
    vexactfull = np.zeros_like(x)
    vexactfull[1:-1] = vexact

    plt.plot(x, vfull, "g-", label="solution");
    plt.plot(x, vexactfull, "*", color='purple', label="solution exacte");
    # plt.title("methode de gradient projete")
    plt.legend()
    plt.show()

    # Etude de convergence
    erreurs = [np.linalg.norm(v-vexact) for v in vlist]
    plt.plot(erreurs, "-*", color='purple')
    plt.xlabel(u"it\u00E9rations")
    plt.ylabel("$\Vert v - v_{exact} \Vert$")
    # plt.title("Convergence de la methode de gradient")
    plt.show()

#------ Methode du gradient projete -------#
def gradient_projete(A, b, gvec):
    val_propres = np.real(np.linalg.eig(A)[0])
    pas = 2 / (min(val_propres) + max(val_propres))     # pas optimal
    nbit = 0
    err = 1.
    v = np.zeros_like(b)            # v initial
    v = np.maximum(v, gvec)         # projection
    vold = v.copy()
    grad = gradJ(A, b, v)
    vlist = [list(v)]

    while (err > 1e-8 and nbit < 1e6):
        vold[:] = v
        v -= pas * grad
        v = np.maximum(v, gvec)
        vlist.append(list(v))
        grad[:] = gradJ(A, b, v)
        err = np.linalg.norm(v - vold)
        nbit += 1

    return v, np.array(vlist), nbit	

#------ Test de la methode du gradient projete -------#
def test_gradient_projete(n):
    #-- mesh
    x = np.linspace(0., 1., n + 2)

    #-- matrice
    dx = 1. / (n + 1)
    A = matrixA(dx, n)
    b = vectorb(n)

    #-- contrainte
    gvec = g(x[1:-1])

    #-- resolution
    start = time.time()
    u , ulist, nbit = gradient_projete( A, b, gvec)
    print("====================================\nGRADIENT PROJETE")
    print("temps d'execution: %.4f"%(time.time()-start), "s")

    print("nombre d'iterations:", nbit)
    # print("solution:       ", u)
    # print("solution exacte du probleme sans contraintes:", np.linalg.solve(A, b))

    print("verif contrainte:", np.prod((A @ u - b) * (u != gvec)))                  # doit afficher 0
    print("verif concavite:", np.all(np.round(A@u, 4) >= np.ones_like(u)))          # doit afficher True
    print("verif superiorite a obstacle:", np.all(u - gvec >= -1e-4))               # doit afficher True
    print("====================================")

    #--- plot solution et contrainte sur l'intervalle [0, 1]
    ufull = np.zeros_like(x)
    ufull[1:-1] = u

    plt.plot(x, ufull, "g-", label="solution");
    plt.plot(x, g(x), "r--", label="obstacle");
    plt.title("Methode du gradient projete")
    plt.legend()
    plt.show()

#------ Methode d'uzawa -------#
def uzawa(A, b, gvec): 
    val_propres = np.real(np.linalg.eig(A)[0])
    
    pas = 9.8     # lorsque n < 10
    # pas = float(str(min(val_propres))[:3])                  # pas pour la maximisation (=9.8 pour n >= 10)
    # if(pas <= 0 or pas != 9.8):                             # pour s'assurer que la matrice est def. pos.
    #     print("\n -- pas pour la methode d'uzawa:", pas)
    v = np.zeros_like(b)    # v initial
    mu = np.zeros_like(b)   # mu initial
    vold = v.copy()
    vlist = [list(v)]
    err = 1.
    nbit = 0

    pas_v = 2 / (min(val_propres) + max(val_propres))       # pas pour la minisation

    while (err > 1e-8 and nbit < 1e6):
        vold[:] = v

        # Minimisation du lagrangien par rapport a v
        err_v = 1.
        nbit_v = 0
        vold_v = v.copy()
        while (err_v > 1e-4 and nbit_v < 1e3):
            vold_v[:] = v
            grad = gradL(A, b, v, mu)
            v -= pas_v * grad                       # gradient a pas optimal
            err_v = np.linalg.norm(v - vold_v)
            nbit_v += 1

        # Maximisation du lagrangien par rapport a mu
        mu += pas*ctrt(gvec, v)
        mu = np.maximum(np.zeros_like(mu), mu)      # projection

        vlist.append(list(v))
        err = np.linalg.norm(v - vold)
        nbit += 1

    return v, np.array(vlist), nbit

#------ Methode d'uzawa optimisee -------#
def uzawa_optimisee(A, b, gvec): 
    val_propres = np.real(np.linalg.eig(A)[0])
    
    pas = 9.8
    # pas = float(str(min(val_propres))[:3])                  # pas pour la maximisation
    # if(pas <= 0 or pas != 9.8):
    #     print("\n -- pas pour la methode d'uzawa:", pas)
    v = np.zeros_like(b)    # v initial
    mu = np.zeros_like(b)   # mu initial
    vold = v.copy()
    vlist = [list(v)]
    err = 1.
    nbit = 0

    inv_A = nplin.inv(A)                                      # pour la minimisation

    while (err > 1e-8 and nbit < 1e6):
        vold[:] = v

        # Minimisations du lagrangien par rapport a v
        v = inv_A @ (b+mu)
        vlist.append(list(v))

        # Maximisation du lagrangien par rapport a mu
        mu += pas*ctrt(gvec, v)
        mu = np.maximum(np.zeros_like(mu), mu)

        err = np.linalg.norm(v - vold)
        nbit += 1

    return v, np.array(vlist), nbit

#------ Test de la methode d'Uzawa et de la methode d'Uzawa optimisee -------#
def test_uzawa(n, type=None):
    #-- mesh
    x = np.linspace(0., 1., n + 2)

    #-- matrice
    dx = 1. / (n + 1)
    A = matrixA(dx, n)
    b = vectorb(n)

    #-- contrainte
    gvec = g(x[1:-1])

    #-- resolution
    start = time.time()
    if type == "opti":
        u , ulist, nbit = uzawa_optimisee(A, b, gvec)
        print("\n====================================\nUZAWA OPTIMISEE")
    else:
        u , ulist, nbit = uzawa(A, b, gvec)
        print("\n====================================\nUZAWA")

    print("temps d'execution: %.4f"%(time.time()-start), "s")

    print("nombre d'iterations:", nbit)
    # print("solution:       ", u)
    # print("solution exacte du probleme sans contraintes:", np.linalg.solve(A, b))

    print("verif contrainte:", np.prod((A @ u - b) * (u != gvec)))                  # doit afficher 0
    print("verif concavite:", np.all(np.round(A@u, 4) >= np.ones_like(u)))          # doit afficher True
    print("verif superiorite a obstacle:", np.all(u - gvec >= -1e-4))               # doit afficher True
    print("====================================")

    #--- plot de la solution et contrainte sur l'intervalle [0, 1]
    ufull = np.zeros_like(x)
    ufull[1:-1] = u
    plt.plot(x, ufull, "g-", label="solution")
    plt.plot(x, g(x), "r--", label="obstacle")
    if type == "opti":
        plt.title("Methode d'Uzawa optimisée")
    else:
        plt.title("Methode d'Uzawa")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = 100

    # test_gradient(n)                    # Methode du gradient a pas fixe (sans contrainte)
    test_gradient_projete(n)            # Methode du gradient projete
    # test_uzawa(n)                       # Methode d'Uzawa ordinaire
    test_uzawa(n, "opti")               # Methode d'Uzawa optimisee
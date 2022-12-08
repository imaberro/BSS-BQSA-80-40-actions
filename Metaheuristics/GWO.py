import numpy as np

def GWO(Problem, DS, poblacion, solutionsRanking, matrixDis, fitness, iter, maxIter, pob, dim,params):
    #GWO
    # guardamos en memoria la mejor solution anterior, para mantenerla
    bestRowAux = solutionsRanking[0]
    BestBinary = matrixDis[bestRowAux]
    BestFitness = np.min(fitness)

    # linear parameter 2->0
    a = 2 - iter * (2/maxIter)

    A1 = 2 * a * np.random.uniform(0,1,size=(pob,dim)) - a; 
    A2 = 2 * a * np.random.uniform(0,1,size=(pob,dim)) - a; 
    A3 = 2 * a * np.random.uniform(0,1,size=(pob,dim)) - a; 

    C1 = 2 *  np.random.uniform(0,1,size=(pob,dim))
    C2 = 2 *  np.random.uniform(0,1,size=(pob,dim))
    C3 = 2 *  np.random.uniform(0,1,size=(pob,dim))

    # eq. 3.6
    Xalfa  = poblacion[solutionsRanking[0]]
    Xbeta  = poblacion[solutionsRanking[1]]
    Xdelta = poblacion[solutionsRanking[2]]

    # eq. 3.5
    Dalfa = np.abs(np.multiply(C1,Xalfa)-poblacion)
    Dbeta = np.abs(np.multiply(C2,Xbeta)-poblacion)
    Ddelta = np.abs(np.multiply(C3,Xdelta)-poblacion)

    # Eq. 3.7
    X1 = Xalfa - np.multiply(A1,Dalfa)
    X2 = Xbeta - np.multiply(A2,Dbeta)
    X3 = Xdelta - np.multiply(A3,Ddelta)

    X = np.divide((X1+X2+X3),3)
    poblacion = X

    return bestRowAux, BestBinary, BestFitness, poblacion
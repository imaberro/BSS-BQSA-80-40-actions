import numpy as np

a = 2

def SCA(Problem, DS, poblacion, solutionsRanking, matrixDis, fitness, iter, maxIter, pob, dim,params):    #SCA
    r1 = a - iter * (a/maxIter)
    r4 = np.random.uniform(low=0.0,high=1.0, size=poblacion.shape[0])
    r2 = (2*np.pi) * np.random.uniform(low=0.0,high=1.0, size=poblacion.shape)
    r3 = np.random.uniform(low=0.0,high=2.0, size=poblacion.shape)
    bestRowAux = solutionsRanking[0]
    Best = poblacion[bestRowAux]
    BestBinary = matrixDis[bestRowAux]
    BestFitness = np.min(fitness)
    poblacion[r4<0.5] = poblacion[r4<0.5] + np.multiply(r1,np.multiply(np.sin(r2[r4<0.5]),np.abs(np.multiply(r3[r4<0.5],Best)-poblacion[r4<0.5])))
    poblacion[r4>=0.5] = poblacion[r4>=0.5] + np.multiply(r1,np.multiply(np.cos(r2[r4>=0.5]),np.abs(np.multiply(r3[r4>=0.5],Best)-poblacion[r4>=0.5])))

    return bestRowAux, BestBinary, BestFitness, poblacion
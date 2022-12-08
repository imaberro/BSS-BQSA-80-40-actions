import numpy as np

b = 1

def WOA(Problem, DS, poblacion, solutionsRanking, matrixDis, fitness, iter, maxIter, pob, dim,params):
    #input = iter, maxIter, pob, dim
    #out = bestRowAux, BestBinary, BestFitness, poblacion
    #WOA
    a = 2 - ((2*iter)/maxIter)
    A = np.random.uniform(low=-a,high=a,size=(pob,dim)) #vector rand de tam (pob,dim)
    Aabs = np.abs(A[0]) # Vector de A absoluto en tam pob
    C = np.random.uniform(low=0,high=2,size=(pob,dim)) #vector rand de tam (pob,dim)
    l = np.random.uniform(low=-1,high=1,size=(pob,dim)) #vector rand de tam (pob,dim)
    p = np.random.uniform(low=0,high=1,size=pob) #vector rand de tam pob ***

    bestRowAux = solutionsRanking[0] #out
    BestBinary = matrixDis[bestRowAux] #out
    BestFitness = np.min(fitness) #out
    Best = poblacion[bestRowAux]

    #ecu 2.1 Pero el movimiento esta en 2.2
    indexCond2_2 = np.intersect1d(np.argwhere(p<0.5),np.argwhere(Aabs<1)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 2.2
    if indexCond2_2.shape[0] != 0:
        poblacion[indexCond2_2] = Best - np.multiply(A[indexCond2_2],np.abs(np.multiply(C[indexCond2_2],Best)-poblacion[indexCond2_2]))

    #ecu 2.8
    indexCond2_8 = np.intersect1d(np.argwhere(p<0.5),np.argwhere(Aabs>=1)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 2.1
    if indexCond2_8.shape[0] != 0:
        Xrand = poblacion[np.random.randint(low=0, high=pob, size=indexCond2_8.shape[0])] #Me entrega un conjunto de soluciones rand de tam indexCond2_2.shape[0] (osea los que cumplen la cond11)

        poblacion[indexCond2_8] = Xrand - np.multiply(A[indexCond2_8],np.abs(np.multiply(C[indexCond2_8],Xrand)-poblacion[indexCond2_8]))

    #ecu 2.5
    indexCond2_5 = np.intersect1d(np.argwhere(p>=0.5),np.argwhere(p>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 2.1
    if indexCond2_5.shape[0] != 0:
        poblacion[indexCond2_5] = np.multiply(np.multiply(np.abs(Best - poblacion[indexCond2_5]),np.exp(b*l[indexCond2_5])),np.cos(2*np.pi*l[indexCond2_5])) + Best
    
    return bestRowAux, BestBinary, BestFitness, poblacion

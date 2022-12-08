# Utils
import numpy as np
import time
from datetime import datetime

# SQL
import json
import Database.Database as Database

# MH
from Problem import RW
from Metrics import Diversidad as dv

connect = Database.Database()

def WOA_RW(id,instance_file,population,maxIter,discretizacionScheme,beta):

    #Inicializamos Problema
    instance = instance_file.split(".")[0]
    Problem = RW.RW(instance,beta)
    
    dim = 6 #Ver si lo dejamos como dato "duro" o un parámetro
    pob = population
    maxIter = maxIter
    DS = discretizacionScheme #[v1,Standard]

    #Variables de diversidad
    diversidades = []
    maxDiversidades = np.zeros(7) #es tamaño 7 porque calculamos 7 diversidades
    PorcentajeExplor = []
    PorcentajeExplot = []
    state = []

    #Generar población inicial
    poblacion = np.random.uniform(low=-1.0, high=1.0, size=(pob,dim))
    matrixDis = Problem.generarPoblacionInicial(pob,dim)
    fitness = np.zeros(pob)
    solutionsRanking = np.zeros(pob)
    matrixDis,fitness,solutionsRanking,BestCostoTotal,BestEmisionTotal,BestVolumenHormigon,BestKilosTotalesAcero  = Problem.evaluarRW(poblacion,matrixDis,solutionsRanking,DS)
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixDis,maxDiversidades)

    #Parámetros fijos de WOA
    b = 1 #Según código

    inicio = datetime.now()
    memory = []

    for iter in range(0, maxIter):
        processTime = time.process_time()  
        timerStart = time.time()
        
        #WOA
        a = 2 - ((2*iter)/maxIter)
        A = np.random.uniform(low=-a,high=a,size=(pob,dim)) #vector rand de tam (pob,dim)
        Aabs = np.abs(A[0]) # Vector de A absoluto en tam pob
        C = np.random.uniform(low=0,high=2,size=(pob,dim)) #vector rand de tam (pob,dim)
        l = np.random.uniform(low=-1,high=1,size=(pob,dim)) #vector rand de tam (pob,dim)
        p = np.random.uniform(low=0,high=1,size=pob) #vector rand de tam pob ***

        bestRowAux = solutionsRanking[0]
        Best = poblacion[bestRowAux]
        BestBinary = matrixDis[bestRowAux]
        BestFitness = np.min(fitness)

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

        #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixDis,fitness,solutionsRanking,BestCostoTotal,BestEmisionTotal,BestVolumenHormigon,BestKilosTotalesAcero  = Problem.evaluarRW(poblacion,matrixDis,solutionsRanking,DS)

        #Conservo el Best
        if fitness[solutionsRanking[0]] >= BestFitness:
            fitness[bestRowAux] = BestFitness
            matrixDis[bestRowAux] = BestBinary
            # print(f'iter: {iter} fitness[bestRowAux]: {fitness[bestRowAux]}')
        # else:
        #     print(f'iter: {iter} fitness[bestRowAux]: {fitness[bestRowAux]}')


        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixDis,maxDiversidades)
        BestFitnes = str(np.min(fitness))

        walltimeEnd = np.round(time.time() - timerStart,6)
        processTimeEnd = np.round(time.process_time()-processTime,6) 

        dataIter = {
            "id_ejecucion": id,
            "numero_iteracion":iter,
            "fitness_mejor": BestFitnes,
            "parametros_iteracion": json.dumps({
                "fitness": BestFitnes,
                "clockTime": walltimeEnd,
                "processTime": processTimeEnd,
                "DS":DS,
                "Diversidades":  str(diversidades),
                "PorcentajeExplor": str(PorcentajeExplor),
                "BestCostoTotal": str(BestCostoTotal),
                "BestEmisionTotal": str(BestEmisionTotal),
                "BestVolumenHormigon": str(BestVolumenHormigon),
                "BestKilosTotalesAcero": str(BestKilosTotalesAcero),
                "Best": str(matrixDis[solutionsRanking[0]])
                #"PorcentajeExplot": str(PorcentajeExplot),
                #"state": str(state)
                })
                }

        memory.append(dataIter)

        if iter % maxIter == 0:
            memory = connect.insertMemory(memory)

    # Si es que queda algo en memoria para insertar
    if(len(memory)>0):
        memory = connect.insertMemory(memory)

    #Actualizamos la tabla resultado_ejecucion, sin mejor_solucion
    memory2 = []
    fin = datetime.now()
    dataResult = {
        "id_ejecucion": id,
        "fitness": BestFitnes,
        "inicio": inicio,
        "fin": fin
        }
    memory2.append(dataResult)
    dataResult = connect.insertMemoryBest(memory2)

    # Update ejecucion
    if not connect.endEjecucion(id,datetime.now(),'terminado'):
        return False

    return True
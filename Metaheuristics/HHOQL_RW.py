# Utils
import numpy as np
import time
from datetime import datetime
import math

# SQL
import json
import Database.Database as Database

# MH
from Problem import RW
from Metrics import Diversidad as dv

# ML
from MachineLearning.QLearning import Q_Learning as QL

connect = Database.Database()

transferFunction = ['V1', 'V2', 'V3', 'V4', 'S1', 'S2', 'S3', 'S4']
DS_actions = transferFunction

def HHOQL_RW(id,instance_file,population,maxIter,discretizacionScheme,beta,ql_alpha,ql_gamma,policy,rewardType,qlAlphaType):

    #Inicializamos Problema
    instance = instance_file.split(".")[0]
    Problem = RW.RW(instance,beta)
    
    dim = 6 #Ver si lo dejamos como dato "duro" o un parámetro
    pob = population
    maxIter = maxIter
    DS = discretizacionScheme #['V4']

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

    # QLEARNING 
    agente = QL(ql_gamma, DS_actions, 2, qlAlphaType, rewardType, maxIter,  qlAlpha = ql_alpha)
    DS = agente.getAccion(0,policy) 

    #Binarizar y calcular fitness
    matrixDis,fitness,solutionsRanking,BestCostoTotal,BestEmisionTotal,BestVolumenHormigon,BestKilosTotalesAcero  = Problem.evaluarRW(poblacion,matrixDis,solutionsRanking,DS_actions[DS])

    #Calcular Diversidad y Estado
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixDis,maxDiversidades)
    
    #QLEARNING
    CurrentState = state[0] #Estamos midiendo según Diversidad "DimensionalHussain"

    #Parámetros de HHO
    beta=1.5 #Escalar según paper
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta) #Escalar
    LB = -10 #Limite inferior de los valores continuos
    UB = 10 #Limite superior de los valores continuos

    inicio = datetime.now()
    memory = []

    for iter in range(0, maxIter):
        processTime = time.process_time()  
        timerStart = time.time()
        
        #HHO
        E0 = np.random.uniform(low=-1.0,high=1.0,size=pob) #vector de tam Pob
        E = 2 * E0 * (1-(iter/maxIter)) #vector de tam Pob
        Eabs = np.abs(E)
        
        q = np.random.uniform(low=0.0,high=1.0,size=pob) #vector de tam Pob
        r = np.random.uniform(low=0.0,high=1.0,size=pob) #vector de tam Pob
        
        Xm = np.mean(poblacion,axis=0)

        bestRowAux = solutionsRanking[0]
        Best = poblacion[bestRowAux]
        BestBinary = matrixDis[bestRowAux]
        BestFitness = np.min(fitness)

        #ecu 1.1
        indexCond1_1 = np.intersect1d(np.argwhere(Eabs>=1),np.argwhere(q>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 1.1
        if indexCond1_1.shape[0] != 0:
            Xrand = poblacion[np.random.randint(low=0, high=pob, size=indexCond1_1.shape[0])] #Me entrega un conjunto de soluciones rand de tam indexCond1_1.shape[0] (osea los que cumplen la cond11)
            poblacion[indexCond1_1] = Xrand - np.multiply(np.random.uniform(low= 0.0, high=1.0, size=indexCond1_1.shape[0]), np.abs(Xrand- (2* np.multiply(np.random.uniform(low= 0.0, high=1.0, size = indexCond1_1.shape[0]),poblacion[indexCond1_1].T).T)).T).T #Aplico la ecu 1.1 solamente a las que cumplen las condiciones np.argwhere(Eabs>=1),np.argwhere(q>=0.5)
        
        #ecu 1.2
        indexCond1_2 = np.intersect1d(np.argwhere(Eabs>=1),np.argwhere(q<0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 1.2 
        if indexCond1_2.shape[0] != 0:
            array_Xm = np.zeros(poblacion[indexCond1_2].shape)
            array_Xm = array_Xm + Xm
            poblacion[indexCond1_2] = ((Best - array_Xm).T - np.multiply( np.random.uniform(low= 0.0, high=1.0, size = indexCond1_2.shape[0]), (LB + np.random.uniform(low= 0.0, high=1.0, size = indexCond1_2.shape[0]) * (UB-LB)) )).T

        #ecu 4
        indexCond4 = np.intersect1d(np.argwhere(Eabs>=0.5),np.argwhere(r>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 4
        if indexCond4.shape[0] != 0:
            array_Xm = np.zeros(poblacion[indexCond4].shape)
            array_Xm = array_Xm + Xm
            poblacion[indexCond4] = ((array_Xm - poblacion[indexCond4]) - np.multiply( E[indexCond4], np.abs(np.multiply( 2*(1-np.random.uniform(low= 0.0, high=1.0, size=indexCond4.shape[0])), array_Xm.T ).T - poblacion[indexCond4]).T).T)

        #ecu 10
        indexCond10 = np.intersect1d(np.argwhere(Eabs>=0.5),np.argwhere(r<0.5))#Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10 
        if indexCond10.shape[0] != 0:
            y10 = poblacion

            #ecu 7
            Array_y10 = np.zeros(poblacion[indexCond10].shape)
            Array_y10 = Array_y10 + y10[bestRowAux]
            y10[indexCond10] = Array_y10- np.multiply( E[indexCond10], np.abs( np.multiply( 2*(1-np.random.uniform(low= 0.0, high=1.0, size=indexCond10.shape[0])), Array_y10.T ).T- Array_y10 ).T ).T  
            
            #ecu 8
            z10 = y10
            S = np.random.uniform(low= 0.0, high=1.0, size=(y10[indexCond10].shape))
            LF = np.divide((0.01 * np.random.uniform(low= 0.0, high=1.0, size=(y10[indexCond10].shape)) * sigma),np.power(np.abs(np.random.uniform(low= 0.0, high=1.0, size=(y10[indexCond10].shape))),(1/beta)))
            z10[indexCond10] = y10[indexCond10] + np.multiply(LF,S)

            #evaluar fitness de ecu 7 y 8
            Fy10 = solutionsRanking
            Fy10[indexCond10] = Problem.evaluarRW(y10[indexCond10],matrixDis[indexCond10],solutionsRanking[indexCond10],DS,)[1]
            
            Fz10 = solutionsRanking
            Fz10[indexCond10] = Problem.evaluarRW(z10[indexCond10],matrixDis[indexCond10],solutionsRanking[indexCond10],DS)[1]
            
            #ecu 10.1
            indexCond101 = np.intersect1d(indexCond10, np.argwhere(Fy10 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10.1
            if indexCond101.shape[0] != 0:
                poblacion[indexCond101] = y10[indexCond101]

            #ecu 10.2
            indexCond102 = np.intersect1d(indexCond10, np.argwhere(Fz10 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 10.2
            if indexCond102.shape[0] != 0:
                poblacion[indexCond102] = z10[indexCond102]

            # ecu 6
            indexCond6 = np.intersect1d(np.argwhere(Eabs<0.5),np.argwhere(r>=0.5)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 6
            if indexCond6.shape[0] != 0:
                poblacion[indexCond6] = Best - np.multiply(E[indexCond6], np.abs(Best - poblacion[indexCond6] ).T ).T

            #ecu 11
            indexCond11 = np.intersect1d(np.argwhere(Eabs<0.5),np.argwhere(r<0.5))#Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11
            if indexCond11.shape[0] != 0:
                #ecu 12
                y11 = poblacion
                array_Xm = np.zeros(poblacion[indexCond11].shape)
                array_Xm = array_Xm + Xm
                y11[indexCond11] = y11[bestRowAux]-  np.multiply(E[indexCond11],  np.abs(  np.multiply(  2*(1-np.random.uniform(low= 0.0, high=1.0, size=poblacion[indexCond11].shape)),  y11[bestRowAux]  )- array_Xm ).T ).T

                #ecu 13
                z11 = y11
                S = np.random.uniform(low= 0.0, high=1.0, size=(y11.shape))
                LF = np.divide((0.01 * np.random.uniform(low= 0.0, high=1.0, size=(y11.shape)) * sigma),np.power(np.abs(np.random.uniform(low= 0.0, high=1.0, size=(y11.shape))),(1/beta)))
                z11[indexCond11] = y11[indexCond11] + np.multiply(S[indexCond11],LF[[indexCond11]])

                #evaluar fitness de ecu 12 y 13
                if solutionsRanking is None: solutionsRanking = np.ones(pob)*999999
                Fy11 = solutionsRanking
                Fy11[indexCond11] = Problem.evaluarRW(y11[indexCond11],matrixDis[indexCond11],solutionsRanking[indexCond11],DS)[1]
                
                Fz11 = solutionsRanking
                Fz11[indexCond11] = Problem.evaluarRW(z11[indexCond11],matrixDis[indexCond11],solutionsRanking[indexCond11],DS)[1]
                
                #ecu 11.1
                indexCond111 = np.intersect1d(indexCond11, np.argwhere(Fy11 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11.1
                if indexCond111.shape[0] != 0:
                    poblacion[indexCond111] = y11[indexCond111]

                #ecu 11.2
                indexCond112 = np.intersect1d(indexCond11, np.argwhere(Fz11 < solutionsRanking)) #Nos entrega los index de las soluciones a las que debemos aplicar la ecu 11.2
                if indexCond112.shape[0] != 0:
                    poblacion[indexCond112] = z11[indexCond112]

        # Escogemos esquema desde QL
        DS = agente.getAccion(CurrentState,policy)
        oldState = CurrentState #Rescatamos estado actual

        #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixDis,fitness,solutionsRanking,BestCostoTotal,BestEmisionTotal,BestVolumenHormigon,BestKilosTotalesAcero  = Problem.evaluarRW(poblacion,matrixDis,solutionsRanking,DS_actions[DS])

        #Conservo el Best
        if fitness[solutionsRanking[0]] >= BestFitness:
            fitness[bestRowAux] = BestFitness
            matrixDis[bestRowAux] = BestBinary
            # print(f'iter: {iter} fitness[bestRowAux]: {fitness[bestRowAux]}')
        # else:
        #     print(f'iter: {iter} fitness[bestRowAux]: {fitness[bestRowAux]}')

        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixDis,maxDiversidades)
        BestFitnes = str(np.min(fitness))

        CurrentState = state[0]
        # Observamos, y recompensa/castigo.  Actualizamos Tabla Q
        agente.updateQtable(np.min(fitness), DS, CurrentState, oldState, iter)

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
                "DS":str(DS),
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
    qtable = agente.getQtable()
    dataResult = {
        "id_ejecucion": id,
        "fitness": BestFitnes,
        "inicio": inicio,
        "fin": fin,
        "mejor_solucion": json.dumps(qtable.tolist())
        }
    memory2.append(dataResult)
    dataResult = connect.insertMemoryBest(memory2)

    # Update ejecucion
    if not connect.endEjecucion(id,datetime.now(),'terminado'):
        return False

    return True
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

transferFunction = ['V1', 'V2', 'V3', 'V4', 'S1', 'S2', 'S3', 'S4']
DS_actions = transferFunction

def MHML_RW(id,instance_file,pob,maxIter,discretizacionScheme,beta_Dis,ql_alpha,ql_gamma,policy,rewardType,qlAlphaType,FO,MH,ML,paramsML):
    # ML
    if ML == "QL":
        from MachineLearning.QLearning import Q_Learning as ML
    if ML == "SA":
        from MachineLearning.SARSA import SARSA as ML

    # MH
    if MH == "HHO":
        from Metaheuristics.HHO import HHO as metaheuristic
    if MH == "GWO":
        from Metaheuristics.GWO import GWO as metaheuristic
    if MH == "SCA":
        from Metaheuristics.SCA import SCA as metaheuristic
    if MH == "WOA":
        from Metaheuristics.WOA import WOA as metaheuristic

    #Inicializamos Problema
    instance = instance_file.split(".")[0]
    Problem = RW.RW(instance,beta_Dis)
    
    dim = 6 #Ver si lo dejamos como dato "duro" o un parámetro
    maxIter = maxIter
    DS = discretizacionScheme #['V4']
    params = {}

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
    agente = ML(ql_gamma, DS_actions, 2, qlAlphaType, rewardType, maxIter,paramsML,  qlAlpha = ql_alpha)
    DS = agente.getAccion(0,policy) 
    
    params['TF'] = DS_actions[DS]
    params['FO'] = FO

    #Binarizar y calcular fitness
    matrixDis,fitness,solutionsRanking,BestCostoTotal,BestEmisionTotal,BestVolumenHormigon,BestKilosTotalesAcero  = Problem.obtenerFitness(poblacion,matrixDis,solutionsRanking,params)

    #Calcular Diversidad y Estado
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixDis,maxDiversidades)
    
    #QLEARNING
    CurrentState = state[0] #Estamos midiendo según Diversidad "DimensionalHussain"

    inicio = datetime.now()
    memory = []

    for iter in range(0, maxIter):
        processTime = time.process_time()  
        timerStart = time.time()

        #MH
        bestRowAux, BestBinary, BestFitness, poblacion = metaheuristic(Problem, DS, poblacion, solutionsRanking, matrixDis, fitness, iter, maxIter, pob, dim, params)

        # Escogemos esquema desde QL
        DS = agente.getAccion(CurrentState,policy)
        params['TF'] = DS_actions[DS]
        oldState = CurrentState #Rescatamos estado actual

        #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixDis,fitness,solutionsRanking,BestCostoTotal,BestEmisionTotal,BestVolumenHormigon,BestKilosTotalesAcero  = Problem.obtenerFitness(poblacion,matrixDis,solutionsRanking,params)

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
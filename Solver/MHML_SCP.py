# Utils
import os
# from envs import env
import numpy as np
import time
from datetime import datetime

# SQL
import json
import Database.Database as Database

# MH
from Problem.util import read_instance as Instance
from Problem import SCP
from Metrics import Diversidad as dv

# Definicion Environments Vars
workdir = os.path.abspath(os.getcwd())
workdirInstance = workdir+ '/Instances/'

connect = Database.Database()

def MHML_SCP(id,instance_file,instance_dir,pob,maxIter,discretizacionScheme,ql_alpha,ql_gamma,repair,policy,rewardType,qlAlphaType,MH,ML,paramsML):
    # ML
    if ML == "QL":
        from MachineLearning.QLearning import Q_Learning as ML
    if ML == "SA":
        from MachineLearning.SARSA import SARSA as ML
    if ML == "BQSA":
        from MachineLearning.BQSA import BQSA as ML

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
    Problem = SCP.SCP(workdirInstance, instance_dir, instance_file)
    instance_path = Problem.obtenerInstancia()

    if not os.path.exists(instance_path):
        print(f'No se encontró la instancia: {instance_path}')
        return False

    instance = Instance.Read(instance_path)    
    
    matrizCobertura = np.array(instance.get_r())
    vectorCostos = np.array(instance.get_c())

    dim = len(vectorCostos)
    maxIter = maxIter
    action = discretizacionScheme #['V4']
    params = {}

    #Variables de diversidad
    diversidades = []
    maxDiversidades = np.zeros(7) #es tamaño 7 porque calculamos 7 diversidades
    PorcentajeExplor = []
    PorcentajeExplot = []
    state = []

    #Generar población inicial
    poblacion = np.random.uniform(low=-10.0, high=10.0, size=(pob,dim))
    matrixBin = np.random.randint(low=0, high=2, size = (pob,dim))
    fitness = np.zeros(pob)
    solutionsRanking = np.zeros(pob)

    # QLEARNING 
    agente = ML(ql_gamma, 2, qlAlphaType, rewardType, maxIter,paramsML,  qlAlpha = ql_alpha)
    action = agente.getAccion(0,policy) 
    DS_actions = paramsML['DS_actions']

    params["costos"] = vectorCostos
    params["cobertura"] = matrizCobertura
    params["ds"] = DS_actions[action]
    params["repairType"] = repair

    #Binarizar y calcular fitness
    matrixBin,fitness,solutionsRanking,numReparaciones  = Problem.obtenerFitness(poblacion,matrixBin,solutionsRanking,params)
    
    #Calcular Diversidad y Estado
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)

    # #QLEARNING
    # state = state[0] #Estamos midiendo según Diversidad "DimensionalHussain"

    inicio = datetime.now()
    memory = []
    # action = np.random.choice(DS_actions)

    for iter in range(0, maxIter):
        processTime = time.process_time()  
        timerStart = time.time()

        #MH
        bestRowAux, BestBinary, BestFitness, poblacion = metaheuristic(Problem, action, poblacion, solutionsRanking, matrixBin, fitness, iter, maxIter, pob, dim, params)

        #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixBin,fitness,solutionsRanking,numReparaciones = Problem.obtenerFitness(poblacion,matrixBin,solutionsRanking,params)

        #Conservo el Best
        if fitness[bestRowAux] > BestFitness:
            fitness[bestRowAux] = BestFitness
            matrixBin[bestRowAux] = BestBinary

        #Calcular Diversidad y Estado
        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)
        BestFitnes = str(np.min(fitness)) # para JSON
        
        # Escogemos esquema desde QL
        newState = state[0]
        newAction = agente.getAccion(newState,policy)
        
        # Observamos, y recompensa/castigo.  Actualizamos Tabla Q
        agente.updateQtable(np.min(fitness), action, newAction, state[0], newState, iter)

        # Actualizamos action y state para la proxima iteración
        action = newAction    
        state = newState 
        params["ds"] = DS_actions[action]

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
                "DS":str(DS_actions[action]),
                "Diversidades":  str(diversidades),
                "PorcentajeExplor": str(PorcentajeExplor),
                "numReparaciones": str(numReparaciones)
                #"PorcentajeExplot": str(PorcentajeExplot),
                #"state": str(state)
                })
                }

        memory.append(dataIter)

        if iter % 50 == 0 and iter != 0:
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

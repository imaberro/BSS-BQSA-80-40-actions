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

def MH_SCP(id,instance_file,instance_dir,pob,maxIter,discretizacionScheme,repair,FO,MH):
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
    DS = discretizacionScheme #[v1,Standard]
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

    params["costos"] = vectorCostos
    params["cobertura"] = matrizCobertura
    params["ds"] = DS
    params["repairType"] = repair

    matrixBin,fitness,solutionsRanking,numReparaciones  = Problem.obtenerFitness(poblacion,matrixBin,solutionsRanking,params)
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)

    inicio = datetime.now()
    memory = []

    for iter in range(0, maxIter):
        processTime = time.process_time()  
        timerStart = time.time()
        
        #MH
        bestRowAux, BestBinary, BestFitness, poblacion = metaheuristic(Problem, DS, poblacion, solutionsRanking, matrixBin, fitness, iter, maxIter, pob, dim,params)
        
        #Binarizar y calcular fitness
        matrixBin,fitness,solutionsRanking,numReparaciones  = Problem.obtenerFitness(poblacion,matrixBin,solutionsRanking,params)

        #Conservo el Best
        if fitness[solutionsRanking[0]] >= BestFitness:
            fitness[bestRowAux] = BestFitness
            matrixBin[bestRowAux] = BestBinary
            # print(f'iter: {iter} fitness[bestRowAux]: {fitness[bestRowAux]}')
        # else:
        #     print(f'iter: {iter} fitness[bestRowAux]: {fitness[bestRowAux]}')

        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)
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

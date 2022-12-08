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

def MH_RW(id,instance_file,population,maxIter,discretizacionScheme,beta_Dis,FO,MH):
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
    pob = population
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
    poblacion = np.random.uniform(low=-1.0, high=1.0, size=(pob,dim))
    matrixDis = Problem.generarPoblacionInicial(pob,dim)
    fitness = np.zeros(pob)
    solutionsRanking = np.zeros(pob)
    params['TF'] = DS
    params['FO'] = FO
    matrixDis,fitness,solutionsRanking,BestCostoTotal,BestEmisionTotal,BestVolumenHormigon,BestKilosTotalesAcero  = Problem.obtenerFitness(poblacion,matrixDis,solutionsRanking,params)
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixDis,maxDiversidades)

    inicio = datetime.now()
    memory = []
    

    for iter in range(0, maxIter):
        processTime = time.process_time()  
        timerStart = time.time()

        #MH
        bestRowAux, BestBinary, BestFitness, poblacion = metaheuristic(Problem, DS, poblacion, solutionsRanking, matrixDis, fitness, iter, maxIter, pob, dim,params)

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
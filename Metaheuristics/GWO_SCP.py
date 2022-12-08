# Utils
import os
from envs import env
import numpy as np
import time
from datetime import datetime

# SQL
import json

import Database.Database as Database
# MH

from Problem.util import read_instance as Instance
from Problem import SCP as Problem
from Metrics import Diversidad as dv


# Definicion Environments Vars
workdir = os.path.abspath(os.getcwd())
workdirInstance = workdir+env('DIR_INSTANCES')

connect = Database.Database()

def GWO_SCP(id,instance_file,instance_dir,population,maxIter,discretizacionScheme,repair):

    instance_path = workdirInstance + instance_dir + instance_file

    if not os.path.exists(instance_path):
        print(f'No se encontró la instancia: {instance_path}')
        return False

    instance = Instance.Read(instance_path)
    
    matrizCobertura = np.array(instance.get_r())
    vectorCostos = np.array(instance.get_c())

    dim = len(vectorCostos)
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
    matrixBin = np.random.randint(low=0, high=2, size = (pob,dim))
    fitness = np.zeros(pob)
    solutionsRanking = np.zeros(pob)
    matrixBin,fitness,solutionsRanking,numReparaciones  = Problem.SCP(poblacion,matrixBin,solutionsRanking,vectorCostos,matrizCobertura,DS,repair)
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)

    inicio = datetime.now()
    memory = []

    for iter in range(0, maxIter):
        processTime = time.process_time()  
        timerStart = time.time()
        
        #GWO
        # guardamos en memoria la mejor solution anterior, para mantenerla
        bestRowAux = solutionsRanking[0]
        BestBinary = matrixBin[bestRowAux]
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
        #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixBin,fitness,solutionsRanking,numReparaciones = Problem.SCP(poblacion,matrixBin,solutionsRanking,vectorCostos,matrizCobertura,DS,repair)

        #Conservo el Best
        if fitness[bestRowAux] > BestFitness:
            fitness[bestRowAux] = BestFitness
            matrixBin[bestRowAux] = BestBinary

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
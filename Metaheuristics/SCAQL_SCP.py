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

# ML
from MachineLearning.QLearning import Q_Learning as QL

# Definicion Environments Vars
workdir = os.path.abspath(os.getcwd())
workdirInstance = workdir+env('DIR_INSTANCES')

connect = Database.Database()


transferFunction = ['V1', 'V2', 'V3', 'V4', 'S1', 'S2', 'S3', 'S4']
operatorBinarization = ['Standard','Complement','Elitist','Static','ElitistRoulette']

DS_actions = [tf + "," + ob for tf in transferFunction for ob in operatorBinarization]

def SCAQL_SCP(id,instance_file,instance_dir,population,maxIter,discretizacionScheme,ql_alpha,ql_gamma,repair,policy,rewardType,qlAlphaType):

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

    # QLEARNING 
    agente = QL(ql_gamma, DS_actions, 2, qlAlphaType, rewardType, maxIter,  qlAlpha = ql_alpha)
    DS = agente.getAccion(0,policy) 

    #Binarizar y calcular fitness
    matrixBin,fitness,solutionsRanking,numReparaciones  = Problem.SCP(poblacion,matrixBin,solutionsRanking,vectorCostos,matrizCobertura,DS_actions[DS],repair)
    
    #Calcular Diversidad y Estado
    diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)

    #QLEARNING
    CurrentState = state[0] #Estamos midiendo según Diversidad "DimensionalHussain"

    #Parámetro de SCA
    a = 2

    inicio = datetime.now()
    memory = []

    for iter in range(0, maxIter):
        processTime = time.process_time()  
        timerStart = time.time()
        
        #SCA
        r1 = a - iter * (a/maxIter)
        r4 = np.random.uniform(low=0.0,high=1.0, size=poblacion.shape[0])
        r2 = (2*np.pi) * np.random.uniform(low=0.0,high=1.0, size=poblacion.shape)
        r3 = np.random.uniform(low=0.0,high=2.0, size=poblacion.shape)
        bestRowAux = solutionsRanking[0]
        Best = poblacion[bestRowAux]
        BestBinary = matrixBin[bestRowAux]
        BestFitness = np.min(fitness)
        poblacion[r4<0.5] = poblacion[r4<0.5] + np.multiply(r1,np.multiply(np.sin(r2[r4<0.5]),np.abs(np.multiply(r3[r4<0.5],Best)-poblacion[r4<0.5])))
        poblacion[r4>=0.5] = poblacion[r4>=0.5] + np.multiply(r1,np.multiply(np.cos(r2[r4>=0.5]),np.abs(np.multiply(r3[r4>=0.5],Best)-poblacion[r4>=0.5])))
    
        # Escogemos esquema desde QL
        DS = agente.getAccion(CurrentState,policy)
        oldState = CurrentState #Rescatamos estado actual

        #Binarizamos y evaluamos el fitness de todas las soluciones de la iteración t
        matrixBin,fitness,solutionsRanking,numReparaciones = Problem.SCP(poblacion,matrixBin,solutionsRanking,vectorCostos,matrizCobertura,DS_actions[DS],repair)

        #Conservo el Best
        if fitness[bestRowAux] > BestFitness:
            fitness[bestRowAux] = BestFitness
            matrixBin[bestRowAux] = BestBinary

        diversidades, maxDiversidades, PorcentajeExplor, PorcentajeExplot, state = dv.ObtenerDiversidadYEstado(matrixBin,maxDiversidades)
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
                "numReparaciones": str(numReparaciones),
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

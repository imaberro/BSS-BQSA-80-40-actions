import numpy as np

#Gracias Mauricio Y Lemus!
from .repair import ReparaStrategy as repara
from Discretization import DiscretizationScheme as DS

#action : esquema de discretizacion DS
class SCP:
    def __init__(self,workdirInstance, instance_dir, instance_file):
        self.workdirInstance = workdirInstance
        self.instance_dir = instance_dir
        self.instance_file = instance_file
        
    def obtenerInstancia(self):
        return self.workdirInstance + self.instance_dir + self.instance_file

    def obtenerFitness(self,poblacion,matrix,solutionsRanking,params):
        
        costos = params["costos"]
        cobertura = params["cobertura"]
        ds = params["ds"]
        repairType = params["repairType"]

        ds = ds.split(",")
        ds = DS.DiscretizationScheme(poblacion,matrix,solutionsRanking,ds[0],ds[1])
        matrix = ds.binariza()

        repair = repara.ReparaStrategy(cobertura,costos,cobertura.shape[0],cobertura.shape[1])
        matrizSinReparar = matrix
        for solucion in range(matrix.shape[0]):
            if repair.cumple(matrix[solucion]) == 0:
                matrix[solucion] = repair.repara_one(matrix[solucion],repairType)[0]
        matrizReparada = matrix
        numReparaciones = np.sum(np.abs(matrizReparada - matrizSinReparar))

        #Calculamos Fitness
        fitness = np.sum(np.multiply(matrix,costos),axis =1)
        solutionsRanking = np.argsort(fitness) # rankings de los mejores fitness

        return matrix,fitness,solutionsRanking,numReparaciones



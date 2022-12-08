#  Author: Diego Tapia R.
#  E-mail: root.chile@gmail.com - diego.tapia.r@mail.pucv.cl
import numpy as np
import math
class BQSA():

    # Initialize alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, gamma, stateQ, qlAlphaType, rewardType, iterMax, paramsML, epsilon = 0.05, qlAlpha =0.1):

        self.actions = paramsML['DS_actions']
        self.gamma = gamma
        self.qlAlphaType = qlAlphaType
        self.rewardType = rewardType
        self.iterMax = iterMax
        self.iter = 0
        self.cond_backward = int(paramsML['cond_backward'])
        self.MinMax = paramsML['MinMax']
        self.contador_backward = 0
        
        self.epsilon = epsilon
        self.qlAlpha = qlAlpha
        self.bestMetric = 999999999 #Esto en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
        self.W = 10
        self.M = [] #Memoria [S_t, a_t,r_t+1, S_t+1]
        
        self.Qvalues = np.zeros(shape=(stateQ,len(self.actions))) #state,actions
        self.visitas = np.zeros(shape=(stateQ,len(self.actions))) #state,actions
        #self.Qvalues[0] = np.zeros(len(self.actions))

    def getReward(self,metric):
        
        if self.MinMax == "min":
            if self.rewardType == "withPenalty1": 
                if self.bestMetric > metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                    self.bestMetric = metric
                    return 1
                return -1

            elif self.rewardType == "withoutPenalty1":
                if self.bestMetric > metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                    self.bestMetric = metric
                    return 1
                return 0

            elif self.rewardType == "globalBest":
                if self.bestMetric > metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                    self.bestMetric = metric
                    return self.W/self.bestMetric
                return 0

            elif self.rewardType == "rootAdaptation":
                if self.bestMetric > metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                    self.bestMetric = metric
                    return math.sqrt(metric)
                return 0

            elif self.rewardType == "escalatingMultiplicativeAdaptation":
                if self.bestMetric > metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                    self.bestMetric = metric
                    return self.W*self.bestMetric
                return 0

        if self.MinMax == "max": 
            if self.rewardType == "withPenalty1": 
                if self.bestMetric < metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                    self.bestMetric = metric
                    return 1
                return -1

            elif self.rewardType == "withoutPenalty1":
                if self.bestMetric < metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                    self.bestMetric = metric
                    return 1
                return 0

            elif self.rewardType == "globalBest":
                if self.bestMetric < metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                    self.bestMetric = metric
                    return self.W/self.bestMetric
                return 0

            elif self.rewardType == "rootAdaptation":
                if self.bestMetric < metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                    self.bestMetric = metric
                    return math.sqrt(metric)
                return 0

            elif self.rewardType == "escalatingMultiplicativeAdaptation":
                if self.bestMetric < metric:#Esta condición en el framework debe ser consultada según el problema, ya que puede min o max la función objetivo
                    self.bestMetric = metric
                    return self.W*self.bestMetric
                return 0

    def getAccion(self,state,policy):
        
        # e-greedy
        if policy == "e-greedy":
            probabilidad = np.random.uniform(low=0.0, high=1.0) #numero aleatorio [0,1]
            if probabilidad <= self.epsilon: #seleccion aleatorio
                accionRandom = np.random.randint(low=0, high=self.Qvalues.shape[1])
                return accionRandom #seleccion aleatoria de una accion     
            else: #selecion de Q_Value mayor      
                maximo = np.amax(self.Qvalues[state]) # retorna el elemento mayor por fila    
                indices = np.where(self.Qvalues[state,:] == maximo)[0]  #retorna los indices donde se ubica el maximo en la fila estado  
                return np.random.choice(indices) # funciona tanto cuando hay varios iguales como cuando hay solo uno
        # greedy
        elif policy == "greedy":
            return np.argmax(self.Qvalues[state])

        # e-soft 
        elif policy == "e-soft":
            probabilidad = np.random.uniform(low=0.0, high=1.0) #numero aleatorio [0,1]
            if probabilidad > self.epsilon: #seleccion aleatorio
                return np.random.randint(low=0, high=self.Qvalues.shape[0]) #seleccion aleatoria de una accion     
            else: #selecion de Q_Value mayor        
                maximo = np.amax(self.Qvalues,axis=1) # retorna el elemento mayor por fila        
                indices = np.where(self.Qvalues[state,:] == maximo[state])[0]  #retorna los indices donde se ubica el maximo en la fila estado        
                return np.random.choice(indices) # funciona tanto cuando hay varios iguales como cuando hay solo uno 

        # softMax seleccion ruleta
        elif policy == "softMax-rulette":
            #*** Falta generar una normalización de las probabilidades que sumen 1, para realizar el choice
            QtablePositiva = (self.Qvalues[state]+np.abs(np.min(self.Qvalues[state])))/np.max(self.Qvalues[state]+np.abs(np.min(self.Qvalues[state])))
            Qtable_normalizada = QtablePositiva/np.sum(QtablePositiva) #La suma de las prob deben ser 1
            seleccionado = np.random.choice(self.Qvalues[state],p=Qtable_normalizada)
            indices = np.where(self.Qvalues[state,:] == seleccionado)[0]
            return np.random.choice(indices)
    
        # softmax seleccion ruleta elitista (25% mejores acciones)
        elif policy == "softMax-rulette-elitist":
            ordenInvertido = np.multiply(self.Qvalues[state],-1)
            sort = np.argsort(ordenInvertido) # argumentos ordenados
            cant_mejores = int(sort.shape[0]*0.25) # obtenemos el 25% de los mejores argumentos
            rulette_elitist = sort[0:cant_mejores] # tiene el 25% de los mejores argumentos
            return np.random.choice(rulette_elitist)
    

    def actualizar_Visitas(self,action,state): # ACTUALIZACION DE LAS VISITAS
        self.visitas[state,action] = self.visitas[state,action] + 1


    def getAlpha(self,state,action,iter):

        if self.qlAlphaType == "static": 
            #alpha estatico 
            return  self.qlAlpha

        elif self.qlAlphaType == "iteration":
            return 1 - (0.9*(iter/self.iterMax))
            
        elif self.qlAlphaType == "visits":
            return (1/(1 + self.visitas[state,action]))

    def updateQtable(self,metric,action,newAction,state,newState,iter): 
        Reward = self.getReward(metric)
        
        alpha = self.getAlpha(state,action,iter)

        # Qnuevo = ( (1 - alpha) * self.Qvalues[state][action]) + alpha * (Reward + (self.gamma  * max(self.Qvalues[state])))
        Qnuevo = self.Qvalues[state][action] + alpha * (Reward + (self.gamma  * max(self.Qvalues[state])) - self.Qvalues[state][action])

        self.actualizar_Visitas(action,state) #Actuzación de visitas

        self.Qvalues[state][action] = Qnuevo
        
        self.M.append([state,action,Reward,state])

        if (iter % self.cond_backward) == 0 :
            for m in reversed(self.M):
                state = m[0]
                action = m[1]
                Reward = m[2]
                state = m[3]

                Qnuevo = self.Qvalues[state][action] + alpha * (Reward + (self.gamma  * max(self.Qvalues[state])) - self.Qvalues[state][action])
                self.actualizar_Visitas(action,state) #Actuzación de visitas
                self.Qvalues[state][action] = Qnuevo
            self.M = []


    def getQtable(self):
        return self.Qvalues

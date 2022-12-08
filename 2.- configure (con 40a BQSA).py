# Utils
# from envs import env
import numpy as np
import configparser

# SQL
import sqlalchemy as db
import json


#Credenciales
config = configparser.ConfigParser()
config.read('db_config.ini')
host = config['postgres']['host']
db_name = config['postgres']['db_name']
port = config['postgres']['port']
user = config['postgres']['user']
pwd = config['postgres']['pass']


# Conexión a la DB de resultados

engine = db.create_engine(f'postgresql://{user}:{pwd}@{host}:{port}/{db_name}')
metadata = db.MetaData()


try: 
    connection = engine.connect()

except db.exc.SQLAlchemyError as e:
    exit(str(e.__dict__['orig']))



datosEjecucion = db.Table('datos_ejecucion', metadata, autoload=True, autoload_with=engine)
insertDatosEjecucion = datosEjecucion.insert().returning(datosEjecucion.c.id)


algorithms = [
'GWO_SCP_40aBQSA1','WOA_SCP_40aBQSA1','SCA_SCP_40aBQSA1',
'GWO_SCP_40aQL1','WOA_SCP_40aQL1','SCA_SCP_40aQL1',
'GWO_SCP_40aSA1','WOA_SCP_40aSA1','SCA_SCP_40aSA1'
]


instances = [
'mscp41','mscp42','mscp43','mscp44','mscp45','mscp46','mscp47','mscp48','mscp49','mscp410',
'mscp51','mscp52','mscp53','mscp54','mscp55','mscp56','mscp57','mscp58','mscp59','mscp510',
'mscp61','mscp62','mscp63','mscp64','mscp65',
'mscpa1','mscpa2','mscpa3','mscpa4','mscpa5',
'mscpb1','mscpb2','mscpb3','mscpb4','mscpb5',
'mscpc1','mscpc2','mscpc3','mscpc4','mscpc5',
'mscpd1','mscpd2','mscpd3','mscpd4','mscpd5'
]
runs = 31
population  = 40
maxIter = 1000
beta_Dis = 0.8 #Parámetro de la discretización de RW
ql_alpha = 0.1
ql_gamma =  0.4
policy = "softMax-rulette-elitist" #puede ser 'e-greedy', 'greedy', 'e-soft', 'softMax-rulette', 'softMax-rulette-elitist'
qlAlphaType = "static" # Puede ser 'static', 'iteration', 'visits'
repair = 2 # 1:Simple; 2:Compleja
instance_dir = "MSCP/"
DS_actions = ['V1,Standard', 'V1,Complement', 'V1,Elitist', 'V1,Static', 'V1,ElitistRoulette', 
              'V2,Standard', 'V2,Complement', 'V2,Elitist', 'V2,Static', 'V2,ElitistRoulette', 
              'V3,Standard', 'V3,Complement', 'V3,Elitist', 'V3,Static', 'V3,ElitistRoulette', 
              'V4,Standard', 'V4,Complement', 'V4,Elitist', 'V4,Static', 'V4,ElitistRoulette', 
              'S1,Standard', 'S1,Complement', 'S1,Elitist', 'S1,Static', 'S1,ElitistRoulette', 
              'S2,Standard', 'S2,Complement', 'S2,Elitist', 'S2,Static', 'S2,ElitistRoulette', 
              'S3,Standard', 'S3,Complement', 'S3,Elitist', 'S3,Static', 'S3,ElitistRoulette', 
              'S4,Standard', 'S4,Complement', 'S4,Elitist', 'S4,Static', 'S4,ElitistRoulette'
              ]
paramsML = {'cond_backward': '10', 'MinMax': 'min', 'DS_actions': DS_actions} 


for run in range(runs):
    for instance in instances:
        for algorithm in algorithms:
            FO = algorithm.split("_")[1].replace("RW","")
            MH = algorithm.split("_")[0]
            ML = algorithm.split("_")[2][:-1] #QL p SA p BQSA
            numRewardType = int(algorithm.split("_")[2][-1]) # 1 2 3 4 5
            if ML == "40aQL" or ML == "40aSA" or ML == "40aBQSA":
                discretizationScheme = DS_actions[np.random.randint(low=0, high=len(DS_actions))]
                if numRewardType == 1:
                    rewardType = "withPenalty1"
                if numRewardType == 2:
                    rewardType = "withoutPenalty1"
                if numRewardType == 3:
                    rewardType = "globalBest"
                if numRewardType == 4:
                    rewardType = "rootAdaptation"
                if numRewardType == 5:
                    rewardType = "escalatingMultiplicativeAdaptation"

            if algorithm.split("_")[2] == "BCL":
                discretizationScheme = 'V4,Elitist'
            if algorithm.split("_")[2] == "MIR":
                discretizationScheme = 'V4,Complement'

            data = {
                'nombre_algoritmo' : algorithm,
                'parametros': json.dumps({
                    'instance_name' : instance,
                    'instance_file': instance+'.txt',
                    'instance_dir': instance_dir,
                    'population': population,
                    'maxIter':maxIter,
                    'discretizationScheme':discretizationScheme,
                    'ql_alpha': ql_alpha,
                    'ql_gamma': ql_gamma,
                    'repair': repair,
                    'policy': policy,
                    'rewardType': rewardType,
                    'qlAlphaType': qlAlphaType,
                    'beta_Dis': beta_Dis,
                    'FO': FO,
                    'MH': MH,
                    'ML': ML,
                    'paramsML': paramsML
            }),
                'estado' : 'pendiente'
            }
            result = connection.execute(insertDatosEjecucion,data)
            idEjecucion = result.fetchone()[0]
            print(f'Poblado ID #:{idEjecucion}')

print("Todo poblado")
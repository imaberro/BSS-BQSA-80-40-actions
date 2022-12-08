
# Utils

import sys
import os
import settings
# from envs import env
import numpy as np
import time
from datetime import datetime
from pathlib import Path

# SQL
import sqlalchemy as db
import psycopg2
import json
import pickle
import zlib

import Database.Database as Database


### Buscamos experimentos pendientes
connect = Database.Database()


# Algorithms

# from Metaheuristics.GWO_SCP import GWO_SCP
# from Metaheuristics.GWOQL_SCP import GWOQL_SCP
# from Metaheuristics.SCA_SCP import SCA_SCP
# from Metaheuristics.SCAQL_SCP import SCAQL_SCP
# from Metaheuristics.HHO_SCP import HHO_SCP
# from Metaheuristics.HHOQL_SCP import HHOQL_SCP
# from Metaheuristics.WOA_SCP import WOA_SCP
# from Metaheuristics.WOAQL_SCP import WOAQL_SCP
# from Metaheuristics.WOA_RW import WOA_RW
# from Metaheuristics.WOAQL_RW import WOAQL_RW
# from Metaheuristics.SCA_RW import SCA_RW
# from Metaheuristics.SCAQL_RW import SCAQL_RW
# from Metaheuristics.GWO_RW import GWO_RW
# from Metaheuristics.GWOQL_RW import GWOQL_RW
# from Metaheuristics.HHO_RW import HHO_RW
# from Metaheuristics.HHOQL_RW import HHOQL_RW
from Solver.MH_RW import MH_RW
from Solver.MHML_RW import MHML_RW
from Solver.MH_SCP import MH_SCP
from Solver.MHML_SCP import MHML_SCP



flag = True
while flag:

    id, algorithm, params = connect.getLastPendingAlgorithm('pendiente')
    
    if id == 0:
        print('No hay más ejecuciones pendientes')
        break
        
       
    print("------------------------------------------------------------------------------------------------------------------\n")
    print(f'Id Execution: {id} -  {algorithm}')
    print(json.dumps(params,indent=4))
    print("------------------------------------------------------------------------------------------------------------------\n")

    if (algorithm.split("_")[1] == 'SCP') and algorithm.split("_")[2][:2] != 'QL' and algorithm.split("_")[2][:2] != 'SA' and algorithm.split("_")[2][:4] != 'BQSA':
        if  MH_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme'],
                params['repair'],
                params['FO'],
                params['MH']
                ) == True:
            print(f'Ejecución {id} completada ')
  
    if (algorithm.split("_")[1] == 'SCP') and (algorithm.split("_")[2][:2] == 'QL' or algorithm.split("_")[2][:2] == 'SA' or algorithm.split("_")[2][:4] == 'BQSA'):
        if  MHML_SCP(id,
                params['instance_file'],
                params['instance_dir'],
                params['population'],
                params['maxIter'],
                params['discretizationScheme'],
                params['ql_alpha'],
                params['ql_gamma'],
                params['repair'],
                params['policy'],
                params['rewardType'],
                params['qlAlphaType'],
                params['MH'],
                params['ML'],
                params['paramsML']
                ) == True:
            print(f'Ejecución {id} completada ')

    # if (algorithm.split("_")[1][:2] == 'RW') and algorithm.split("_")[2][:2] != 'QL' and algorithm.split("_")[2][:2] != 'SA' and algorithm.split("_")[2][:4] != 'BQSA':
    #     if  MH_RW(id,
    #             params['instance_file'],
    #             params['population'],
    #             params['maxIter'],
    #             params['discretizationScheme'],
    #             params['beta_Dis'],
    #             params['FO'],
    #             params['MH']

    #             ) == True:
    #         print(f'Ejecución {id} completada ')

    # if (algorithm.split("_")[1][:2] == 'RW') and (algorithm.split("_")[2][:2] == 'QL' or algorithm.split("_")[2][:2] == 'SA' or algorithm.split("_")[2][:4] == 'BQSA'):
    #     if  MHML_RW(id,
    #             params['instance_file'],
    #             params['population'],
    #             params['maxIter'],
    #             params['discretizationScheme'],
    #             params['beta_Dis'],
    #             params['ql_alpha'],
    #             params['ql_gamma'],
    #             params['policy'],
    #             params['rewardType'],
    #             params['qlAlphaType'],
    #             params['FO'],
    #             params['MH'],
    #             params['ML'],
    #             params['paramsML']

    #             ) == True:
    #         print(f'Ejecución {id} completada ')


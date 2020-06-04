from joblib import Parallel, delayed
from joblib import parallel_backend

from itertools import starmap, product
import multiprocessing

import numpy as np
import random
from math import log2, cos, erf
from numpy.random import rand

import time

from collections import namedtuple


def hardfunction(d, n):
    temp = 0
    for i in range(n):
        temp = (temp + d**0.57 + log2(abs(d) ) + erf(d) + cos(d)) % 10000
    return temp
    
hard = 3
def plusalgo(dataset, g, h)     :    return sum( [     hardfunction(d, hard) for d in dataset      ]           ) * (1 + h*rand()/(1+g))
def timesalgo(dataset, g, h)    :    return 2**(sum( [    hardfunction(d, hard) for d in dataset   ]     ) % 30) * (1 + h*rand()/(1+g))
def minusalgo(dataset, g, h)    :    return sum([   hardfunction(d, hard)  for d in  dataset[::2]  ]           ) * (1 + h*rand()/(1+g))

algo_dict = {    'plus':  plusalgo ,
                 'times': timesalgo,
                 'minus': minusalgo     }

dataset_dict =  {    'smallnums' :   [5, 2, 4, 3, 1     ]  *300000   , 
                     'mednums'   :   [14, 13, 12, 11, 15]  *300000   , 
                     'bignums'   :   [99, 62, 73, 85, 51]  *300000       }
                     
hyperp_dict =  {  'h' : [0.5, 0.1],      'g' : [3,6,9]     }

def pmetr(result): return round(abs(result)**0.01   , 5)
def lmetr(result): return round(    log2(abs(log2(abs(log2(abs(result) ) ) ) ) )   , 5)  

metrics_dict = {    'p-met'   :   pmetr    ,
                    'l-met'   :   lmetr         }
                    
                    
                    
def experiment_fn(dataset, algorithm, hparams, metrics_dict):
    print(".", end="")
    result = algorithm(dataset=dataset, **hparams)
    return {n: m( result ) for n, m in metrics_dict.items() }                    


def run_experiments(algo_dict, dataset_dict, metrics_dict, hyperp_dict, experiment_fn , n_jobs=16, rchoice_hparam = -1, rchoice_tot = -1, verbose=False):
    
        '''
    Runs experiments in parallel using joblib
    
    PARAMETERS
    
    algo_dict     :  Dictionary of algorithms
    dataset_dict  :  Dictionary of datasets
    metrics_dict  :  Dictionary of metrics
    hyperp_dict   :  Dictionary of hyperparams
    experiment_fn :  Function that runs a single experiment, given a dataset, algorithm and dictionary of hyperparameter values. 
                     The recommended syntax is something like this, though it will vary depending on how the metric is computed. 
    
                     def experiment_fn(dataset, algorithm, hparams, metrics_dict):
                         result = algorithm(dataset=dataset, **hparams)
                         return {n: m( result ) for n, m in metrics_dict.items() }  
    
    n_jobs: max number of processes to spawn, default 16
    
    rchoice_hparam: randomly choose up to this many hyperparameter sets. 
                    Default is -1, which indicates using all sets of hyperparameters to make experiments
                    
    rchoice_tot:    randomly choose up to this many experiments to run. 
                    Default is -1, which indicates running all experiments
    
    '''

    hyperp_settings_list = [   dict(  zip(  hyperp_dict.keys() ,  hparam_tuple  ) )  for    hparam_tuple  in  product(*hyperp_dict.values() )     ]
    
    if  0 < rchoice_hparam < len(hyperp_settings_list) :    hyperp_settings_list = random.sample(hyperp_settings_list, rchoice_hparam)
        
    experi_names_list =      [   dict(  zip(  ['dataset', 'algorithm', 'hparams'] ,  exp_tuple  ) )  
                                 for   exp_tuple  in  product( dataset_dict.keys(), algo_dict.keys(), hyperp_settings_list  )       ]
    
    if  0 < rchoice_tot < len(experi_names_list) :    experi_names_list = random.sample(experi_names_list, rchoice_tot)
    
    if verbose: print(    f"Running {len(experi_names_list)} experiments"    )
    
    experi_settings_list = [   { 'dataset'      :  dataset_dict[setting_n['dataset']]      ,   
                                 'algorithm'    :  algo_dict[setting_n['algorithm']]       , 
                                 'hparams'      :  setting_n['hparams']                    ,
                                 'metrics_dict' :  metrics_dict                                } 
                            
                               for setting_n in experi_names_list                                       ]
    
    start_t = time.time()
    
    with parallel_backend('multiprocessing', n_jobs=n_jobs):
        results = Parallel()(delayed(experiment_fn)(**setting) for setting in experi_settings_list)
    
    end_t = time.time()
    if verbose: print("\n%.2f seconds elapsed \n" % (end_t - start_t) )
        
    results_w_settings_list = [  {'setting': s, 'result' : r} for s, r in zip(experi_names_list, results) ]
        
    return results_w_settings_list


run_experiments(algo_dict, dataset_dict, metrics_dict, hyperp_dict, experiment_fn, rchoice_tot = -1  )





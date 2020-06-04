from joblib import Parallel, delayed
from joblib import parallel_backend

from itertools import product
from functools import cmp_to_key
from more_itertools import unique_everseen

import random

import time


def run_experiments(algo_dict, dataset_dict, metrics_dict, hyperp_dict, experiment_fn , n_jobs=16, rchoice_hparam = -1, rchoice_tot = -1, verbose=True, is_sorted='asc'):
    
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
                    
    verbose: verbosity
    is_sorted: sort results by the first metric given, default 'asc' for descending. Possible values: False, 'asc', 'desc'
    '''

    # Get a list of all possible hyperparameter settings 
    hyperp_settings_list = [   dict(  zip(  hyperp_dict.keys() ,  hparam_tuple  ) )  for    hparam_tuple  in  product(*hyperp_dict.values() )     ]
    
    if  0 < rchoice_hparam < len(hyperp_settings_list) :    hyperp_settings_list = random.sample(hyperp_settings_list, rchoice_hparam)
    
    # Get a list of all possible experiments 
    experi_names_list =      [   dict(  zip(  ['dataset', 'algorithm', 'hparams'] ,  exp_tuple  ) )  
                                 for   exp_tuple  in  product( dataset_dict.keys(), algo_dict.keys(), hyperp_settings_list  )      ]
    
    # Here we remove hyperparameter names/values if the algorithm being used doesn't have them as parameters
    for experi_name in experi_names_list:
        required_hparams_this_experiment = algo_dict[experi_name['algorithm']].__code__.co_varnames
        filtered_hparams_this_experiment = {hpname:hpval for (hpname, hpval) in experi_name['hparams'].items() if hpname in required_hparams_this_experiment }
        experi_name['hparams'] = filtered_hparams_this_experiment
    
    # remove dupliicate experiments that have been created by dropping unneeded hyperparameters
    experi_names_list = list(unique_everseen(experi_names_list))  
    
    if  0 < rchoice_tot < len(experi_names_list) :    experi_names_list = random.sample(experi_names_list, rchoice_tot)
    
    if verbose: print(    f"Running {len(experi_names_list)} experiments"    )
    
    # convert the names into actual objects for experiments
    experi_settings_list = [   { 'dataset'      :  dataset_dict[setting_n['dataset']]      ,   
                                 'algorithm'    :  algo_dict[setting_n['algorithm']]       , 
                                 'hparams'      :  setting_n['hparams']                    ,
                                 'metrics_dict' :  metrics_dict                                } 
                            
                               for setting_n in experi_names_list                                 ]
    
    start_t = time.time()
    
    ################################################################################################
    # run all the experiments in parallel with joblib 
    with parallel_backend('multiprocessing', n_jobs=n_jobs):
        results = Parallel()(delayed(experiment_fn)(**setting) for setting in experi_settings_list)
    ################################################################################################
    
    end_t = time.time()
    if verbose: print("\n%.2f seconds elapsed \n" % (end_t - start_t) )
        
    results_w_settings_list = [  {'setting': s, 'result' : r} for s, r in zip(experi_names_list, results) ]
    
    if is_sorted == 'asc' or is_sorted == 'desc': 
        
        first_metric = list(metrics_dict.keys())[0]
        
        def compare_fn(item1, item2):
            return (-1 if is_sorted == 'desc' else 1)*(item1['result'][first_metric]  - item2['result'][first_metric] )

        results_w_settings_list = sorted(results_w_settings_list , key=cmp_to_key(compare_fn))
        
        
    return results_w_settings_list




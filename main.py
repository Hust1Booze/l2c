#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:24:33 2021

@author: abdel
"""


#Test different nodesels under different problems



from pathlib import Path 
from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorEstimator, OracleNodeSelectorAbdel
from node_selection.recorders import LPFeatureRecorder, CompFeaturizer
import pyscipopt.scip as sp
import numpy as np
import torch
import multiprocessing as md
from functools import partial



#take a list of nodeselectors to evaluate, a list of instance to test on, and the 
#problem type for printing purposes
def record_stats(nodesels, instances, problem):
    
    nodesels_record = dict((nodesel, []) for nodesel in nodesels)
    model = sp.Model()
    model.hideOutput()
    
    comp_featurizer = CompFeaturizer()
    oracle_estimator = OracleNodeSelectorEstimator(problem, comp_featurizer)
    
    oracle = OracleNodeSelectorAbdel("optimal_plunger")
    
    model.includeNodesel(oracle_estimator, "oracle_estimator", 'testing',100, 100)
    model.includeNodesel(oracle, "oracle", 'testing',100, 100)
    for instance in instances:
        
        instance = str(instance)
        model.readProblem(instance)
        
        oracle_estimator.set_LP_feature_recorder(LPFeatureRecorder(model.getVars(),
                                                                   model.getConss()))
        optsol = model.readSolFile(instance.replace(".lp", ".sol"))
        oracle.setOptsol(optsol)
        print("----------------------------")
        print(f" {problem}  {instance.split('/')[-1].split('.lp')[0] } ")
       #test nodesels
        for nodesel in nodesels:
            
            model.freeTransform()
            model.readProblem(instance)
            
            #canceL all otheer nodesels, WORKS
            for other in nodesels:
                    model.setNodeselPriority(other, 100)
                    
            #activate this nodesel, WORKS
            model.setNodeselPriority(nodesel, 536870911)
            
            model.optimize()
            print(f"  Nodeselector : {nodesel}")
            print(f"    # of processed nodes : {model.getNNodes()} \n")
            print(f"    Time                 : {model.getSolvingTime()} \n")
            if nodesel == "oracle_estimator":
                print(f"fe time : {oracle_estimator.fe_time}")
                print(f"inference time : {oracle_estimator.inference_time}")
            
                
            with open(f"nnodes_{problem}_{nodesel}.csv", "a+") as f:
                f.write(f"{model.getNNodes()},")
                f.close()
            with open(f"times_{problem}_{nodesel}.csv", "a+") as f:
                f.write(f"{model.getSolvingTime()},")
                f.close()
            

    return nodesels_record, oracle_estimator.decision




def display_stats(nodesels, problem):
   
   print("========================")    
   print(f'{problem}') 
   for nodesel in nodesels:
        nnodes = np.genfromtxt(f"nnodes_{problem}_{nodesel}.csv", delimiter=",")[:-1]
        times = np.genfromtxt(f"times_{problem}_{nodesel}.csv", delimiter=",")[:-1]
        print(f"  {nodesel} ")
        print(f"    Mean number of node created   : {np.mean(nnodes):.2f}")
        print(f"    Mean solving time             : {np.mean(times):.2f}")
        print(f"    Median number of node created : {np.median(nnodes):.2f}")
        print(f"    Median solving time           : {np.median(times):.2f}")
        print("--------------------------")
   

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_count = 1
problems = ["GISP"]
nodesels = ["oracle", "oracle_estimator", "dfs", "bfs", "estimate"] 

for problem in problems:
    
    #clear records
    for nodesel in nodesels:
        with open(f"nnodes_{problem}_{nodesel}.csv", "w") as f:
            f.write("")
            f.close()
        with open(f"times_{problem}_{nodesel}.csv", "w") as f:
            f.write("")
            f.close()
        
    instances = list(Path(f"./problem_generation/data/{problem}/train").glob("*.lp"))[:10]
        
    if cpu_count == 1:
        record_stats(nodesels, instances, problem)
    else:
        chunck_size = int(np.ceil(len(instances)/cpu_count))
        processes = [  md.Process(name=f"worker {p}", 
                                        target=partial(record_stats,
                                                        instances=instances[ p*chunck_size : (p+1)*chunck_size], 
                                                        problem=problem,
                                                        nodesels=nodesels))
                        for p in range(cpu_count) ]
            
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
    
    print("SUMMARIES")
    display_stats(nodesels, problem)






"""
GISP
{'oracle': [(21980, 536.88903),
  (8485, 172.335328),
  (4880, 106.475179),
  (3868, 75.894912),
  (3107, 52.380571),
  (6803, 100.78423)],
 'estimate': [(29442, 481.002816),
  (9990, 154.707946),
  (8920, 131.88883),
  (6870, 82.019173),
  (5031, 62.321967),
  (8453, 117.592189)],
 'dfs': [(25974, 383.803651),
  (16747, 188.050372),
  (7569, 92.087239),
  (7759, 73.404249),
  (6688, 59.423487),
  (9597, 103.478273)]}
"""
   


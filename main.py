#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:24:33 2021

@author: abdel
"""


#Test different nodesels under different problems



from pathlib import Path 
from node_selection.node_selectors.oracle_selectors import OracleNodeSelectorEstimator
from node_selection.recorders import LPFeatureRecorder, CompFeaturizer
import pyscipopt.scip as sp
import numpy as np
import torch


#take a list of nodeselectors to evaluate, a list of instance to test on, and the 
#problem type for printing purposes
def get_stats(nodesels, instances, problem):
    
    nodesels_record = dict((nodesel, []) for nodesel in nodesels)
    model = sp.Model()
    model.hideOutput()
    comp_featurizer = CompFeaturizer()
    oracle_estimator = OracleNodeSelectorEstimator(problem, comp_featurizer)
    model.includeNodesel(oracle_estimator, nodesels[0], 'testing',100, 100)
    
    for instance in instances:
        
        instance = str(instance)
        model.readProblem(instance)
        
        oracle_estimator.set_LP_feature_recorder(LPFeatureRecorder(model.getVars(),
                                                                   model.getConss()))
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
                
            nodesels_record[nodesel].append((model.getNNodes(), model.getSolvingTime()))

    return nodesels_record, oracle_estimator.decision




def display_stats(nodesels_record, problem):
    
    for k in nodesels_record:
        nnode_mean, time_mean = np.mean(nodesels_record[k], axis=0)
        nnode_med, time_med = np.median(nodesels_record[k], axis=0)
        print(f"Problem {problem}")
        print( k + f"\n \t Means : NNode {int(nnode_mean)}, time {int(time_mean)}" + 
              f"\n \t Medians : NNodes {int(nnode_med)}, time {int(time_med)}" )
        


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
problems = ["GISP"]
nodesels = ["oracle_estimator", "dfs", "bfs", "estimate"] #always start with oracle_estimator

import matplotlib.pyplot as plt

for problem in problems:
    instances = Path(f"./problem_generation/data/{problem}/test").glob("*.lp")
    stats, decisions = get_stats(nodesels,instances, problem)
    display_stats(stats, problem)
    plt.hist(decisions)
    plt.title(f"{problem}")
    plt.savefig(f'{problem}.png')
    
    





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
   


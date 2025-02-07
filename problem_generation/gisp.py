import os
import sys
import networkx as nx
import random
import pyscipopt as sp
import numpy as np
import multiprocessing as md
from functools import partial
import imp


def dimacsToNx(filename):
    g = nx.Graph()
    with open(filename, 'r') as f:
        for line in f:
            arr = line.split()
            if line[0] == 'e':
                g.add_edge(int(arr[1]), int(arr[2]))
    return g


def generateRevsCosts(g, whichSet, setParam):
    if whichSet == 'SET1':
        for node in g.nodes():
            g.nodes[node]['revenue'] = random.randint(1, 100)
        for u, v, edge in g.edges(data=True):
            edge['cost'] = (g.node[u]['revenue'] /
                            + g.node[v]['revenue'])/float(setParam)
    elif whichSet == 'SET2':
        for node in g.nodes():
            g.nodes[node]['revenue'] = float(setParam)
        for u, v, edge in g.edges(data=True):
            edge['cost'] = 1.0


def generateE2(g, alphaE2):
    E2 = set()
    for edge in g.edges():
        if random.random() <= alphaE2:
            E2.add(edge)
    return E2


def createIP(g, E2, ipfilename):
    with open(ipfilename, 'w') as lp_file:
        val = 100
        lp_file.write("maximize\nOBJ:")
        lp_file.write("100x0")
        count = 0
        for node in g.nodes():
            if count:
                lp_file.write(" + " + str(val) + "x" + str(node))
            count += 1
        for edge in E2:
            lp_file.write(" - y" + str(edge[0]) + '_' + str(edge[1]))
        lp_file.write("\n Subject to\n")
        constraint_count = 1
        for node1, node2, edge in g.edges(data=True):
            if (node1, node2) in E2:
                lp_file.write("C" + str(constraint_count) + ": x" + str(node1)
                              + "+x" + str(node2) + "-y" + str(node1) + "_"
                              + str(node2) + " <=1 \n")
            else:
                lp_file.write("C" + str(constraint_count) + ": x" + str(node1)
                              + "+" + "x" + str(node2) + " <=1 \n")
            constraint_count += 1

        lp_file.write("\nbinary\n")
        for node in g.nodes():
            lp_file.write(f"x{node}\n")
            
def generate_instances(seed_start, seed_end, whichSet, setParam, alphaE2, min_n, max_n, er_prob, instance, lp_dir, solve) :
    
    for seed in range(seed_start, seed_end):
         
        random.seed(seed)
        if instance is None:
            # Generate random graph
            numnodes = random.randint(min_n, max_n)
            g = nx.erdos_renyi_graph(n=numnodes, p=er_prob, seed=seed)
            lpname = ("er_n=%d_m=%d_p=%.2f_%s_setparam=%.2f_alpha=%.2f_%d"
                    % (numnodes, nx.number_of_edges(g), er_prob, whichSet,
                        setParam, alphaE2, seed))
        else:
            g = dimacsToNx(instance)
            # instanceName = os.path.splitext(instance)[1]
            instanceName = instance.split('/')[-1]
            lpname = ("%s_%s_%g_%g_%d" % (instanceName, whichSet, alphaE2,
                    setParam, seed))
        
        # Generate node revenues and edge costs
        generateRevsCosts(g, whichSet, setParam)
        # Generate the set of removable edges
        E2 = generateE2(g, alphaE2)
        # Create IP, write it to file, and solve it with CPLEX
        #print(lpname)
        # ip = createIP(g, E2, lp_dir + "/" + lpname)
        createIP(g, E2, lp_dir + "/" + lpname + ".lp")
        if solve:
            model = sp.Model()
            model.hideOutput()
            model.readProblem(lp_dir +"/" + lpname + ".lp")
            model.optimize()
            model.writeBestSol(lp_dir +"/" + lpname + ".sol")
        

def distribute(n_instance, n_cpu):
    if n_cpu == 1:
        return [(0, n_instance)]
    
    k = n_instance //( n_cpu -1 )
    r = n_instance % (n_cpu - 1 )
    res = []
    for i in range(n_cpu -1):
        res.append( ((k*i), (k*(i+1))) )
    
    res.append(((n_cpu - 1) *k ,(n_cpu - 1) *k + r ))
    return res


if __name__ == "__main__":
    instance = None
    n_cpu = 16
    exp_dir = "data/GISP/"
    data_partition = None
    min_n = 60
    max_n = 70
    er_prob = 0.6
    whichSet = 'SET2'
    setParam = 100.0
    alphaE2 = 0.5
    timelimit = 7200.0
    solveInstance = True
    n_instance = 1000
    seed = 0
    data_partition = 'train'
    

    # seed = 0
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-instance':
            instance = sys.argv[i + 1]
        if sys.argv[i] == '-data_partition':
            data_partition = sys.argv[i + 1]
        if sys.argv[i] == '-min_n':
            min_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-max_n':
            max_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-er_prob':
            er_prob = float(sys.argv[i + 1])
        if sys.argv[i] == '-whichSet':
            whichSet = sys.argv[i + 1]
        if sys.argv[i] == '-setParam':
            setParam = float(sys.argv[i + 1])
        if sys.argv[i] == '-alphaE2':
            alphaE2 = float(sys.argv[i + 1])
        if sys.argv[i] == '-timelimit':
            timelimit = float(sys.argv[i + 1])
        if sys.argv[i] == '-solve':
            solveInstance = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-seed_start':
            seed = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
    
    assert exp_dir is not None
    if instance is None:
        assert min_n is not None
        assert max_n is not None
    
    exp_dir = exp_dir + data_partition
    lp_dir= os.path.join(os.path.dirname(__file__), exp_dir)
    try:
        os.makedirs(lp_dir)
    except FileExistsError:
        ""
    
    print("Summary for GISP generation")
    print(f"n_instance    :     {n_instance}")
    print(f"size interval :     {min_n, max_n}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    print(f"saving dir    :     {lp_dir}")
    
        
            
    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu
    

    
    processes = [  md.Process(name=f"worker {p}", target=partial(generate_instances,
                                                                  seed + p1, 
                                                                  seed + p2, 
                                                                  whichSet, 
                                                                  setParam, 
                                                                  alphaE2, 
                                                                  min_n, 
                                                                  max_n, 
                                                                  er_prob, 
                                                                  instance, 
                                                                  lp_dir, 
                                                                  solveInstance))
                 for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]
    
 
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    print('Generated')
 
    
            
        

